#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

// constants mirrored from app.py
static const int SR = 24000;
static const int WIN_SIZE = 420;   // seconds
static const int HOP_SIZE = 420;
// app.py accumulates window logits at 8.333 fps for both variants. This is
// exact for the upstream "songformer" head (stride 3 -> 8.333 fps) and is a
// replicated quirk for the "edmformer" head (stride 2 -> 12.5 fps), where it
// only affects songs longer than one 420 s window.
static const double ACCUM_FRAME_RATES = 8.333;

// concatenate feature-wise: fused[t] = [mfm30 | muq30 | mfm420 | muq420]
static std::vector<float> fuse(const std::vector<float> * e[4], int64_t L, int d) {
    std::vector<float> fused((size_t) L * 4 * d);
    for (int64_t t = 0; t < L; t++) {
        float * dst = fused.data() + (size_t) t * 4 * d;
        for (int i = 0; i < 4; i++) {
            memcpy(dst + (size_t) i * d, e[i]->data() + (size_t) t * d, d * sizeof(float));
        }
    }
    return fused;
}

song_logits process_song(const std::vector<float> & audio, const ssl_model & musicfm,
                         const ssl_model & muq, const songformer_model & sf,
                         int n_threads, bool verbose) {
    const int64_t n = (int64_t) audio.size();
    const int d = musicfm.d_model;

    const int64_t total_len = ((n / SR) / WIN_SIZE) * WIN_SIZE + WIN_SIZE;  // seconds
    const int64_t total_frames = (int64_t) ceil((double) total_len * ACCUM_FRAME_RATES);

    std::vector<float>  fun_sum((size_t) total_frames * sf.num_classes, 0.f);
    std::vector<float>  fun_cnt((size_t) total_frames * sf.num_classes, 0.f);
    std::vector<float>  bnd_val(total_frames, 0.f);
    int64_t lens = 0;

    for (int64_t i = 0; i * SR < n; i += HOP_SIZE) {
        const int64_t s0 = i * SR;
        const int64_t s1 = std::min((i + WIN_SIZE) * SR, n);
        if (s1 - s0 <= 1024) break;

        if (verbose) fprintf(stderr, "window @%llds: extracting 420s embeddings...\n", (long long) i);
        std::vector<float> mfm420, muq420;
        const int64_t T420m = musicfm.extract(audio.data() + s0, s1 - s0, n_threads, mfm420);
        const int64_t T420q = muq.extract(audio.data() + s0, s1 - s0, n_threads, muq420);

        std::vector<float> mfm30, muq30;
        int64_t T30 = 0;
        for (int64_t j = i; j < i + HOP_SIZE; j += 30) {
            const int64_t c0 = j * SR;
            const int64_t c1 = std::min({(j + 30) * SR, n, (i + HOP_SIZE) * SR});
            if (c0 >= n) break;
            if (c1 - c0 <= 1024) continue;
            if (verbose) fprintf(stderr, "window @%llds: 30s chunk @%llds\n", (long long) i, (long long) j);
            std::vector<float> em, eq;
            const int64_t tm = musicfm.extract(audio.data() + c0, c1 - c0, n_threads, em);
            const int64_t tq = muq.extract(audio.data() + c0, c1 - c0, n_threads, eq);
            mfm30.insert(mfm30.end(), em.begin(), em.end());
            muq30.insert(muq30.end(), eq.begin(), eq.end());
            T30 += std::min(tm, tq);
            if (tm != tq) {  // keep time-aligned if chunk lengths ever differ
                mfm30.resize((size_t) T30 * d);
                muq30.resize((size_t) T30 * d);
            }
        }
        if (T30 == 0) continue;

        const int64_t L = std::min(std::min(T30, T30), std::min(T420m, T420q));
        const std::vector<float> * embds[4] = { &mfm30, &muq30, &mfm420, &muq420 };
        std::vector<float> fused = fuse(embds, L, d);

        if (verbose) fprintf(stderr, "window @%llds: songformer head (T=%lld)...\n", (long long) i, (long long) L);
        std::vector<float> bnd, fun;
        const int64_t T2 = sf.infer(fused.data(), L, n_threads, bnd, fun);

        const int64_t start = (int64_t) ((double) i * ACCUM_FRAME_RATES);
        const int64_t cnt = std::min((int64_t) ceil((double) HOP_SIZE * ACCUM_FRAME_RATES), T2);
        for (int64_t t = 0; t < cnt; t++) {
            float * fs = fun_sum.data() + (size_t) (start + t) * sf.num_classes;
            float * fc = fun_cnt.data() + (size_t) (start + t) * sf.num_classes;
            const float * src = fun.data() + (size_t) t * sf.num_classes;
            for (int c = 0; c < sf.num_classes; c++) {
                fs[c] += src[c];
                fc[c] += 1.f;
            }
            bnd_val[start + t] = bnd[t];
        }
        lens += cnt;
    }

    song_logits out;
    out.boundary.assign(bnd_val.begin(), bnd_val.begin() + lens);
    out.function.resize((size_t) lens * sf.num_classes);
    for (int64_t t = 0; t < lens; t++) {
        for (int c = 0; c < sf.num_classes; c++) {
            const float cnt = std::max(1.f, fun_cnt[(size_t) t * sf.num_classes + c]);
            out.function[(size_t) t * sf.num_classes + c] = fun_sum[(size_t) t * sf.num_classes + c] / cnt;
        }
    }
    return out;
}

msa_info logits_to_msa(const song_logits & lg, const songformer_model & sf) {
    postprocess_params p;
    p.frame_rates = sf.frame_rates;
    p.local_maxima_filter_size = sf.local_maxima_filter_size;
    p.peak_window_sec = sf.peak_window_sec;
    p.num_classes = sf.num_classes;
    p.allowed_ids = sf.allowed_ids;
    p.allowed_labels = sf.allowed_labels;
    return postprocess_functional_structure(lg.boundary, lg.function, p);
}
