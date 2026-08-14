// parity tests against fixtures dumped from the Python pipeline
//
//   edmformer-test <fixtures_dir> <models_dir> [stage]
//
// stages: postprocess, songformer, mel, musicfm, muq, fused, all

#include "common.h"
#include "pipeline.h"
#include "postprocess.h"
#include "songformer.h"
#include "ssl_conformer.h"
#include "compute.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

static int n_threads = (int) std::thread::hardware_concurrency();
static std::string fix_dir, model_dir;
static int n_fail = 0;

static bool check(const char * name, float diff, float tol) {
    const bool ok = diff <= tol;
    printf("%-28s max|diff| = %.6f  (tol %.4f)  %s\n", name, diff, tol, ok ? "OK" : "FAIL");
    if (!ok) n_fail++;
    return ok;
}

static npy_array fx(const char * name) {
    npy_array a;
    if (!npy_load(fix_dir + "/" + name, a)) {
        fprintf(stderr, "missing fixture %s\n", name);
        exit(1);
    }
    return a;
}

static void test_postprocess(const songformer_model & sf) {
    npy_array bnd = fx("boundary_logits.npy");
    npy_array fun = fx("function_logits.npy");

    song_logits lg;
    lg.boundary = bnd.data;
    lg.function = fun.data;
    msa_info msa = rule_post_processing(logits_to_msa(lg, sf));

    FILE * f = fopen((fix_dir + "/segments_ref.txt").c_str(), "rb");
    std::string ref;
    char c;
    while (f && fread(&c, 1, 1, f) == 1) ref += c;
    if (f) fclose(f);

    const std::string got = format_msa(msa);
    const bool ok = got == ref;
    printf("%-28s %s\n", "postprocess segments", ok ? "OK (exact match)" : "FAIL");
    if (!ok) {
        printf("--- expected ---\n%s\n--- got ---\n%s\n", ref.c_str(), got.c_str());
        n_fail++;
    }
}

// full pipeline on the exact audio samples python used; compares accumulated
// logits and the final segment list (slow: runs both SSL models on the song)
static void test_e2e(const ssl_model & musicfm, const ssl_model & muq, const songformer_model & sf) {
    npy_array audio = fx("audio_full.npy");

    song_logits lg = process_song(audio.data, musicfm, muq, sf, n_threads, true);

    npy_array ref_b = fx("boundary_logits.npy");
    npy_array ref_f = fx("function_logits.npy");
    printf("e2e: T=%zu (ref %zu)\n", lg.boundary.size(), ref_b.data.size());
    if (lg.boundary.size() == ref_b.data.size()) {
        check("e2e boundary logits", max_abs_diff(lg.boundary.data(), ref_b.data.data(), lg.boundary.size()), 2e-1f);
        check("e2e function logits", max_abs_diff(lg.function.data(), ref_f.data.data(), lg.function.size()), 2e-1f);
    } else {
        printf("e2e: length mismatch\n");
        n_fail++;
    }

    msa_info msa = rule_post_processing(logits_to_msa(lg, sf));
    FILE * f = fopen((fix_dir + "/segments_ref.txt").c_str(), "rb");
    std::string ref;
    char c;
    while (f && fread(&c, 1, 1, f) == 1) ref += c;
    if (f) fclose(f);
    const std::string got = format_msa(msa);
    const bool ok = got == ref;
    printf("%-28s %s\n", "e2e segments", ok ? "OK (exact match)" : "FAIL");
    if (!ok) {
        printf("--- expected ---\n%s\n--- got ---\n%s\n", ref.c_str(), got.c_str());
        n_fail++;
    }
}

static void test_mel(const ssl_model & musicfm, const ssl_model & muq) {
    npy_array audio = fx("audio_30s.npy");

    std::vector<float> mel;
    const int64_t T = musicfm.mel(audio.data.data(), audio.data.size(), mel);
    // note: reference fixtures are fp32; near spectral nulls the fp32 pipeline
    // itself deviates ~4e-3 from a float64 computation, so tolerance is 1e-2
    npy_array ref = fx("mel_musicfm_30s.npy");
    printf("mel: T=%lld (ref %lld)\n", (long long) T, (long long) ref.shape[1]);
    check("mel musicfm", max_abs_diff(mel.data(), ref.data.data(), mel.size()), 1e-2f);

    std::vector<float> mel2;
    muq.mel(audio.data.data(), audio.data.size(), mel2);
    npy_array ref2 = fx("mel_muq_30s.npy");
    check("mel muq", max_abs_diff(mel2.data(), ref2.data.data(), mel2.size()), 1e-2f);
}

static void test_ssl(const ssl_model & model, const char * name, const char * h10_fix,
                     const char * frontend_fix) {
    npy_array audio = fx("audio_30s.npy");

    std::vector<float> mel;
    const int64_t T_mel = model.mel(audio.data.data(), audio.data.size(), mel);

    std::vector<float> emb, frontend;
    const int64_t T = model.encode_mel(mel.data(), T_mel, n_threads, emb,
                                       frontend_fix ? &frontend : nullptr);

    if (frontend_fix) {
        npy_array ref_fe = fx(frontend_fix);
        char label[64];
        snprintf(label, sizeof(label), "%s frontend", name);
        check(label, max_abs_diff(frontend.data(), ref_fe.data.data(), frontend.size()), 5e-2f);
    }

    npy_array ref = fx(h10_fix);
    printf("%s: T=%lld (ref %lld)\n", name, (long long) T, (long long) ref.shape[0]);

    // relative-ish tolerance: hidden states have values up to ~50
    char label[64];
    snprintf(label, sizeof(label), "%s hidden10", name);
    check(label, max_abs_diff(emb.data(), ref.data.data(), emb.size()), 5e-1f);

    // also report mean abs diff for context
    double acc = 0;
    for (size_t i = 0; i < emb.size(); i++) acc += fabsf(emb[i] - ref.data[i]);
    printf("%-28s mean|diff| = %.6f\n", label, acc / emb.size());
}

static void test_fused(const ssl_model & musicfm, const ssl_model & muq) {
    npy_array audio = fx("audio_30s.npy");

    std::vector<float> em, eq;
    const int64_t tm = musicfm.extract(audio.data.data(), audio.data.size(), n_threads, em);
    const int64_t tq = muq.extract(audio.data.data(), audio.data.size(), n_threads, eq);
    const int64_t L = std::min(tm, tq);

    const int d = musicfm.d_model;
    std::vector<float> fused((size_t) L * 4 * d);
    for (int64_t t = 0; t < L; t++) {
        float * dst = fused.data() + (size_t) t * 4 * d;
        memcpy(dst + 0 * d, em.data() + (size_t) t * d, d * sizeof(float));
        memcpy(dst + 1 * d, eq.data() + (size_t) t * d, d * sizeof(float));
        memcpy(dst + 2 * d, em.data() + (size_t) t * d, d * sizeof(float));
        memcpy(dst + 3 * d, eq.data() + (size_t) t * d, d * sizeof(float));
    }
    npy_array ref = fx("fused_30s.npy");
    check("fused embedding", max_abs_diff(fused.data(), ref.data.data(), fused.size()), 5e-1f);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <fixtures_dir> <models_dir> [stage] [variant]\n", argv[0]);
        fprintf(stderr, "stages: postprocess mel musicfm muq fused all e2e\n");
        fprintf(stderr, "variant: songformer | edmformer (default: GGUF metadata)\n");
        return 1;
    }
    fix_dir = argv[1];
    model_dir = argv[2];
    const std::string stage = argc > 3 ? argv[3] : "all";

    songformer_model sf;
    if (!sf.load(model_dir + "/songformer-f32.gguf")) return 1;
    if (argc > 4 && !sf.set_variant(argv[4])) {
        fprintf(stderr, "unknown variant: %s\n", argv[4]);
        return 1;
    }
    printf("head variant: %s (stride %d, %.3f fps)\n", sf.variant.c_str(), sf.ds_stride, sf.frame_rates);

    if (stage == "postprocess" || stage == "all") test_postprocess(sf);

    ssl_model musicfm, muq;
    bool ssl_loaded = false;
    auto load_ssl = [&]() {
        if (ssl_loaded) return;
        if (!musicfm.load(model_dir + "/musicfm-f16.gguf")) exit(1);
        if (!muq.load(model_dir + "/muq-f16.gguf")) exit(1);
        ssl_loaded = true;
    };

    if (stage == "mel" || stage == "all") { load_ssl(); test_mel(musicfm, muq); }
    if (stage == "musicfm" || stage == "all") {
        load_ssl();
        test_ssl(musicfm, "musicfm", "h10_musicfm_30s.npy", "frontend_musicfm_30s.npy");
    }
    if (stage == "muq" || stage == "all") {
        load_ssl();
        test_ssl(muq, "muq", "h10_muq_30s.npy", nullptr);
    }
    if (stage == "fused" || stage == "all") { load_ssl(); test_fused(musicfm, muq); }
    if (stage == "e2e") { load_ssl(); test_e2e(musicfm, muq, sf); }  // slow; not in 'all'

    compute_profile_report();
    printf("\n%s\n", n_fail == 0 ? "ALL OK" : "FAILURES");
    return n_fail == 0 ? 0 : 1;
}
