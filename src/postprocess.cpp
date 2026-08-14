#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

// local_maxima with filter_size window: keep value where it equals the window
// max (window padded with -inf), else 0
static std::vector<float> local_maxima(const std::vector<float> & x, int filter_size) {
    const int64_t n = (int64_t) x.size();
    const int r = filter_size / 2;
    std::vector<float> out(n, 0.f);
    for (int64_t i = 0; i < n; i++) {
        float mx = -INFINITY;
        for (int64_t j = i - r; j <= i + r; j++) {
            if (j >= 0 && j < n) mx = std::max(mx, x[j]);
        }
        if (x[i] == mx) out[i] = x[i];
    }
    return out;
}

// port of helpers.peak_picking (window_past == window_future == W)
static std::vector<float> peak_picking(const std::vector<float> & x, int W) {
    const int64_t n = (int64_t) x.size();
    std::vector<float> strength(n, 0.f);
    for (int64_t i = 0; i < n; i++) {
        if (!(x[i] > 0.f)) continue;
        float mx = 0.f;  // zero padding participates in the max
        for (int64_t j = i - W; j <= i + W; j++) {
            float v = (j >= 0 && j < n) ? x[j] : 0.f;
            mx = std::max(mx, v);
        }
        if (x[i] != mx) continue;
        double past = 0.0, future = 0.0;
        for (int64_t j = i - W; j < i; j++) past += (j >= 0 && j < n) ? x[j] : 0.f;
        for (int64_t j = i + 1; j <= i + W; j++) future += (j >= 0 && j < n) ? x[j] : 0.f;
        past /= W;
        future /= W;
        strength[i] = x[i] - (float) ((past + future) / 2.0);
    }
    return strength;
}

msa_info postprocess_functional_structure(const std::vector<float> & boundary_logits,
                                          const std::vector<float> & function_logits,
                                          const postprocess_params & p) {
    const int64_t T = (int64_t) boundary_logits.size();
    const int C = p.num_classes;

    std::map<int, std::string> id2label;
    for (size_t i = 0; i < p.allowed_ids.size(); i++) {
        id2label[p.allowed_ids[i]] = p.allowed_labels[i];
    }

    // sigmoid boundary probs
    std::vector<float> prob(T);
    for (int64_t t = 0; t < T; t++) prob[t] = 1.f / (1.f + expf(-boundary_logits[t]));

    // softmax over classes per frame (rows [T][C], -inf handled naturally)
    std::vector<float> pfun((size_t) T * C);
    for (int64_t t = 0; t < T; t++) {
        const float * row = function_logits.data() + t * C;
        float mx = -INFINITY;
        for (int c = 0; c < C; c++) mx = std::max(mx, row[c]);
        double sum = 0.0;
        float * dst = pfun.data() + t * C;
        for (int c = 0; c < C; c++) {
            dst[c] = std::isinf(row[c]) && row[c] < 0 ? 0.f : expf(row[c] - mx);
            sum += dst[c];
        }
        for (int c = 0; c < C; c++) dst[c] = (float) (dst[c] / sum);
    }

    std::vector<float> lm = local_maxima(prob, p.local_maxima_filter_size);
    const int W = (int) (p.peak_window_sec * p.frame_rates);
    std::vector<float> strength = peak_picking(lm, W);

    std::vector<int64_t> bidx;  // frames where strength > 0
    for (int64_t t = 0; t < T; t++) {
        if (strength[t] > 0.f) bidx.push_back(t);
    }

    const double duration = (double) T / p.frame_rates;
    std::vector<double> btimes;
    for (int64_t i : bidx) btimes.push_back((double) i / p.frame_rates);
    if (btimes.empty() || btimes.front() != 0.0) btimes.insert(btimes.begin(), 0.0);
    if (btimes.back() != duration) btimes.push_back(duration);

    // segment boundaries in frames (excluding 0) for label pooling
    std::vector<int64_t> splits;
    for (int64_t i : bidx) {
        if (i > 0) splits.push_back(i);
    }

    msa_info segments;
    int64_t seg_start = 0;
    for (size_t s = 0; s <= splits.size(); s++) {
        const int64_t seg_end = s < splits.size() ? splits[s] : T;  // exclusive
        // mean prob over segment frames, argmax class
        int best = 0;
        double best_v = -1.0;
        for (int c = 0; c < C; c++) {
            double acc = 0.0;
            for (int64_t t = seg_start; t < seg_end; t++) acc += pfun[(size_t) t * C + c];
            acc /= std::max<int64_t>(1, seg_end - seg_start);
            if (acc > best_v) {
                best_v = acc;
                best = c;
            }
        }
        const double t0 = btimes[s];
        auto it = id2label.find(best);
        segments.push_back({t0, it != id2label.end() ? it->second : "class_" + std::to_string(best)});
        seg_start = seg_end;
    }
    segments.push_back({btimes.back(), "end"});
    return segments;
}

msa_info rule_post_processing(const msa_info & in) {
    if (in.size() <= 2) return in;
    msa_info r = in;

    while (r.size() > 2) {
        const double first_dur = r[1].first - r[0].first;
        if (first_dur < 1.0) {
            r[0] = {r[0].first, r[1].second};
            r.erase(r.begin() + 1);
        } else {
            break;
        }
    }
    while (r.size() > 2) {
        const double last_dur = r[r.size() - 1].first - r[r.size() - 2].first;
        if (last_dur < 1.0) {
            r.erase(r.end() - 2);
        } else {
            break;
        }
    }
    while (r.size() > 2) {
        if (r[0].second == r[1].second && r[1].first <= 10.0) {
            r.erase(r.begin() + 1);
        } else {
            break;
        }
    }
    while (r.size() > 2) {
        const double last_dur = r[r.size() - 1].first - r[r.size() - 2].first;
        if (r.size() >= 3 && r[r.size() - 2].second == r[r.size() - 3].second && last_dur <= 10.0) {
            r.erase(r.end() - 2);
        } else {
            break;
        }
    }
    return r;
}

std::string format_msa(const msa_info & msa) {
    std::string out;
    char buf[128];
    for (auto & [t, label] : msa) {
        snprintf(buf, sizeof(buf), "%.2f %s\n", t, label.c_str());
        out += buf;
    }
    if (!out.empty()) out.pop_back();
    return out;
}

std::string format_json(const msa_info & msa) {
    std::string out = "[\n";
    char buf[256];
    for (size_t i = 0; i + 1 < msa.size(); i++) {
        snprintf(buf, sizeof(buf),
                 "  {\"start\": %.2f, \"end\": %.2f, \"label\": \"%s\"}%s\n",
                 msa[i].first, msa[i + 1].first, msa[i].second.c_str(),
                 i + 2 < msa.size() ? "," : "");
        out += buf;
    }
    out += "]";
    return out;
}
