#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using msa_info = std::vector<std::pair<double, std::string>>;

struct postprocess_params {
    float frame_rates = 12.5f;
    int   local_maxima_filter_size = 3;
    float peak_window_sec = 12.f;
    int   num_classes = 128;
    // id -> label name for the classes that can win (others are -inf anyway)
    std::vector<int32_t>     allowed_ids;
    std::vector<std::string> allowed_labels;
};

// port of postprocess_functional_structure (boundary sigmoid + local maxima +
// peak picking; per-segment mean-softmax label)
msa_info postprocess_functional_structure(const std::vector<float> & boundary_logits,
                                          const std::vector<float> & function_logits,  // [T][C]
                                          const postprocess_params & p);

// port of app.rule_post_processing
msa_info rule_post_processing(const msa_info & in);

std::string format_msa(const msa_info & msa);
std::string format_json(const msa_info & msa);
