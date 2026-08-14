#pragma once

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

// SongFormer MSA head:
//   Linear(4096->2048) -> LayerNorm -> TimeDownsample(->1024)
//   -> +dataset embedding -> input_proj(->512) -> 4x (attn, ff) pre-norm
//      rotary transformer -> boundary head (1) + function head (128)
//
// Two inference variants share identical weights and differ only in the
// TimeDownsample stride (and thus the logit frame rate):
//   "songformer" (upstream): stride 3 -> 8.333 fps
//   "edmformer":             stride 2 -> 12.5  fps
struct songformer_model {
    gguf_model m;

    std::string variant = "edmformer";

    int   input_dim_raw = 4096;
    int   input_dim     = 2048;
    int   enc_input_dim = 1024;
    int   dim           = 512;
    int   n_layers      = 4;
    int   n_heads       = 8;
    int   rot_dim       = 32;
    int   num_classes   = 128;
    int   dataset_id    = 5;
    int   ds_kernel     = 3;
    int   ds_stride     = 2;
    float frame_rates   = 12.5f;
    float rope_base     = 10000.f;
    int   local_maxima_filter_size = 3;
    float peak_window_sec = 12.f;

    std::vector<int32_t>     allowed_ids;
    std::vector<std::string> allowed_labels;
    std::vector<bool>        allowed_mask;  // size num_classes

    bool load(const std::string & path);

    // switch between "songformer" and "edmformer" at runtime (same weights)
    bool set_variant(const std::string & name);

    // fused embeddings [T][4096] (row-major) -> logits at frame_rates
    // boundary: [T_out], function: [T_out][num_classes] with disallowed = -inf
    // returns T_out
    int64_t infer(const float * fused, int64_t T, int n_threads,
                  std::vector<float> & boundary, std::vector<float> & function) const;
};
