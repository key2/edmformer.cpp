#pragma once

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

// MusicFM-25Hz / MuQ-large feature extractor:
//   log-mel -> Conv2dSubsampling (2x Res2d, BN folded) -> N conformer layers
// Returns hidden_states[extract_layer] at 25 Hz, i.e. the output of the first
// `n_layers` (=10) conformer layers, without the final encoder LayerNorm.
struct ssl_model {
    gguf_model m;

    int   n_layers  = 10;
    int   n_heads   = 16;
    int   d_model   = 1024;
    float ln_eps    = 1e-5f;
    float rope_base = 10000.f;
    int   conv_kernel = 31;
    int   n_mels    = 128;
    int   n_fft     = 2048;
    int   hop       = 240;
    float mel_mean  = 0.f;
    float mel_std   = 1.f;

    std::vector<float> fbank;   // [n_mels][n_fft/2+1]
    std::vector<float> window;  // [n_fft]

    bool load(const std::string & path);

    // audio (24 kHz mono) -> normalized log-mel, [n_mels][T_mel] (time contiguous)
    int64_t mel(const float * samples, int64_t n, std::vector<float> & out) const;

    // mel -> Conv2dSubsampling frontend embeddings [1024][T4] (chunked over time)
    int64_t frontend_forward(const float * mel_data, int64_t T_mel, int n_threads,
                             std::vector<float> & fe) const;

    // normalized mel [n_mels][T_mel] -> hidden layer-10 embeddings.
    // out: [T_emb][1024] row-major; returns T_emb (= T_mel/4 conv subsampling)
    int64_t encode_mel(const float * mel_data, int64_t T_mel, int n_threads,
                       std::vector<float> & out,
                       std::vector<float> * frontend_out = nullptr) const;

    // convenience: audio -> embeddings
    int64_t extract(const float * samples, int64_t n, int n_threads, std::vector<float> & out) const;
};
