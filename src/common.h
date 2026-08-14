#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// minimal .npy (fp32, C-order) reader/writer for parity fixtures
// ---------------------------------------------------------------------------

struct npy_array {
    std::vector<int64_t> shape;
    std::vector<float>   data;

    int64_t n_elements() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

bool npy_load(const std::string & path, npy_array & out);
bool npy_save(const std::string & path, const float * data, const std::vector<int64_t> & shape);

// ---------------------------------------------------------------------------
// gguf-backed model weights (loaded into CPU memory)
// ---------------------------------------------------------------------------

struct gguf_model {
    struct ggml_context * ctx = nullptr;         // tensor metadata
    struct gguf_context * g   = nullptr;
    struct ggml_backend_buffer * buf = nullptr;  // weight data (GPU when available)

    gguf_model() = default;
    gguf_model(const gguf_model &) = delete;
    gguf_model & operator=(const gguf_model &) = delete;
    ~gguf_model();

    // loads weights into the compute environment's weight buffer type
    bool load(const std::string & path);

    struct ggml_tensor * get(const char * fmt, ...) const;

    // copy a tensor's data to host memory (f32 tensors)
    std::vector<float> to_host_f32(struct ggml_tensor * t) const;

    float       kv_f32(const char * key, float def) const;
    uint32_t    kv_u32(const char * key, uint32_t def) const;
    std::string kv_str(const char * key, const std::string & def) const;
    std::vector<int32_t>     kv_arr_i32(const char * key) const;
    std::vector<std::string> kv_arr_str(const char * key) const;
};

// max |a-b| over n elements
float max_abs_diff(const float * a, const float * b, int64_t n);
