#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>

// shared compute environment: GPU (Metal on macOS) primary with CPU fallback,
// scheduled per-graph with ggml_backend_sched.

// initialize explicitly (idempotent). use_gpu=false forces CPU-only.
bool compute_init(int n_threads, bool use_gpu);
void compute_free();

// true when a GPU backend is active
bool compute_has_gpu();
const char * compute_gpu_name();

// buffer type where model weights should live (GPU memory when available)
ggml_backend_buffer_type_t compute_weight_buft();

// allocate + run one graph on the environment (lazy-inits with defaults)
struct graph_runner {
    ggml_backend_sched_t sched = nullptr;

    explicit graph_runner(size_t graph_size, int n_threads);
    graph_runner(const graph_runner &) = delete;
    ~graph_runner();

    bool alloc(struct ggml_cgraph * gf);    // then set inputs via ggml_backend_tensor_set
    bool compute(struct ggml_cgraph * gf);  // synchronous
};

// print accumulated per-op timings when EDMFORMER_PROFILE=1
void compute_profile_report();
