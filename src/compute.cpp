#include "compute.h"

#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

// ---- optional per-op profiling (EDMFORMER_PROFILE=1) ----
static std::map<std::string, double> g_prof;
static std::chrono::steady_clock::time_point g_prof_t0;
static bool g_profiling = false;

static bool prof_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) {
        g_prof_t0 = std::chrono::steady_clock::now();
        return true;
    }
    const auto now = std::chrono::steady_clock::now();
    g_prof[ggml_op_desc(t)] += std::chrono::duration<double>(now - g_prof_t0).count();
    g_prof_t0 = now;
    return true;
}

void compute_profile_report() {
    if (!g_profiling || g_prof.empty()) return;
    std::vector<std::pair<double, std::string>> v;
    double total = 0;
    for (auto & [k, s] : g_prof) { v.push_back({s, k}); total += s; }
    std::sort(v.rbegin(), v.rend());
    fprintf(stderr, "---- op profile (total %.2fs) ----\n", total);
    for (auto & [s, k] : v) {
        if (s / total > 0.005) fprintf(stderr, "%-24s %8.2fs  %5.1f%%\n", k.c_str(), s, 100.0 * s / total);
    }
    g_prof.clear();
}

struct compute_env {
    bool initialized = false;
    ggml_backend_t gpu = nullptr;
    ggml_backend_t cpu = nullptr;
    std::vector<ggml_backend_t> backends;  // [gpu?, cpu]
    char gpu_name[128] = "none";
};

static compute_env g_env;

bool compute_init(int n_threads, bool use_gpu) {
    if (g_env.initialized) return true;

    if (use_gpu && getenv("EDMFORMER_CPU") == nullptr) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev) {
            g_env.gpu = ggml_backend_dev_init(dev, nullptr);
            if (g_env.gpu) {
                snprintf(g_env.gpu_name, sizeof(g_env.gpu_name), "%s (%s)",
                         ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
            }
        }
        if (!g_env.gpu) {
            fprintf(stderr, "compute: no GPU backend available, using CPU\n");
        }
    }

    g_env.cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(g_env.cpu, n_threads);

    if (g_env.gpu) g_env.backends.push_back(g_env.gpu);

    // accelerator backends (e.g. BLAS) speed up large f32 matmuls when no GPU
    // is in front of them in the scheduler priority order
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
            if (b) g_env.backends.push_back(b);
        }
    }

    g_env.backends.push_back(g_env.cpu);

    g_env.initialized = true;
    return true;
}

void compute_free() {
    if (!g_env.initialized) return;
    for (ggml_backend_t b : g_env.backends) ggml_backend_free(b);
    g_env = compute_env();
}

bool compute_has_gpu() {
    return g_env.gpu != nullptr;
}

const char * compute_gpu_name() {
    return g_env.gpu_name;
}

ggml_backend_buffer_type_t compute_weight_buft() {
    compute_init(4, true);  // lazy default
    return ggml_backend_get_default_buffer_type(g_env.backends[0]);
}

graph_runner::graph_runner(size_t graph_size, int n_threads) {
    compute_init(n_threads, true);
    sched = ggml_backend_sched_new(g_env.backends.data(), nullptr,
                                   (int) g_env.backends.size(), graph_size,
                                   /*parallel=*/false, /*op_offload=*/true);
    if (getenv("EDMFORMER_PROFILE")) {
        g_profiling = true;
        ggml_backend_sched_set_eval_callback(sched, prof_cb, nullptr);
    }
}

graph_runner::~graph_runner() {
    if (sched) ggml_backend_sched_free(sched);
}

bool graph_runner::alloc(struct ggml_cgraph * gf) {
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "graph_runner: allocation failed\n");
        return false;
    }
    return true;
}

bool graph_runner::compute(struct ggml_cgraph * gf) {
    return ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
}
