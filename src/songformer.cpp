#include "songformer.h"
#include "graph_utils.h"
#include "compute.h"

#include <cmath>
#include <cstring>
#include <numeric>

bool songformer_model::set_variant(const std::string & name) {
    if (name == "songformer") {
        ds_stride = 3;
        frame_rates = 8.333f;
    } else if (name == "edmformer") {
        ds_stride = 2;
        frame_rates = 12.5f;
    } else {
        return false;
    }
    variant = name;
    return true;
}

bool songformer_model::load(const std::string & path) {
    if (!m.load(path)) return false;

    variant       = m.kv_str("sf.variant", "edmformer");
    input_dim_raw = (int) m.kv_u32("sf.input_dim_raw", 4096);
    input_dim     = (int) m.kv_u32("sf.input_dim", 2048);
    enc_input_dim = (int) m.kv_u32("sf.enc_input_dim", 1024);
    dim           = (int) m.kv_u32("sf.dim", 512);
    n_layers      = (int) m.kv_u32("sf.n_layers", 4);
    n_heads       = (int) m.kv_u32("sf.n_heads", 8);
    rot_dim       = (int) m.kv_u32("sf.rot_dim", 32);
    num_classes   = (int) m.kv_u32("sf.num_classes", 128);
    dataset_id    = (int) m.kv_u32("sf.dataset_id", 5);
    ds_kernel     = (int) m.kv_u32("sf.ds_kernel", 3);
    ds_stride     = (int) m.kv_u32("sf.ds_stride", 2);
    frame_rates   = m.kv_f32("sf.frame_rates", 12.5f);
    rope_base     = m.kv_f32("sf.rope_base", 10000.f);
    local_maxima_filter_size = (int) m.kv_u32("sf.local_maxima_filter_size", 3);
    peak_window_sec = m.kv_f32("sf.peak_window_sec", 12.f);

    allowed_ids    = m.kv_arr_i32("sf.allowed_ids");
    allowed_labels = m.kv_arr_str("sf.allowed_labels");
    allowed_mask.assign(num_classes, false);
    for (int32_t id : allowed_ids) allowed_mask[id] = true;

    // x-transformers stores gamma with unit offset: effective scale = gamma + 1.
    // Patch the weights once at load time (via tensor get/set: weights may live
    // in GPU memory).
    for (ggml_tensor * t = ggml_get_first_tensor(m.ctx); t; t = ggml_get_next_tensor(m.ctx, t)) {
        const char * name = ggml_get_name(t);
        const size_t len = strlen(name);
        if (len > 6 && strcmp(name + len - 6, ".gamma") == 0) {
            std::vector<float> d = m.to_host_f32(t);
            for (float & v : d) v += 1.0f;
            ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
        }
    }
    return true;
}

int64_t songformer_model::infer(const float * fused, int64_t T, int n_threads,
                                std::vector<float> & boundary, std::vector<float> & function) const {
    const int hd = dim / n_heads;  // 64
    const float ln_eps = 1e-5f;

    const size_t meta_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false);
    ggml_init_params ip = { meta_size, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim_raw, T);
    ggml_set_input(inp);

    // mixed_win_downsample + input_norm
    ggml_tensor * x = gg_linear(ctx, m.get("mixed_win_downsample.weight"), m.get("mixed_win_downsample.bias"), inp);
    x = gg_layer_norm(ctx, x, m.get("input_norm.weight"), m.get("input_norm.bias"), ln_eps);  // [2048, T]

    // ---- TimeDownsample ----
    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));  // [T, 2048]
    // depthwise conv k=3 s=2 p=0
    ggml_tensor * xc = ggml_reshape_4d(ctx, xt, T, 1, input_dim, 1);
    xc = ggml_conv_2d_dw_direct(ctx, m.get("down_sample_conv.depthwise_conv.weight"), xc,
                                ds_stride, 1, 0, 0, 1, 1);   // [T2, 1, 2048, 1]
    const int64_t T2 = xc->ne[0];
    xc = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, xc, T2, input_dim)));  // [2048, T2]
    xc = ggml_mul_mat(ctx, m.get("down_sample_conv.pointwise_conv.weight"), xc);        // [1024, T2]
    // residual: AvgPool1d(k,s) + 1x1 conv
    ggml_tensor * res = ggml_pool_1d(ctx, xt, GGML_OP_POOL_AVG, ds_kernel, ds_stride, 0);  // [T2, 2048]
    res = ggml_cont(ctx, ggml_transpose(ctx, res));                                        // [2048, T2]
    res = ggml_mul_mat(ctx, m.get("down_sample_conv.residual_conv.weight"), res);          // [1024, T2]
    x = ggml_add(ctx, xc, res);
    x = gg_layer_norm(ctx, x, m.get("down_sample_conv.norm1.weight"), m.get("down_sample_conv.norm1.bias"), ln_eps);
    x = ggml_gelu_erf(ctx, x);  // [1024, T2]

    // ---- dataset embedding (single id, broadcast over time) ----
    {
        ggml_tensor * emb = m.get("dataset_class_prefix.weight");  // ggml [1024, 64]
        ggml_tensor * row = ggml_view_2d(ctx, emb, enc_input_dim, 1, emb->nb[1],
                                         (size_t) dataset_id * emb->nb[1]);
        x = ggml_add(ctx, x, row);
    }

    // ---- input projection ----
    x = gg_linear(ctx, m.get("transformer.input_proj.0.weight"), m.get("transformer.input_proj.0.bias"), x);
    x = gg_layer_norm(ctx, x, m.get("transformer.input_proj.1.weight"), m.get("transformer.input_proj.1.bias"), ln_eps);
    x = ggml_gelu_erf(ctx, x);
    x = gg_linear(ctx, m.get("transformer.input_proj.4.weight"), m.get("transformer.input_proj.4.bias"), x);  // [512, T2]

    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T2);
    ggml_set_input(pos);

    const float attn_scale = 1.0f / sqrtf((float) hd);

    // ---- x-transformers Encoder: depth blocks of (attn, ff), pre-norm ----
    for (int b = 0; b < n_layers; b++) {
        const int la = 2 * b;      // attention sub-layer index
        const int lf = 2 * b + 1;  // feed-forward sub-layer index

        // attention
        ggml_tensor * h = gg_layer_norm(ctx, x, m.get("transformer.transformer.layers.%d.0.0.gamma", la), nullptr, ln_eps);
        ggml_tensor * q = ggml_mul_mat(ctx, m.get("transformer.transformer.layers.%d.1.to_q.weight", la), h);
        ggml_tensor * k = ggml_mul_mat(ctx, m.get("transformer.transformer.layers.%d.1.to_k.weight", la), h);
        ggml_tensor * v = ggml_mul_mat(ctx, m.get("transformer.transformer.layers.%d.1.to_v.weight", la), h);

        ggml_tensor * q3 = ggml_reshape_3d(ctx, q, hd, n_heads, T2);
        ggml_tensor * k3 = ggml_reshape_3d(ctx, k, hd, n_heads, T2);
        // partial rotary, GPT-J interleaved (x-transformers convention)
        q3 = ggml_rope_ext(ctx, q3, pos, nullptr, rot_dim, GGML_ROPE_TYPE_NORMAL, 0,
                           rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_ext(ctx, k3, pos, nullptr, rot_dim, GGML_ROPE_TYPE_NORMAL, 0,
                           rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor * att = gg_attention(ctx, q3, k3, ggml_reshape_3d(ctx, v, hd, n_heads, T2), attn_scale);
        att = ggml_mul_mat(ctx, m.get("transformer.transformer.layers.%d.1.to_out.weight", la), att);
        x = ggml_add(ctx, x, att);

        // feed-forward
        h = gg_layer_norm(ctx, x, m.get("transformer.transformer.layers.%d.0.0.gamma", lf), nullptr, ln_eps);
        h = gg_linear(ctx, m.get("transformer.transformer.layers.%d.1.ff.0.0.weight", lf),
                      m.get("transformer.transformer.layers.%d.1.ff.0.0.bias", lf), h);
        h = ggml_gelu_erf(ctx, h);
        h = gg_linear(ctx, m.get("transformer.transformer.layers.%d.1.ff.2.weight", lf),
                      m.get("transformer.transformer.layers.%d.1.ff.2.bias", lf), h);
        x = ggml_add(ctx, x, h);
    }

    x = gg_layer_norm(ctx, x, m.get("transformer.transformer.final_norm.gamma"), nullptr, ln_eps);

    ggml_tensor * bnd = gg_linear(ctx, m.get("boundary_head.net.0.weight"), m.get("boundary_head.net.0.bias"), x);  // [1, T2]
    ggml_tensor * fun = gg_linear(ctx, m.get("function_head.net.0.weight"), m.get("function_head.net.0.bias"), x);  // [128, T2]
    ggml_set_output(bnd);
    ggml_set_output(fun);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, bnd);
    ggml_build_forward_expand(gf, fun);

    graph_runner runner(4096, n_threads);
    if (!runner.alloc(gf)) abort();

    ggml_backend_tensor_set(inp, fused, 0, (size_t) T * input_dim_raw * sizeof(float));
    std::vector<int32_t> positions(T2);
    std::iota(positions.begin(), positions.end(), 0);
    ggml_backend_tensor_set(pos, positions.data(), 0, T2 * sizeof(int32_t));

    if (!runner.compute(gf)) abort();

    boundary.resize(T2);
    ggml_backend_tensor_get(bnd, boundary.data(), 0, T2 * sizeof(float));
    function.resize((size_t) T2 * num_classes);
    ggml_backend_tensor_get(fun, function.data(), 0, function.size() * sizeof(float));

    // label-id mask: disallowed classes -> -inf
    for (int64_t t = 0; t < T2; t++) {
        float * row = function.data() + t * num_classes;
        for (int c = 0; c < num_classes; c++) {
            if (!allowed_mask[c]) row[c] = -INFINITY;
        }
    }

    ggml_free(ctx);
    return T2;
}
