#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// shared ggml graph building helpers

// LayerNorm over feature dim (ne0). w/b are [C]; b may be null.
static inline ggml_tensor * gg_layer_norm(ggml_context * ctx, ggml_tensor * x,
                                          ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

// y = W x + b ; W stored as ggml [in, out], x [in, T] -> [out, T]; b may be null
static inline ggml_tensor * gg_linear(ggml_context * ctx, ggml_tensor * w,
                                      ggml_tensor * b, ggml_tensor * x) {
    x = ggml_mul_mat(ctx, w, x);
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

// flash attention over full sequence (no mask, batch 1)
// q3/k3/v3: [head_dim, n_heads, T] f32 -> returns [head_dim*n_heads, T]
static inline ggml_tensor * gg_attention(ggml_context * ctx, ggml_tensor * q3,
                                         ggml_tensor * k3, ggml_tensor * v3, float scale) {
    ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, q3, 0, 2, 1, 3));                  // [d, T, H]
    ggml_tensor * k = ggml_cast(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3), GGML_TYPE_F16);   // [d, T, H]
    ggml_tensor * v = ggml_cast(ctx, ggml_permute(ctx, v3, 0, 2, 1, 3), GGML_TYPE_F16);   // [d, T, H]
    ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    return ggml_reshape_2d(ctx, o, o->ne[0] * o->ne[1], o->ne[2]);
}
