#include "ssl_conformer.h"
#include "graph_utils.h"
#include "compute.h"
#include "mel.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

bool ssl_model::load(const std::string & path) {
    if (!m.load(path)) return false;

    n_layers    = (int) m.kv_u32("edmformer.n_layers", 10);
    n_heads     = (int) m.kv_u32("edmformer.n_heads", 16);
    d_model     = (int) m.kv_u32("edmformer.d_model", 1024);
    conv_kernel = (int) m.kv_u32("edmformer.conv_kernel", 31);
    n_mels      = (int) m.kv_u32("edmformer.mel.n_mels", 128);
    n_fft       = (int) m.kv_u32("edmformer.mel.n_fft", 2048);
    hop         = (int) m.kv_u32("edmformer.mel.hop", 240);
    ln_eps      = m.kv_f32("edmformer.ln_eps", 1e-5f);
    rope_base   = m.kv_f32("edmformer.rope_base", 10000.f);
    mel_mean    = m.kv_f32("edmformer.mel.mean", 0.f);
    mel_std     = m.kv_f32("edmformer.mel.std", 1.f);

    fbank  = m.to_host_f32(m.get("mel.fbank"));
    window = m.to_host_f32(m.get("mel.window"));
    return true;
}

int64_t ssl_model::mel(const float * samples, int64_t n, std::vector<float> & out) const {
    mel_params p;
    p.n_fft = n_fft; p.hop = hop; p.n_mels = n_mels;
    p.mean = mel_mean; p.std_dev = mel_std;
    p.fbank = fbank.data(); p.window = window.data();
    return mel_spectrogram(samples, n, p, out);
}

// conv2d via f32 im2col + matmul: numerically exact and fast on GPU and CPU
// (the "direct" conv kernels are an order of magnitude slower on both).
// layout matches ggml_conv_2d_direct: w [KW,KH,IC,OC], x [W,H,C,N] -> [W',H',OC,N]
static ggml_tensor * conv2d_mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, int stride) {
    ggml_tensor * im = ggml_im2col(ctx, w, x, stride, stride, 1, 1, 1, 1, true, GGML_TYPE_F32);
    ggml_tensor * r = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2] * im->ne[3]),
        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));
    r = ggml_reshape_4d(ctx, r, im->ne[1], im->ne[2], im->ne[3], w->ne[3]);
    return ggml_cont(ctx, ggml_permute(ctx, r, 0, 1, 3, 2));
}

// one Res2dModule (BN folded): relu(conv2(relu(conv1(x))) + conv3(x))
static ggml_tensor * res2d(ggml_context * ctx, const gguf_model & m, int r, ggml_tensor * x) {
    auto conv = [&](const char * name, ggml_tensor * inp, int stride) {
        ggml_tensor * w = m.get("frontend.res%d.%s.weight", r, name);
        ggml_tensor * b = m.get("frontend.res%d.%s.bias", r, name);
        ggml_tensor * y = conv2d_mm(ctx, w, inp, stride);
        // y: [W,H,OC,1]; bias per OC
        return ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, 1, b->ne[0]));
    };
    ggml_tensor * a = conv("conv1", x, 2);
    a = ggml_relu(ctx, a);
    a = conv("conv2", a, 1);
    ggml_tensor * res = conv("conv3", x, 2);
    return ggml_relu(ctx, ggml_add(ctx, a, res));
}

// frontend is computed in time chunks so the f32 im2col buffers stay bounded.
// the conv stack's receptive field is mel[4*t4-9 .. 4*t4+9] per output frame,
// so a halo of 12 mel frames (multiple of 4, keeps the /4 grids aligned) makes
// chunked outputs bit-identical to a single full pass.
static const int64_t FE_CHUNK = 512;
static const int64_t FE_HALO  = 12;

// mel [n_mels][T_mel] -> frontend embeddings [1024, T4]; returns T4
int64_t ssl_model::frontend_forward(const float * mel_data, int64_t T_mel, int n_threads,
                                    std::vector<float> & fe) const {
    const int64_t T2 = (T_mel - 1) / 2 + 1;
    const int64_t T4 = (T2 - 1) / 2 + 1;
    fe.resize((size_t) T4 * d_model);

    std::vector<float> chunk_mel;
    for (int64_t a = 0; a < T_mel; a += FE_CHUNK) {
        const int64_t b  = std::min(T_mel, a + FE_CHUNK);
        const int64_t c0 = std::max<int64_t>(0, a - FE_HALO);
        const int64_t c1 = std::min(T_mel, b + FE_HALO);
        const int64_t len = c1 - c0;

        const size_t meta_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false);
        ggml_init_params ip = { meta_size, nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * inp = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, len, n_mels, 1, 1);
        ggml_set_input(inp);

        ggml_tensor * x = res2d(ctx, m, 0, inp);   // [len/2, 64, 512]
        x = res2d(ctx, m, 1, x);                   // [len/4, 32, 512]
        const int64_t T4_loc = x->ne[0];
        // rearrange "b c f t -> b t (c f)"; flatten index = c*32 + f
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));      // ne = [32, 512, T4_loc]
        x = ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], T4_loc);  // [16384, T4_loc]
        x = gg_linear(ctx, m.get("frontend.linear.weight"), m.get("frontend.linear.bias"), x);
        ggml_set_output(x);

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 512, false);
        ggml_build_forward_expand(gf, x);

        graph_runner runner(512, n_threads);
        if (!runner.alloc(gf)) abort();

        // mel rows are time-contiguous over the full signal; copy the slice
        chunk_mel.resize((size_t) len * n_mels);
        for (int f = 0; f < n_mels; f++) {
            memcpy(chunk_mel.data() + (size_t) f * len, mel_data + (size_t) f * T_mel + c0,
                   len * sizeof(float));
        }
        ggml_backend_tensor_set(inp, chunk_mel.data(), 0, chunk_mel.size() * sizeof(float));

        if (!runner.compute(gf)) abort();

        // keep the core range [a/4, b/4) (chunk grids align: c0 % 4 == 0)
        const int64_t t4_0 = a / 4;
        const int64_t t4_1 = (b == T_mel) ? T4 : b / 4;
        const int64_t local0 = t4_0 - c0 / 4;
        ggml_backend_tensor_get(x, fe.data() + (size_t) t4_0 * d_model,
                                (size_t) local0 * d_model * sizeof(float),
                                (size_t) (t4_1 - t4_0) * d_model * sizeof(float));
        ggml_free(ctx);
    }
    return T4;
}

int64_t ssl_model::encode_mel(const float * mel_data, int64_t T_mel, int n_threads,
                              std::vector<float> & out, std::vector<float> * frontend_out) const {
    const int hd = d_model / n_heads;  // 64

    std::vector<float> fe;
    const int64_t T = frontend_forward(mel_data, T_mel, n_threads, fe);
    if (frontend_out) *frontend_out = fe;

    const size_t meta_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    ggml_init_params ip = { meta_size, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, T);
    ggml_set_name(inp, "frontend");
    ggml_set_input(inp);
    ggml_tensor * x = inp;

    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    const float attn_scale = 1.0f / sqrtf((float) hd);

    for (int il = 0; il < n_layers; il++) {
        // ---- FFN1 (half-step) ----
        ggml_tensor * h = gg_layer_norm(ctx, x, m.get("l%d.ffn1_ln.weight", il), m.get("l%d.ffn1_ln.bias", il), ln_eps);
        h = gg_linear(ctx, m.get("l%d.ffn1.w1.weight", il), m.get("l%d.ffn1.w1.bias", il), h);
        h = ggml_silu(ctx, h);
        h = gg_linear(ctx, m.get("l%d.ffn1.w2.weight", il), m.get("l%d.ffn1.w2.bias", il), h);
        x = ggml_add(ctx, x, ggml_scale(ctx, h, 0.5f));

        // ---- self-attention (rotary applied to hidden states before q/k proj) ----
        h = gg_layer_norm(ctx, x, m.get("l%d.attn_ln.weight", il), m.get("l%d.attn_ln.bias", il), ln_eps);
        ggml_tensor * hr = ggml_reshape_3d(ctx, h, hd, n_heads, T);
        hr = ggml_rope_ext(ctx, hr, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                           rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        hr = ggml_reshape_2d(ctx, hr, d_model, T);

        ggml_tensor * q = gg_linear(ctx, m.get("l%d.attn.q.weight", il), m.get("l%d.attn.q.bias", il), hr);
        ggml_tensor * k = gg_linear(ctx, m.get("l%d.attn.k.weight", il), m.get("l%d.attn.k.bias", il), hr);
        ggml_tensor * v = gg_linear(ctx, m.get("l%d.attn.v.weight", il), m.get("l%d.attn.v.bias", il), h);

        ggml_tensor * att = gg_attention(ctx,
            ggml_reshape_3d(ctx, q, hd, n_heads, T),
            ggml_reshape_3d(ctx, k, hd, n_heads, T),
            ggml_reshape_3d(ctx, v, hd, n_heads, T), attn_scale);
        att = gg_linear(ctx, m.get("l%d.attn.out.weight", il), m.get("l%d.attn.out.bias", il), att);
        x = ggml_add(ctx, x, att);

        // ---- convolution module ----
        h = gg_layer_norm(ctx, x, m.get("l%d.conv_ln.weight", il), m.get("l%d.conv_ln.bias", il), ln_eps);
        h = ggml_mul_mat(ctx, m.get("l%d.conv.pw1.weight", il), h);   // [2048, T]
        // GLU over channels: first half * sigmoid(second half)
        {
            ggml_tensor * a = ggml_view_2d(ctx, h, d_model, T, h->nb[1], 0);
            ggml_tensor * b = ggml_view_2d(ctx, h, d_model, T, h->nb[1], d_model * sizeof(float));
            h = ggml_mul(ctx, ggml_cont(ctx, a), ggml_sigmoid(ctx, ggml_cont(ctx, b)));
        }
        // depthwise conv k=31, 'same' padding (+ folded BatchNorm), then swish
        {
            ggml_tensor * ht = ggml_cont(ctx, ggml_transpose(ctx, h));            // [T, 1024]
            ht = ggml_reshape_4d(ctx, ht, T, 1, d_model, 1);                       // [W=T, H=1, C, N=1]
            ht = ggml_conv_2d_dw_direct(ctx, m.get("l%d.conv.dw.weight", il), ht,
                                        1, 1, (conv_kernel - 1) / 2, 0, 1, 1);     // [T, 1, C, 1]
            ht = ggml_reshape_2d(ctx, ht, T, d_model);
            ggml_tensor * dwb = m.get("l%d.conv.dw.bias", il);
            ht = ggml_add(ctx, ht, ggml_reshape_2d(ctx, dwb, 1, d_model));
            ht = ggml_silu(ctx, ht);
            h = ggml_cont(ctx, ggml_transpose(ctx, ht));                           // [1024, T]
        }
        h = ggml_mul_mat(ctx, m.get("l%d.conv.pw2.weight", il), h);
        x = ggml_add(ctx, x, h);

        // ---- FFN2 (half-step) ----
        h = gg_layer_norm(ctx, x, m.get("l%d.ffn2_ln.weight", il), m.get("l%d.ffn2_ln.bias", il), ln_eps);
        h = gg_linear(ctx, m.get("l%d.ffn2.w1.weight", il), m.get("l%d.ffn2.w1.bias", il), h);
        h = ggml_silu(ctx, h);
        h = gg_linear(ctx, m.get("l%d.ffn2.w2.weight", il), m.get("l%d.ffn2.w2.bias", il), h);
        x = ggml_add(ctx, x, ggml_scale(ctx, h, 0.5f));

        // ---- final block LayerNorm ----
        x = gg_layer_norm(ctx, x, m.get("l%d.final_ln.weight", il), m.get("l%d.final_ln.bias", il), ln_eps);
    }

    ggml_set_output(x);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, x);

    graph_runner runner(8192, n_threads);
    if (!runner.alloc(gf)) abort();

    ggml_backend_tensor_set(inp, fe.data(), 0, fe.size() * sizeof(float));
    std::vector<int32_t> positions(T);
    std::iota(positions.begin(), positions.end(), 0);
    ggml_backend_tensor_set(pos, positions.data(), 0, T * sizeof(int32_t));

    if (!runner.compute(gf)) abort();

    out.resize((size_t) T * d_model);
    ggml_backend_tensor_get(x, out.data(), 0, out.size() * sizeof(float));

    ggml_free(ctx);
    return T;
}

int64_t ssl_model::extract(const float * samples, int64_t n, int n_threads, std::vector<float> & out) const {
    std::vector<float> m_;
    const int64_t T_mel = mel(samples, n, m_);
    if (T_mel <= 0) return 0;
    return encode_mel(m_.data(), T_mel, n_threads, out);
}
