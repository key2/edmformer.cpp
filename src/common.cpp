#include "common.h"
#include "compute.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// npy
// ---------------------------------------------------------------------------

bool npy_load(const std::string & path, npy_array & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "npy_load: cannot open %s\n", path.c_str());
        return false;
    }
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "npy_load: bad magic in %s\n", path.c_str());
        fclose(f);
        return false;
    }
    int major = magic[6];
    uint32_t hlen = 0;
    if (major == 1) {
        uint16_t h16;
        fread(&h16, 2, 1, f);
        hlen = h16;
    } else {
        fread(&hlen, 4, 1, f);
    }
    std::string header(hlen, '\0');
    fread(&header[0], 1, hlen, f);

    if (header.find("'<f4'") == std::string::npos && header.find("'|f4'") == std::string::npos) {
        fprintf(stderr, "npy_load: only <f4 supported (%s)\n", path.c_str());
        fclose(f);
        return false;
    }
    if (header.find("'fortran_order': False") == std::string::npos) {
        fprintf(stderr, "npy_load: only C-order supported (%s)\n", path.c_str());
        fclose(f);
        return false;
    }
    size_t p = header.find("'shape': (");
    p += strlen("'shape': (");
    size_t e = header.find(")", p);
    std::string shape_str = header.substr(p, e - p);

    out.shape.clear();
    const char * s = shape_str.c_str();
    while (*s) {
        while (*s == ' ' || *s == ',') s++;
        if (!*s) break;
        out.shape.push_back(strtoll(s, (char **) &s, 10));
    }
    if (out.shape.empty()) out.shape.push_back(1);

    out.data.resize(out.n_elements());
    size_t got = fread(out.data.data(), sizeof(float), out.data.size(), f);
    fclose(f);
    if (got != out.data.size()) {
        fprintf(stderr, "npy_load: short read in %s\n", path.c_str());
        return false;
    }
    return true;
}

bool npy_save(const std::string & path, const float * data, const std::vector<int64_t> & shape) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    std::string shp;
    for (size_t i = 0; i < shape.size(); i++) {
        shp += std::to_string(shape[i]);
        if (shape.size() == 1 || i + 1 < shape.size()) shp += ",";
        if (i + 1 < shape.size()) shp += " ";
    }
    char hdr[512];
    snprintf(hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", shp.c_str());
    size_t hl = strlen(hdr);
    size_t total = 10 + hl;
    size_t pad = (64 - (total % 64)) % 64;
    std::string header(hdr);
    header += std::string(pad, ' ');
    header.back() = '\n';
    uint16_t hlen = (uint16_t) header.size();
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hlen, 2, 1, f);
    fwrite(header.data(), 1, header.size(), f);
    int64_t n = 1;
    for (auto s : shape) n *= s;
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// gguf model
// ---------------------------------------------------------------------------

gguf_model::~gguf_model() {
    if (buf) ggml_backend_buffer_free(buf);
    if (g)   gguf_free(g);
    if (ctx) ggml_free(ctx);
}

bool gguf_model::load(const std::string & path) {
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &ctx,
    };
    g = gguf_init_from_file(path.c_str(), params);
    if (!g) {
        fprintf(stderr, "failed to load gguf: %s\n", path.c_str());
        return false;
    }

    // allocate all tensors in the weight buffer (GPU memory when available)
    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, compute_weight_buft());
    if (!buf) {
        fprintf(stderr, "failed to allocate weight buffer for %s\n", path.c_str());
        return false;
    }

    // stream tensor data from the file into the buffer
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "failed to reopen %s\n", path.c_str());
        return false;
    }
    const size_t data_off = gguf_get_data_offset(g);
    std::vector<uint8_t> tmp;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const char * name = gguf_get_tensor_name(g, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx, name);
        const size_t nbytes = ggml_nbytes(t);
        tmp.resize(nbytes);
        if (fseek(f, (long) (data_off + gguf_get_tensor_offset(g, i)), SEEK_SET) != 0 ||
            fread(tmp.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "failed to read tensor %s from %s\n", name, path.c_str());
            fclose(f);
            return false;
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
    }
    fclose(f);
    return true;
}

std::vector<float> gguf_model::to_host_f32(struct ggml_tensor * t) const {
    std::vector<float> out(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return out;
}

struct ggml_tensor * gguf_model::get(const char * fmt, ...) const {
    char name[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    struct ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "missing tensor: %s\n", name);
        abort();
    }
    return t;
}

float gguf_model::kv_f32(const char * key, float def) const {
    int64_t i = gguf_find_key(g, key);
    return i < 0 ? def : gguf_get_val_f32(g, i);
}

uint32_t gguf_model::kv_u32(const char * key, uint32_t def) const {
    int64_t i = gguf_find_key(g, key);
    return i < 0 ? def : gguf_get_val_u32(g, i);
}

std::string gguf_model::kv_str(const char * key, const std::string & def) const {
    int64_t i = gguf_find_key(g, key);
    return i < 0 ? def : gguf_get_val_str(g, i);
}

std::vector<int32_t> gguf_model::kv_arr_i32(const char * key) const {
    std::vector<int32_t> out;
    int64_t i = gguf_find_key(g, key);
    if (i < 0) return out;
    const int64_t n = gguf_get_arr_n(g, i);
    const void * data = gguf_get_arr_data(g, i);
    const enum gguf_type t = gguf_get_arr_type(g, i);
    for (int64_t j = 0; j < n; j++) {
        switch (t) {
            case GGUF_TYPE_INT32:  out.push_back(((const int32_t  *) data)[j]); break;
            case GGUF_TYPE_INT64:  out.push_back((int32_t)((const int64_t *) data)[j]); break;
            case GGUF_TYPE_UINT32: out.push_back((int32_t)((const uint32_t *) data)[j]); break;
            default: fprintf(stderr, "kv_arr_i32: unsupported type %d\n", t); abort();
        }
    }
    return out;
}

std::vector<std::string> gguf_model::kv_arr_str(const char * key) const {
    std::vector<std::string> out;
    int64_t i = gguf_find_key(g, key);
    if (i < 0) return out;
    const int64_t n = gguf_get_arr_n(g, i);
    for (int64_t j = 0; j < n; j++) {
        out.push_back(gguf_get_arr_str(g, i, j));
    }
    return out;
}

float max_abs_diff(const float * a, const float * b, int64_t n) {
    float m = 0.f;
    for (int64_t i = 0; i < n; i++) {
        // treat matching infinities as equal
        if (std::isinf(a[i]) && std::isinf(b[i]) && ((a[i] > 0) == (b[i] > 0))) continue;
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}
