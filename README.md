# edmformer.cpp

Music structure analysis in C++ — a dependency-free port of the
[SongFormer](https://github.com/ASLP-lab/SongFormer) / EDMFormer pipeline on
[ggml](https://github.com/ggml-org/ggml). It takes a song and returns its
structural segments:

```
$ ./build/edmformer-cli song.mp3
===== Detected structure (MSA) =====
0.00 intro
56.24 chorus
71.28 verse
86.24 inst
120.00 chorus
...
187.12 end
```

No Python at inference time: one binary (or one static library) plus the
GGUF weights (two SSL models + one head per variant). Runs on GPU — Metal on
macOS, Vulkan or CUDA on Linux and Windows (cross-compiled, WINE-verified) —
with BLAS/CPU fallback. A ~3 minute song is analyzed in **~18 s** on Apple
Silicon GPU (~75 s CPU+BLAS) and **~17 s** end-to-end on an RTX 5090 with
Vulkan, model load included.

## How it works

```
audio (24 kHz mono)
  ├─> MuQ-large      ──┐  SSL conformers, layer-10 hidden states @ 25 Hz
  ├─> MusicFM-25Hz   ──┤  (each computed on 420 s windows AND wrapped 30 s chunks)
  │                    ▼
  │        fused embedding [T, 4096]
  │                    ▼
  │        SongFormer head (transformer, ~26 M params)
  │                    ▼
  │        boundary logits + 128-class function logits
  │                    ▼
  └──────> peak picking + segment labeling + rule cleanup
                       ▼
        [(0.00, "intro"), (56.24, "chorus"), ..., (187.12, "end")]
```

Two heads are supported, each with its own weights file, TimeDownsample
stride (and thus logit frame rate) and label set:

| Variant | Weights file | Checkpoint | Stride | Frame rate | Labels |
|---|---|---|---|---|---|
| `songformer` | `songformer-f32.gguf` | upstream ASLP-lab SongFormer | 3 | 8.333 fps | intro/verse/chorus/bridge/inst/outro/silence/pre-chorus |
| `edmformer` | `edmformer-f32.gguf` | EDM-98 fine-tune (EMA, step 2400) | 2 | 12.5 fps | intro/buildup/drop/breakdown/outro/silence |

`-V` picks the weights file; stride, frame rate, dataset-embedding id and
label set then come from that file's GGUF metadata. Without `-V`,
`edmformer-f32.gguf` is preferred when present. If the requested file is
missing, the CLI falls back to the other one with a runtime stride override
(the pre-fine-tune behavior) and a warning. See
[Models: provenance and GGUF conversion](#models-provenance-and-gguf-conversion)
for where to download the checkpoints and how to convert them.

## Building

Requires CMake ≥ 3.14, a C++17 compiler, and a ggml checkout (sibling
directory `../ggml` by default, configurable with `-DEDMFORMER_GGML_DIR=`):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # add -DGGML_METAL=ON on macOS
cmake --build build -j8
```

On Linux, pick a GPU backend for ggml at configure time:

```bash
# Vulkan (any GPU with a Vulkan 1.3 driver; needs libvulkan-dev, glslc, spirv-headers)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
# or CUDA (needs the CUDA toolkit)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

The code is backend-agnostic (it asks ggml for any GPU device), so no source
changes are needed to switch backends. mp3/wav/npy decode is built in
(dr_libs); other containers (m4a/aac/mp4/...) use CoreAudio on macOS and fall
back to piping through the `ffmpeg` CLI on Linux/Windows (runtime dependency
only).

### Cross-compiling for Windows (from Linux)

Uses MinGW-w64 (`gcc-mingw-w64-x86-64`, POSIX threads flavor) with the Vulkan
backend; the result is a self-contained `edmformer-cli.exe` that only needs
the system `vulkan-1.dll` (any GPU driver provides it). One-time deps:

```bash
# Vulkan headers (any recent tag) — used as-is, no install step
git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers.git win-deps/Vulkan-Headers
# vulkan-1 import library: extract exports from any vulkan-1.dll (wine's works)
gendef /opt/wine-stable/lib/wine/x86_64-windows/vulkan-1.dll   # -> vulkan-1.def
sed -i 's/ = winevulkan\..*$//' vulkan-1.def                    # plain export names
x86_64-w64-mingw32-dlltool -d vulkan-1.def -l win-deps/libvulkan-1.a -D vulkan-1.dll
# SPIRV-Headers: header-only host package (apt install spirv-headers); ggml's
# native build finds it in /usr/include implicitly, the cross build must not
# see /usr/include, so expose just the spirv/ subtree:
mkdir -p win-deps/spirv-include && ln -s /usr/include/spirv win-deps/spirv-include/spirv
```

Then configure with the bundled toolchain file (`cmake/toolchain-mingw64.cmake`):

```bash
cmake -B build-win -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-mingw64.cmake \
    -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON \
    -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON \
    -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
    -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CXX_FLAGS="-isystem $PWD/../win-deps/spirv-include" \
    -DVulkan_INCLUDE_DIR=$PWD/../win-deps/Vulkan-Headers/include \
    -DVulkan_LIBRARY=$PWD/../win-deps/libvulkan-1.a \
    -DVulkan_GLSLC_EXECUTABLE=/usr/bin/glslc \
    -DSPIRV-Headers_DIR=/usr/share/cmake/SPIRV-Headers
cmake --build build-win -j$(nproc)
```

`ggml` builds its Vulkan shader generator with the host compiler
automatically. The exe runs under WINE with full GPU acceleration
(`wine build-win/edmformer-cli.exe -m models song.mp3`) — verified
bit-identical to the native Linux/Vulkan output. Note: wine's winevulkan
(≤ 11.1) matches `vkGetDeviceQueue2` queues by `memcmp` over the whole
`VkDeviceQueueInfo2` including uninitialized padding, which makes stock ggml
submit on a NULL queue and crash. Apply the included fix to the ggml checkout:

```bash
git -C ../ggml apply ../edmformer.cpp/patches/ggml-vulkan-winevulkan-queue-lookup.patch
```

(plain `getQueue` when no queue flags are used, zero-initialized struct
otherwise — behavior on native drivers is unchanged).

Build products:

- `libedmformer_core.a` — the library (link target `edmformer_core`)
- `edmformer-cli` — command-line tool
- `edmformer-test` — parity test harness

## CLI usage

```bash
./build/edmformer-cli [options] song.{mp3,wav,m4a,npy}

  -m DIR    model directory (default: models)
  -V NAME   head model: edmformer (EDM fine-tune) | songformer (upstream)
            (default: edmformer-f32.gguf when present)
  -t N      threads for the CPU backend
  -o PREFIX write PREFIX.txt (MSA format) and PREFIX.json
  -c        CPU only (disable GPU backend)
  -q        quiet
```

`.npy` inputs (fp32 mono @ 24 kHz) bypass decoding/resampling — useful for
reproducible comparisons. `EDMFORMER_PROFILE=1` prints a per-op time
breakdown; `EDMFORMER_CPU=1` is equivalent to `-c`.

## Using it as a library

Link the `edmformer_core` target; it exports the `src/` and `third_party/`
include directories and links ggml transitively:

```cmake
add_subdirectory(edmformer.cpp)          # or FetchContent
target_link_libraries(my_app PRIVATE edmformer_core)
```

### High-level API (whole song in, segments out)

```cpp
#include "audio.h"       // load_audio
#include "compute.h"     // compute_init / backends
#include "pipeline.h"    // process_song, logits_to_msa
#include "postprocess.h" // msa_info, rule_post_processing, formatting

int main() {
    // 1. backends: GPU (Metal) -> BLAS -> CPU. Call once, before loading models.
    //    compute_init(n_threads, /*use_gpu=*/true)
    compute_init(8, true);

    // 2. load the three models (weights go to GPU memory when available)
    ssl_model musicfm, muq;
    songformer_model head;
    if (!musicfm.load("models/musicfm-f16.gguf")) return 1;
    if (!muq.load("models/muq-f16.gguf"))         return 1;
    if (!head.load("models/edmformer-f32.gguf"))  return 1;  // EDM fine-tune
    // (or "models/songformer-f32.gguf" for the upstream 8-class head;
    //  set_variant() only overrides the stride, not labels/weights)

    // 3. audio: any decoder works, the pipeline just needs 24 kHz mono f32.
    //    load_audio handles mp3/wav (dr_libs), m4a/aac (CoreAudio on macOS,
    //    ffmpeg CLI on Linux), downmixing and sinc resampling.
    std::vector<float> samples = load_audio("song.mp3", 24000);

    // 4. run the full pipeline (420 s windows, 30 s wrapped chunks, fusion,
    //    head inference, logit accumulation — faithful to the reference app.py)
    song_logits lg = process_song(samples, musicfm, muq, head,
                                  /*n_threads=*/8, /*verbose=*/false);

    // 5. logits -> segments
    msa_info msa = logits_to_msa(lg, head);   // peak picking + labeling
    msa = rule_post_processing(msa);          // merge spurious tiny segments

    // msa is a std::vector<std::pair<double, std::string>>:
    // {(0.0,"intro"), (56.24,"chorus"), ..., (187.12,"end")}
    // the last entry is the song-end timestamp with the sentinel label "end"
    for (auto & [t, label] : msa)
        printf("%8.2f  %s\n", t, label.c_str());

    // or serialized: format_msa(msa) / format_json(msa)
    compute_free();
}
```

Notes for integrators:

- **Thread safety**: models are immutable after `load()`; each
  `infer()`/`extract()` call builds and frees its own ggml graph. The shared
  compute environment is global — run inferences sequentially (the GPU is
  saturated by a single song anyway).
- **Memory**: ~1.3 GB of weights (GPU memory when Metal is active) plus a few
  hundred MB of transient compute buffers.
- **Raw logits**: `song_logits` gives you the averaged boundary logits `[T]`
  and function logits `[T][128]` (frame rate = `head.frame_rates`) if you want
  custom post-processing instead of `logits_to_msa`.

### Lower-level building blocks

Every stage is callable on its own (this is exactly what the test harness does):

```cpp
// SSL embeddings for an arbitrary audio span (25 Hz, layer-10 hidden states)
std::vector<float> emb;                       // [T][1024] row-major
int64_t T = musicfm.extract(samples.data(), n_samples, n_threads, emb);

// or split it: mel spectrogram -> frontend -> conformer
std::vector<float> mel;                       // [128][T_mel]
int64_t T_mel = musicfm.mel(samples.data(), n_samples, mel);
int64_t T4    = musicfm.encode_mel(mel.data(), T_mel, n_threads, emb);

// head inference on a fused embedding you assembled yourself
// fused: [T][4096] = [musicfm_30s | muq_30s | musicfm_420s | muq_420s]
std::vector<float> boundary, function_logits;
int64_t T_out = head.infer(fused.data(), T, n_threads, boundary, function_logits);
```

The label set for the loaded head is exposed as `head.allowed_ids` /
`head.allowed_labels` (from GGUF metadata), so downstream code never needs a
hardcoded label table.

## Models: provenance and GGUF conversion

### Where the weights come from

| GGUF file | Source checkpoint | Author / license | Content |
|---|---|---|---|
| `musicfm-f16.gguf` (590 MB) | [`minzwon/MusicFM`](https://huggingface.co/minzwon/MusicFM) — `pretrained_msd.pt` + `msd_stats.json` | ByteDance / Minz Won (MIT) | MusicFM-25Hz self-supervised music model, trained on the Million Song Dataset. 12-layer Wav2Vec2-Conformer (hidden 1024, rotary attention) with a mel + Conv2dSubsampling frontend |
| `muq-f16.gguf` (590 MB) | [`OpenMuQ/MuQ-large-msd-iter`](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) — `model.safetensors` | Tencent AI Lab / OpenMuQ (weights: CC-BY-NC-4.0) | MuQ-large SSL model — deliberately reuses the MusicFM skeleton (same 12-layer conformer architecture), trained with Mel-RVQ targets |
| `songformer-f32.gguf` (104 MB) | [`ASLP-lab/SongFormer`](https://huggingface.co/ASLP-lab/SongFormer) — `SongFormer.safetensors` | ASLP@NPU (CC-BY-4.0), paper [arXiv:2510.02797](https://arxiv.org/abs/2510.02797) | The MSA head: input fusion, TimeDownsample, 4-layer x-transformers encoder, boundary + function heads (EMA weights) |
| `edmformer-f32.gguf` (104 MB) | [`25ohms/EDM-98`](https://github.com/25ohms/EDM-98) — `data/checkpoints/model.pt` (git LFS) | 25ohms | Same architecture, fine-tuned on the EDM-98 dataset via [`25ohms/EDMFormer`](https://github.com/25ohms/EDMFormer) (raw trainer checkpoint, EMA state at step 2400, dataset id 9, labels intro/buildup/drop/breakdown/outro/silence) |

### Downloading the source checkpoints

The converters expect the checkpoints in the layout of the
[`25ohms/EDMFormer`](https://github.com/25ohms/EDMFormer) repo
(`src/SongFormer/ckpts/...`), so clone it next to this repo and fetch the
weights into it. All commands run from the directory that contains both
checkouts:

```bash
# one-time Python env — used for the MuQ download below and the converters
python3 -m venv .venv && . .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy safetensors gguf huggingface_hub omegaconf

git clone https://github.com/25ohms/EDMFormer.git
cd EDMFormer/src/SongFormer

# MusicFM (SSL model, MIT) + SongFormer head (upstream weights, CC-BY-4.0)
curl -L --create-dirs --parallel \
  -o ckpts/MusicFM/msd_stats.json      "https://huggingface.co/minzwon/MusicFM/resolve/main/msd_stats.json" \
  -o ckpts/MusicFM/pretrained_msd.pt   "https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt" \
  -o ckpts/SongFormer.safetensors      "https://huggingface.co/ASLP-lab/SongFormer/resolve/main/SongFormer.safetensors"

# verify against the checksums shipped in the repo
sed 's/    /  /' ckpts/md5sum.txt | grep -v "SongFormer\.pt" | (cd ckpts && md5sum -c -)

# EDM-98 fine-tuned head (git LFS object, ~418 MB)
curl -L -o ckpts/edm98-model.pt \
  "https://media.githubusercontent.com/media/25ohms/EDM-98/main/data/checkpoints/model.pt"
echo "1412e207645e9a71adc09777714dd251ce7805cada9bf19518d2e455a977e165  ckpts/edm98-model.pt" | sha256sum -c -

# MuQ (CC-BY-NC-4.0): the converter reads it from the HuggingFace cache
python -c "from huggingface_hub import hf_hub_download as d; \
  d('OpenMuQ/MuQ-large-msd-iter','model.safetensors'); \
  d('OpenMuQ/MuQ-large-msd-iter','config.json')"
```

This is exactly what the Python reference does itself
(`src/SongFormer/utils/fetch_pretrained.py` and `MuQ.from_pretrained()`).
The SongFormer head weights referenced by the EDMFormer repo are identical to
upstream (same md5), so no separate download exists for it; the EDM fine-tune
lives in the [`25ohms/EDM-98`](https://github.com/25ohms/EDM-98) repo under
`data/checkpoints/model.pt` (tracked with git LFS — a plain
`raw.githubusercontent.com` fetch returns the 134-byte pointer file, hence
the `media.githubusercontent.com` URL above).

### How the GGUF files are generated

The converters live in `convert/` and run in the venv created above (torch
CPU is enough; `omegaconf` is needed because the EDM-98 trainer checkpoint
pickles its config — the EDMFormer PDM venv works too). Run them from the
EDMFormer repo root:

```bash
cd EDMFormer
python ../edmformer.cpp/convert/convert_ssl.py --model musicfm --out ../edmformer.cpp/models/musicfm-f16.gguf
python ../edmformer.cpp/convert/convert_ssl.py --model muq     --out ../edmformer.cpp/models/muq-f16.gguf
# upstream head -> songformer-f32.gguf (8-class general music labels)
python ../edmformer.cpp/convert/convert_songformer.py --variant songformer --out ../edmformer.cpp/models/songformer-f32.gguf
# EDM fine-tuned head -> edmformer-f32.gguf (EDM-98 trainer checkpoint;
# .pt with model_ema/ema_model.* EMA state is unwrapped automatically)
python ../edmformer.cpp/convert/convert_songformer.py \
    --ckpt src/SongFormer/ckpts/edm98-model.pt \
    --variant edmformer --dataset EDMFormer \
    --out ../edmformer.cpp/models/edmformer-f32.gguf
```

Afterwards `models/` contains all four GGUFs the CLI needs
(`musicfm-f16.gguf`, `muq-f16.gguf`, `songformer-f32.gguf`,
`edmformer-f32.gguf` — ~1.4 GB total).

The conversion is not a plain dump — several inference-time transforms are
baked in:

**`convert_ssl.py`** (MusicFM and MuQ share one converter since the
architectures match):

- keeps only what layer-10 feature extraction needs: `hidden_states[10]` is
  the output of conformer layers 0–9, so **layers 10–11, the final encoder
  LayerNorm, the random-projection quantizer, the codebook head and the CLS
  token are dropped** (~17 % less compute, smaller files);
- **folds all BatchNorms** into the preceding convolutions (the two Res2d
  blocks of the Conv2dSubsampling frontend, and the BatchNorm1d inside every
  conformer convolution module, which gives the bias-less depthwise convs an
  explicit bias);
- reshapes 1×1 "pointwise conv" weights into plain matrices, and depthwise
  conv kernels into the layout `ggml_conv_2d_dw_direct` expects;
- embeds the **exact mel filterbank and STFT window** as tensors
  (`mel.fbank`, `mel.window`) — taken from the checkpoint buffers for MusicFM
  (its stored window is a *symmetric* Hann from an older torchaudio, not the
  current periodic default!) and from `torchaudio.transforms.MelSpectrogram`
  for MuQ, which doesn't persist them — plus the per-model normalization
  mean/std (`msd_stats.json` / HF config) as metadata;
- stores 2-D weights as **f16** and everything sensitive (frontend convs,
  norms, biases, dw kernels) as f32. `--dtype f32` gives bit-tighter parity
  at 2× size.

**`convert_songformer.py`**:

- unwraps the **EMA weights** (`ema_model.*`) exactly like the reference
  `initialize_models()` does;
- stores everything in f32 (the head is small) with original tensor names;
- writes the variant preset (`--variant songformer|edmformer` → TimeDownsample
  stride 3/2, frame rate 8.333/12.5) and the label configuration
  (`--dataset SongForm-HX-8Class|EDMFormer` → dataset-embedding id, allowed
  label ids and names) as GGUF metadata, so the C++ side is fully
  self-describing. The x-transformers gamma-only norms are stored raw; the
  unit offset (`scale = gamma + 1`) is applied at load time.

Architectural conventions that the C++ graphs replicate (verified against the
installed Python sources): HF Wav2Vec2-Conformer applies **full-head NeoX
rotary embeddings to the hidden states before the Q/K projections**
(`ggml_rope_ext`, mode NEOX, n_dims 64; V uses un-rotated states), while the
x-transformers head uses **partial GPT-J interleaved rotary** (n_dims 32 of
64) on Q/K after projection; torch's exact-erf GELU maps to `ggml_gelu_erf`
and "swish" to `ggml_silu`; attention runs through `ggml_flash_attn_ext`
(f16 K/V, f32 accumulation) so 420 s windows (~10 k tokens) never materialize
T×T score matrices; and the conv frontend uses f32 im2col + matmul in
time-chunks with a 12-frame halo — bit-identical to a full pass, an order of
magnitude faster than the "direct" conv ops on both Metal and CPU.

## Validation

Every stage is compared against golden tensors dumped from the Python
pipeline (`convert/dump_fixtures.py`, `convert/dump_fixtures_upstream.py`):

```bash
./build/edmformer-test fixtures models all                               # fast stages
./build/edmformer-test fixtures models e2e                               # full song, edmformer variant
./build/edmformer-test fixtures-upstream models e2e songformer           # full song, upstream variant
```

| Stage | Parity vs Python (fp32 reference) |
|---|---|
| log-mel front-end | max diff 7e-5 (limited by fp32 noise in the reference itself) |
| MusicFM / MuQ layer-10 embeddings | mean diff ~9e-3 / ~1e-3 (f16 weights) |
| full-song logits (both variants) | max diff ≤ 0.13 |
| **final segments, same input samples** | **byte-identical**, both variants |
| final segments, independent audio decode | identical labels, boundaries within ±0.16 s |

## Repository layout

```
convert/                    Python: GGUF converters + parity fixture dumpers
src/
  compute.{h,cpp}           backend scheduler (Metal -> BLAS -> CPU), profiler
  common.{h,cpp}            GGUF loading, npy I/O
  mel.{h,cpp}               torchaudio-compatible log-mel (double-precision FFT)
  ssl_conformer.{h,cpp}     shared MusicFM/MuQ graph (frontend + conformer)
  songformer.{h,cpp}        MSA head graph (both stride variants)
  postprocess.{h,cpp}       peak picking, labeling, rule cleanup (plain C++)
  pipeline.{h,cpp}          windowing / fusion / logit accumulation
  audio.{h,cpp}             dr_mp3, dr_wav, CoreAudio/ffmpeg decode + sinc resampler
  main.cpp                  CLI
tests/test_parity.cpp       stage-by-stage comparison against fixtures
third_party/                dr_mp3.h, dr_wav.h (public domain)
models/                     generated GGUF files (not committed)
fixtures*/                  generated parity fixtures (not committed)
```

## Known deviations / notes

- SSL weights are f16; final segments have matched the Python output exactly
  in all tests. Use `--dtype f32` in `convert_ssl.py` for tighter parity.
- Audio decoding is not bit-exact vs librosa/audioread (different decoder and
  resampler): boundaries can shift by ±1–2 frames (0.08–0.16 s).
- `app.py` accumulates window logits at 8.333 fps for both variants; this is
  exact for the upstream `songformer` head (stride 3) and a replicated quirk
  for the `edmformer` head (stride 2), where it only affects songs > 420 s.
- On Metal, outputs differ slightly from the CPU path (matmul reduction
  order); parity vs Python stays within the same tolerances.
- Respect the upstream licenses when distributing converted weights — note
  MuQ weights are **CC-BY-NC-4.0 (non-commercial)**; MusicFM is MIT and the
  SongFormer head is CC-BY-4.0 (cite [arXiv:2510.02797](https://arxiv.org/abs/2510.02797)).
