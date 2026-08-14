"""Convert MusicFM-25Hz / MuQ-large checkpoints to GGUF for edmformer.cpp.

Usage (run with the EDMFormer PDM venv, which has torch+gguf installed):
    pdm run python ../edmformer.cpp/convert/convert_ssl.py --model musicfm --out ../edmformer.cpp/models/musicfm-f16.gguf
    pdm run python ../edmformer.cpp/convert/convert_ssl.py --model muq     --out ../edmformer.cpp/models/muq-f16.gguf

Only what inference needs is exported:
  - mel filterbank + STFT window (exact buffers from the checkpoint)
  - Conv2dSubsampling frontend (BatchNorms folded into convs)
  - conformer layers 0..extract_layer-1 (BatchNorm in conv module folded into
    the depthwise conv). hidden_states[10] == output of layers 0..9, so the
    remaining layers and the final encoder LayerNorm are dropped.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import gguf

EXTRACT_LAYER = 10  # hidden_states[10] = output after conformer layers 0..9


def fold_bn_conv2d(w, b, bn_w, bn_b, bn_m, bn_v, eps=1e-5):
    """Fold BatchNorm2d into a preceding Conv2d. w: (OC,IC,KH,KW), b: (OC,)."""
    scale = bn_w / np.sqrt(bn_v + eps)
    w2 = w * scale.reshape(-1, 1, 1, 1)
    b2 = (b - bn_m) * scale + bn_b
    return w2.astype(np.float32), b2.astype(np.float32)


def fold_bn_dwconv1d(w, bn_w, bn_b, bn_m, bn_v, eps=1e-5):
    """Fold BatchNorm1d into a bias-less depthwise Conv1d. w: (C,1,K)."""
    scale = bn_w / np.sqrt(bn_v + eps)
    w2 = w * scale.reshape(-1, 1, 1)
    b2 = bn_b - bn_m * scale
    return w2.astype(np.float32), b2.astype(np.float32)


def load_state_dict(model_name: str):
    if model_name == "musicfm":
        path = os.path.join("src", "SongFormer", "ckpts", "MusicFM", "pretrained_msd.pt")
        S = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k[6:]: v for k, v in S.items()}  # strip "model."
        stats = json.load(open(os.path.join("src", "SongFormer", "ckpts", "MusicFM", "msd_stats.json")))
        mean = float(stats["melspec_2048_mean"])
        std = float(stats["melspec_2048_std"])
    elif model_name == "muq":
        import glob
        from safetensors.torch import load_file

        snaps = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--OpenMuQ--MuQ-large-msd-iter/snapshots/*/model.safetensors"))
        if not snaps:
            print("MuQ safetensors not found in HF cache; run the python pipeline once first")
            sys.exit(1)
        sd = load_file(snaps[0])
        sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
        cfg = json.load(open(os.path.join(os.path.dirname(snaps[0]), "config.json")))
        mean = float(cfg["stat"]["melspec_2048_mean"])
        std = float(cfg["stat"]["melspec_2048_std"])
    else:
        raise ValueError(model_name)
    return sd, mean, std


def np32(t):
    return t.detach().cpu().float().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["musicfm", "muq"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    args = ap.parse_args()

    sd, mean, std = load_state_dict(args.model)
    big = np.float16 if args.dtype == "f16" else np.float32

    w = gguf.GGUFWriter(args.out, arch=f"edmformer-ssl-{args.model}")
    w.add_string("edmformer.model_type", args.model)
    w.add_float32("edmformer.mel.mean", mean)
    w.add_float32("edmformer.mel.std", std)
    w.add_uint32("edmformer.mel.n_mels", 128)
    w.add_uint32("edmformer.mel.n_fft", 2048)
    w.add_uint32("edmformer.mel.hop", 240)
    w.add_uint32("edmformer.sr", 24000)
    w.add_uint32("edmformer.n_layers", EXTRACT_LAYER)
    w.add_uint32("edmformer.n_heads", 16)
    w.add_uint32("edmformer.d_model", 1024)
    w.add_uint32("edmformer.d_ffn", 4096)
    w.add_uint32("edmformer.conv_kernel", 31)
    w.add_uint32("edmformer.conv_dim", 512)
    w.add_float32("edmformer.ln_eps", 1e-5)
    w.add_float32("edmformer.rope_base", 10000.0)

    def add(name, arr, dtype=None):
        arr = np.ascontiguousarray(arr, dtype=dtype if dtype is not None else arr.dtype)
        w.add_tensor(name, arr)

    # ---- mel buffers (exact ones from the checkpoint, else torchaudio defaults) ----
    if "preprocessor_melspec_2048.mel_stft.mel_scale.fb" in sd:
        fb = np32(sd["preprocessor_melspec_2048.mel_stft.mel_scale.fb"])         # (1025, 128)
        win = np32(sd["preprocessor_melspec_2048.mel_stft.spectrogram.window"])  # (2048,)
    else:
        import torchaudio
        m = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=2048,
                                                 hop_length=240, n_mels=128)
        fb = np32(m.mel_scale.fb)
        win = np32(m.spectrogram.window)
    add("mel.fbank", fb.T.copy(), np.float32)  # store (128, 1025): row m = filter for mel bin m
    add("mel.window", win, np.float32)

    # ---- conv frontend: 2x Res2dModule with folded BN ----
    for r in range(2):
        p = f"conv.conv.{r}."
        for ci, bn in (("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")):
            key = p + ci + ".weight"
            if key not in sd:
                continue  # conv3 absent when not diff
            cw, cb = fold_bn_conv2d(
                np32(sd[key]), np32(sd[p + ci + ".bias"]),
                np32(sd[p + bn + ".weight"]), np32(sd[p + bn + ".bias"]),
                np32(sd[p + bn + ".running_mean"]), np32(sd[p + bn + ".running_var"]))
            # frontend stays f32: it is small and errors here propagate through
            # all conformer layers
            add(f"frontend.res{r}.{ci}.weight", cw, np.float32)
            add(f"frontend.res{r}.{ci}.bias", cb, np.float32)
    add("frontend.linear.weight", np32(sd["conv.linear.weight"]), np.float32)
    add("frontend.linear.bias", np32(sd["conv.linear.bias"]), np.float32)

    # ---- conformer layers 0..EXTRACT_LAYER-1 ----
    for i in range(EXTRACT_LAYER):
        p = f"conformer.layers.{i}."
        o = f"l{i}."
        def cp(dst, src, dtype):
            add(o + dst, np32(sd[p + src]), dtype)

        for ffn in ("ffn1", "ffn2"):
            cp(f"{ffn}_ln.weight", f"{ffn}_layer_norm.weight", np.float32)
            cp(f"{ffn}_ln.bias", f"{ffn}_layer_norm.bias", np.float32)
            cp(f"{ffn}.w1.weight", f"{ffn}.intermediate_dense.weight", big)
            cp(f"{ffn}.w1.bias", f"{ffn}.intermediate_dense.bias", np.float32)
            cp(f"{ffn}.w2.weight", f"{ffn}.output_dense.weight", big)
            cp(f"{ffn}.w2.bias", f"{ffn}.output_dense.bias", np.float32)

        cp("attn_ln.weight", "self_attn_layer_norm.weight", np.float32)
        cp("attn_ln.bias", "self_attn_layer_norm.bias", np.float32)
        for d, s in (("q", "linear_q"), ("k", "linear_k"), ("v", "linear_v"), ("out", "linear_out")):
            cp(f"attn.{d}.weight", f"self_attn.{s}.weight", big)
            cp(f"attn.{d}.bias", f"self_attn.{s}.bias", np.float32)

        cp("conv_ln.weight", "conv_module.layer_norm.weight", np.float32)
        cp("conv_ln.bias", "conv_module.layer_norm.bias", np.float32)
        # pointwise convs (k=1) as plain matrices
        pw1 = np32(sd[p + "conv_module.pointwise_conv1.weight"])[:, :, 0]  # (2048,1024)
        pw2 = np32(sd[p + "conv_module.pointwise_conv2.weight"])[:, :, 0]  # (1024,1024)
        add(o + "conv.pw1.weight", pw1, big)
        add(o + "conv.pw2.weight", pw2, big)
        # depthwise conv with folded BN -> kernel for ggml_conv_2d_dw_direct: np (C,1,1,K)
        dw, dwb = fold_bn_dwconv1d(
            np32(sd[p + "conv_module.depthwise_conv.weight"]),
            np32(sd[p + "conv_module.batch_norm.weight"]),
            np32(sd[p + "conv_module.batch_norm.bias"]),
            np32(sd[p + "conv_module.batch_norm.running_mean"]),
            np32(sd[p + "conv_module.batch_norm.running_var"]))
        add(o + "conv.dw.weight", dw.reshape(1024, 1, 1, 31), np.float32)
        add(o + "conv.dw.bias", dwb, np.float32)

        cp("final_ln.weight", "final_layer_norm.weight", np.float32)
        cp("final_ln.bias", "final_layer_norm.bias", np.float32)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
