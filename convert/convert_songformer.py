"""Convert SongFormer.safetensors (EMA weights) to GGUF for edmformer.cpp.

Supports both inference variants (identical weights, different TimeDownsample
stride / output frame rate):

  --variant songformer   upstream SongFormer:  stride 3 -> 8.333 fps
  --variant edmformer    EDMFormer:            stride 2 -> 12.5 fps

and the label set / dataset id used for masking:

  --dataset SongForm-HX-8Class   (id 5, default: 7 classes + pre-chorus)
  --dataset EDMFormer            (id 9: intro/buildup/drop/breakdown/outro/silence,
                                  for EDM fine-tuned checkpoints)

Usage (from the EDMFormer repo root, using its PDM venv):
    pdm run python ../edmformer.cpp/convert/convert_songformer.py \
        --variant edmformer --out ../edmformer.cpp/models/songformer-f32.gguf

--ckpt accepts either the released safetensors (ema_model.* keys) or a raw
training checkpoint from the EDMFormer trainer (model.ckpt-<step>.pt /
model.pt: a torch.save dict whose "model_ema" entry holds the EMA state).
For EDM fine-tuned checkpoints pass --dataset EDMFormer.
"""

import argparse
import os

import numpy as np
import gguf
from safetensors.torch import load_file

VARIANTS = {
    #                stride  frame_rates
    "songformer": (3, 8.333),
    "edmformer":  (2, 12.5),
}

DATASETS = {
    "SongForm-HX-8Class": {
        "id": 5,
        "allowed_ids": [0, 1, 2, 3, 4, 5, 6, 26],
        "allowed_labels": ["intro", "verse", "chorus", "bridge", "inst", "outro",
                           "silence", "pre-chorus"],
    },
    "EDMFormer": {
        "id": 9,
        "allowed_ids": [0, 5, 6, 36, 69, 70],
        "allowed_labels": ["intro", "outro", "silence", "breakdown", "buildup", "drop"],
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("src", "SongFormer", "ckpts", "SongFormer.safetensors"))
    ap.add_argument("--variant", choices=sorted(VARIANTS.keys()), default="edmformer")
    ap.add_argument("--dataset", choices=sorted(DATASETS.keys()), default="SongForm-HX-8Class")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stride, frame_rates = VARIANTS[args.variant]
    ds = DATASETS[args.dataset]

    if args.ckpt.endswith(".pt"):
        # raw trainer checkpoint: {"model", "optimizer", ..., "model_ema"} where
        # model_ema = EMA(include_online_model=False).state_dict() -> ema_model.*
        import torch
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck["model_ema"] if isinstance(ck, dict) and "model_ema" in ck else ck
        sd = {k: v for k, v in sd.items() if hasattr(v, "detach")}
    else:
        sd = load_file(args.ckpt)
    sd = {k[len("ema_model."):]: v for k, v in sd.items() if k.startswith("ema_model.")}

    w = gguf.GGUFWriter(args.out, arch="edmformer-songformer")
    w.add_string("sf.variant", args.variant)
    w.add_uint32("sf.input_dim_raw", 4096)
    w.add_uint32("sf.input_dim", 2048)
    w.add_uint32("sf.enc_input_dim", 1024)
    w.add_uint32("sf.dim", 512)
    w.add_uint32("sf.n_layers", 4)
    w.add_uint32("sf.n_heads", 8)
    w.add_uint32("sf.rot_dim", 32)
    w.add_uint32("sf.num_classes", 128)
    w.add_uint32("sf.dataset_id", ds["id"])
    w.add_uint32("sf.ds_kernel", 3)
    w.add_uint32("sf.ds_stride", stride)
    w.add_float32("sf.frame_rates", frame_rates)
    w.add_uint32("sf.local_maxima_filter_size", 3)
    w.add_float32("sf.peak_window_sec", 12.0)
    w.add_float32("sf.rope_base", 10000.0)
    w.add_array("sf.allowed_ids", ds["allowed_ids"])
    w.add_array("sf.allowed_labels", ds["allowed_labels"])

    for k, v in sorted(sd.items()):
        arr = v.detach().cpu().float().numpy().astype(np.float32)
        # depthwise conv (2048,1,3) -> (2048,1,1,3) for ggml_conv_2d_dw_direct
        if k == "down_sample_conv.depthwise_conv.weight":
            arr = arr.reshape(2048, 1, 1, 3)
        # pointwise convs (O,I,1) -> (O,I)
        if k in ("down_sample_conv.pointwise_conv.weight", "down_sample_conv.residual_conv.weight"):
            arr = arr[:, :, 0]
        w.add_tensor(k, np.ascontiguousarray(arr))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB) "
          f"variant={args.variant} dataset={args.dataset}")


if __name__ == "__main__":
    main()
