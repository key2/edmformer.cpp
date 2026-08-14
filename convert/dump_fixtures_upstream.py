"""Dump golden fixtures for the upstream SongFormer variant (stride 3, 8.333 fps).

app.py is identical in both repos; only configs/SongFormer.yaml differs
(down_sample_conv_stride/frame_rates). This script temporarily swaps the
EDMFormer config for the upstream one, runs the pipeline, and restores it.

Run from the EDMFormer repo root:
    pdm run python ../edmformer.cpp/convert/dump_fixtures_upstream.py \
        ../guarix_animal.mp3 ../edmformer.cpp/fixtures-upstream \
        ../SongFormer/src/SongFormer/configs/SongFormer.yaml
"""

import os
import shutil
import sys

import numpy as np

REPO = os.getcwd()
sys.path.insert(0, REPO)
audio_path = os.path.abspath(sys.argv[1])
out_dir = os.path.abspath(sys.argv[2])
upstream_yaml = os.path.abspath(sys.argv[3])
os.makedirs(out_dir, exist_ok=True)

cfg = os.path.join(REPO, "src", "SongFormer", "configs", "SongFormer.yaml")
bak = cfg + ".edmformer.bak"
shutil.copyfile(cfg, bak)
shutil.copyfile(upstream_yaml, cfg)

try:
    import app  # noqa: E402  (chdirs into src/SongFormer, reads the swapped yaml)

    app.download_all(use_mirror=False)
    app.initialize_models(model_name="SongFormer", checkpoint="SongFormer.safetensors",
                          config_path="SongFormer.yaml")

    logits, msa = app.process_audio(audio_path)
    msa = app.rule_post_processing(msa)
    np.save(os.path.join(out_dir, "boundary_logits.npy"),
            np.ascontiguousarray(logits["boundary_logits"][0].numpy().astype(np.float32)))
    np.save(os.path.join(out_dir, "function_logits.npy"),
            np.ascontiguousarray(logits["function_logits"][0].numpy().astype(np.float32)))
    with open(os.path.join(out_dir, "segments_ref.txt"), "w") as f:
        f.write(app.format_as_msa(msa))
    print("upstream fixtures written to", out_dir)
finally:
    shutil.copyfile(bak, cfg)
    os.remove(bak)
    print("restored EDMFormer config")
