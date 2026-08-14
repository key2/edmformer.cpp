"""Dump golden fixtures from the Python pipeline for C++ parity testing.

Run from the EDMFormer repo root:
    pdm run python ../edmformer.cpp/convert/dump_fixtures.py ../guarix_animal.mp3 ../edmformer.cpp/fixtures

Produces (fp32 .npy):
  audio_full.npy         full song @24k mono (exact samples the python pipeline used)
  audio_30s.npy          first 30 s
  mel_musicfm_30s.npy    normalized mel, (128, T_mel)  [musicfm path]
  mel_muq_30s.npy        normalized mel, (128, T_mel)  [muq path - same params]
  frontend_musicfm_30s.npy  conv frontend output (T, 1024)
  h10_musicfm_30s.npy    hidden_states[10] (T, 1024)
  h10_muq_30s.npy        hidden_states[10] (T, 1024)
  fused_30s.npy          fused embedding for the 30s clip treated as a song (T, 4096)
  boundary_logits.npy    full-song accumulated boundary logits (T,)
  function_logits.npy    full-song accumulated function logits (T, 128) (masked: -inf)
  segments_ref.txt       final MSA output (after rule post-processing)
"""

import os
import sys

import numpy as np
import torch

REPO = os.getcwd()
sys.path.insert(0, REPO)
audio_path = os.path.abspath(sys.argv[1])
out_dir = os.path.abspath(sys.argv[2])
os.makedirs(out_dir, exist_ok=True)

import app  # noqa: E402  (chdirs into src/SongFormer)

app.download_all(use_mirror=False)
app.initialize_models(model_name="SongFormer", checkpoint="SongFormer.safetensors",
                      config_path="SongFormer.yaml")

import librosa  # noqa: E402

SR = app.INPUT_SAMPLING_RATE
wav, _ = librosa.load(audio_path, sr=SR)
np.save(os.path.join(out_dir, "audio_full.npy"), wav.astype(np.float32))
clip = wav[: 30 * SR].astype(np.float32)
np.save(os.path.join(out_dir, "audio_30s.npy"), clip)

x = torch.tensor(clip).unsqueeze(0)

# ---- MusicFM internals on the 30 s clip ----
mfm = app.musicfm_model
with torch.no_grad():
    mel = mfm.preprocessing(x, features=["melspec_2048"])
    mel = mfm.normalize(mel)["melspec_2048"]  # (1, 128, T)
    np.save(os.path.join(out_dir, "mel_musicfm_30s.npy"), mel[0].numpy().astype(np.float32).copy(order="C"))
    fe = mfm.conv(mel)  # (1, T', 1024)
    np.save(os.path.join(out_dir, "frontend_musicfm_30s.npy"), fe[0].numpy().astype(np.float32).copy(order="C"))
    _, hs = mfm.get_predictions(x)
    np.save(os.path.join(out_dir, "h10_musicfm_30s.npy"), hs[10][0].numpy().astype(np.float32).copy(order="C"))

# ---- MuQ internals on the 30 s clip ----
muq = app.muq_model
with torch.no_grad():
    mel2 = muq.model.preprocessing(x, features=["melspec_2048"])
    mel2 = muq.model.normalize(mel2)["melspec_2048"]
    np.save(os.path.join(out_dir, "mel_muq_30s.npy"), mel2[0].numpy().astype(np.float32).copy(order="C"))
    out = muq(x, output_hidden_states=True)
    np.save(os.path.join(out_dir, "h10_muq_30s.npy"), out["hidden_states"][10][0].numpy().astype(np.float32).copy(order="C"))

# ---- fused embedding for the 30 s clip (as process_audio builds it) ----
with torch.no_grad():
    muq_420 = muq(x, output_hidden_states=True)["hidden_states"][10]
    mfm_420 = mfm.get_predictions(x)[1][10]
    muq_30 = muq(x, output_hidden_states=True)["hidden_states"][10]
    mfm_30 = mfm.get_predictions(x)[1][10]
    embds = [mfm_30, muq_30, mfm_420, muq_420]
    L = min(e.shape[1] for e in embds)
    fused = torch.cat([e[:, :L, :] for e in embds], dim=-1)
    np.save(os.path.join(out_dir, "fused_30s.npy"), fused[0].numpy().astype(np.float32).copy(order="C"))

# ---- full-song logits + final segments ----
logits, msa = app.process_audio(audio_path)
msa = app.rule_post_processing(msa)
np.save(os.path.join(out_dir, "boundary_logits.npy"),
        logits["boundary_logits"][0].numpy().astype(np.float32).copy(order="C"))
np.save(os.path.join(out_dir, "function_logits.npy"),
        logits["function_logits"][0].numpy().astype(np.float32).copy(order="C"))
with open(os.path.join(out_dir, "segments_ref.txt"), "w") as f:
    f.write(app.format_as_msa(msa))

print("fixtures written to", out_dir)
