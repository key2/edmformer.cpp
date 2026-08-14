#pragma once

#include <cstdint>
#include <vector>

// torchaudio-compatible log-mel spectrogram used by MusicFM / MuQ:
//   MelSpectrogram(sr=24000, n_fft=2048, hop=240, n_mels=128, power=2,
//                  center=True, pad_mode='reflect', window=hann_periodic)
//   -> AmplitudeToDB(power)  == 10*log10(clamp(x, 1e-10))
//   -> [..., :-1]  (drop last frame)
//   -> (x - mean) / std
//
// fbank:  n_mels rows of n_fft/2+1 coefficients (row-major [n_mels][1025])
// window: n_fft floats
// output: [n_mels][n_frames] row-major (time contiguous per mel bin)
struct mel_params {
    int   n_fft   = 2048;
    int   hop     = 240;
    int   n_mels  = 128;
    float mean    = 0.f;
    float std_dev = 1.f;
    const float * fbank  = nullptr;
    const float * window = nullptr;
};

// returns n_frames; out is resized to n_mels*n_frames
int64_t mel_spectrogram(const float * samples, int64_t n, const mel_params & p, std::vector<float> & out);
