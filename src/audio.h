#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Load .mp3 / .wav / .npy (fp32 mono) audio, downmix to mono and resample to
// target_sr (windowed-sinc). Returns empty vector on failure.
std::vector<float> load_audio(const std::string & path, int target_sr);
