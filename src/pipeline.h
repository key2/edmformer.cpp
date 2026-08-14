#pragma once

#include "postprocess.h"
#include "songformer.h"
#include "ssl_conformer.h"

#include <cstdint>
#include <vector>

// full-song logits, faithfully replicating app.py:process_audio windowing
// (420 s windows, wrapped 30 s chunks, logit accumulation/averaging)
struct song_logits {
    std::vector<float> boundary;  // [T]
    std::vector<float> function;  // [T][num_classes]
};

song_logits process_song(const std::vector<float> & audio24k,
                         const ssl_model & musicfm, const ssl_model & muq,
                         const songformer_model & sf, int n_threads, bool verbose);

msa_info logits_to_msa(const song_logits & lg, const songformer_model & sf);
