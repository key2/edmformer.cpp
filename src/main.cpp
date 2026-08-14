#include "audio.h"
#include "compute.h"
#include "pipeline.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s [options] <audio.{mp3,wav,m4a,npy}>\n"
        "\n"
        "options:\n"
        "  -m DIR    model directory containing musicfm-f16.gguf, muq-f16.gguf,\n"
        "            and the head weights (default: models)\n"
        "  -V NAME   head model: edmformer (edmformer-f32.gguf, EDM fine-tune,\n"
        "            12.5 fps, EDM labels) or songformer (songformer-f32.gguf,\n"
        "            upstream weights, 8.333 fps, 8-class labels);\n"
        "            default: edmformer when edmformer-f32.gguf exists\n"
        "  -t N      threads (default: hardware concurrency)\n"
        "  -o PREFIX write PREFIX.txt (MSA) and PREFIX.json\n"
        "  -c        CPU only (disable GPU backend)\n"
        "  -q        quiet\n",
        argv0);
}

static bool file_exists(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (f) fclose(f);
    return f != nullptr;
}

// -V selects which head weights to load: each variant has its own GGUF
// (edmformer = EDM fine-tuned checkpoint, songformer = upstream checkpoint);
// stride, frame rate, label set and dataset id then come from that file's
// metadata. Falls back to the other file (with a runtime stride override and
// a warning) when the requested one is missing.
static std::string pick_head_path(const std::string & model_dir, const std::string & variant) {
    const std::string edm  = model_dir + "/edmformer-f32.gguf";
    const std::string song = model_dir + "/songformer-f32.gguf";
    if (variant.empty()) return file_exists(edm) ? edm : song;
    const std::string & want = variant == "edmformer" ? edm : song;
    const std::string & alt  = variant == "edmformer" ? song : edm;
    if (file_exists(want) || !file_exists(alt)) return want;
    fprintf(stderr, "warning: %s not found, using %s with the %s stride "
            "(weights are the other head's)\n", want.c_str(), alt.c_str(), variant.c_str());
    return alt;
}

int main(int argc, char ** argv) {
    std::string model_dir = "models";
    std::string out_prefix;
    std::string audio_path;
    std::string variant;
    int n_threads = (int) std::thread::hardware_concurrency();
    bool verbose = true;
    bool cpu_only = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "-V") && i + 1 < argc) variant = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out_prefix = argv[++i];
        else if (!strcmp(argv[i], "-c")) cpu_only = true;
        else if (!strcmp(argv[i], "-q")) verbose = false;
        else if (argv[i][0] != '-') audio_path = argv[i];
        else { usage(argv[0]); return 1; }
    }
    if (audio_path.empty()) { usage(argv[0]); return 1; }
    if (!variant.empty() && variant != "songformer" && variant != "edmformer") {
        fprintf(stderr, "unknown variant: %s (use songformer or edmformer)\n", variant.c_str());
        return 1;
    }

    compute_init(n_threads, !cpu_only);
    fprintf(stderr, "compute: %s\n",
            compute_has_gpu() ? compute_gpu_name() : "CPU only");

    fprintf(stderr, "loading audio: %s\n", audio_path.c_str());
    std::vector<float> audio = load_audio(audio_path, 24000);
    if (audio.empty()) return 1;
    fprintf(stderr, "audio: %.1f s @ 24 kHz\n", audio.size() / 24000.0);

    ssl_model musicfm, muq;
    songformer_model sf;
    fprintf(stderr, "loading models from %s ...\n", model_dir.c_str());
    if (!musicfm.load(model_dir + "/musicfm-f16.gguf")) return 1;
    if (!muq.load(model_dir + "/muq-f16.gguf")) return 1;
    const std::string head_path = pick_head_path(model_dir, variant);
    if (!sf.load(head_path)) return 1;
    if (!variant.empty() && sf.variant != variant) sf.set_variant(variant);  // fallback only
    fprintf(stderr, "head: %s — variant %s (stride %d, %.3f fps, %d labels)\n",
            head_path.c_str(), sf.variant.c_str(), sf.ds_stride, sf.frame_rates,
            (int) sf.allowed_labels.size());

    song_logits lg = process_song(audio, musicfm, muq, sf, n_threads, verbose);
    msa_info msa = logits_to_msa(lg, sf);
    msa = rule_post_processing(msa);

    const std::string txt = format_msa(msa);
    const std::string json = format_json(msa);

    printf("===== Detected structure (MSA) =====\n%s\n", txt.c_str());
    printf("\n===== Segments (JSON) =====\n%s\n", json.c_str());

    if (!out_prefix.empty()) {
        FILE * f = fopen((out_prefix + ".txt").c_str(), "w");
        if (f) { fputs(txt.c_str(), f); fclose(f); }
        f = fopen((out_prefix + ".json").c_str(), "w");
        if (f) { fputs(json.c_str(), f); fclose(f); }
        fprintf(stderr, "wrote %s.txt and %s.json\n", out_prefix.c_str(), out_prefix.c_str());
    }
    return 0;
}
