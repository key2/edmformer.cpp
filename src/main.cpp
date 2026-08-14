#include "audio.h"
#include "compute.h"
#include "pipeline.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s [options] <audio.{mp3,wav,npy}>\n"
        "\n"
        "options:\n"
        "  -m DIR    model directory containing musicfm-f16.gguf, muq-f16.gguf,\n"
        "            songformer-f32.gguf (default: models)\n"
        "  -V NAME   head variant: songformer (8.333 fps, upstream) or\n"
        "            edmformer (12.5 fps); default: from GGUF metadata\n"
        "  -t N      threads (default: hardware concurrency)\n"
        "  -o PREFIX write PREFIX.txt (MSA) and PREFIX.json\n"
        "  -c        CPU only (disable GPU backend)\n"
        "  -q        quiet\n",
        argv0);
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
    if (!sf.load(model_dir + "/songformer-f32.gguf")) return 1;
    if (!variant.empty() && !sf.set_variant(variant)) {
        fprintf(stderr, "unknown variant: %s (use songformer or edmformer)\n", variant.c_str());
        return 1;
    }
    fprintf(stderr, "head variant: %s (stride %d, %.3f fps)\n",
            sf.variant.c_str(), sf.ds_stride, sf.frame_rates);

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
