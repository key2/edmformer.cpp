#include "audio.h"
#include "common.h"

#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#ifdef __APPLE__
#include <AudioToolbox/AudioToolbox.h>
#endif

// windowed-sinc resampler (Kaiser window), decent quality for feature extraction
static std::vector<float> resample_sinc(const std::vector<float> & in, double sr_in, double sr_out) {
    if (sr_in == sr_out) return in;

    const double ratio = sr_out / sr_in;
    const int64_t n_out = (int64_t) llround((double) in.size() * ratio);
    std::vector<float> out(n_out);

    const int TAPS = 32;  // half-width in input samples (scaled when downsampling)
    const double cutoff = std::min(1.0, ratio) * 0.97;
    const double beta = 12.0;

    auto bessel_i0 = [](double x) {
        double sum = 1.0, term = 1.0;
        for (int k = 1; k < 32; k++) {
            term *= (x / (2.0 * k)) * (x / (2.0 * k));
            sum += term;
            if (term < 1e-12 * sum) break;
        }
        return sum;
    };
    const double i0b = bessel_i0(beta);

    const double half_width = TAPS / std::min(1.0, ratio);
    for (int64_t i = 0; i < n_out; i++) {
        const double t = (double) i / ratio;  // position in input samples
        const int64_t j0 = (int64_t) ceil(t - half_width);
        const int64_t j1 = (int64_t) floor(t + half_width);
        double acc = 0.0;
        for (int64_t j = j0; j <= j1; j++) {
            const double d = (t - (double) j) * cutoff;
            const double s = d == 0.0 ? 1.0 : sin(M_PI * d) / (M_PI * d);
            const double u = (t - (double) j) / half_width;
            if (u < -1.0 || u > 1.0) continue;
            const double w = bessel_i0(beta * sqrt(1.0 - u * u)) / i0b;
            if (j >= 0 && j < (int64_t) in.size()) acc += s * w * cutoff * in[j];
        }
        out[i] = (float) acc;
    }
    return out;
}

#ifdef __APPLE__
// decode any CoreAudio-supported container (m4a/aac/mp3/...) to float32 PCM
static OSStatus open_ext_audio(const char * path, ExtAudioFileRef * ea) {
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        nullptr, (const UInt8 *) path, (CFIndex) strlen(path), false);
    OSStatus st = ExtAudioFileOpenURL(url, ea);
    CFRelease(url);
    return st;
}

static std::vector<float> decode_coreaudio(const std::string & path, double & sr, uint32_t & channels) {
    char abs_path[PATH_MAX];
    if (!realpath(path.c_str(), abs_path)) return {};

    ExtAudioFileRef ea = nullptr;
    OSStatus st = open_ext_audio(abs_path, &ea);

    std::string tmp_link;
    if (st != noErr) {
        // CoreAudio uses the extension as a container hint; files with a wrong
        // extension (e.g. MP4/AAC named .mp3) fail. Detect ISO-BMFF ("ftyp" at
        // offset 4) and retry through a correctly named temp symlink.
        FILE * f = fopen(abs_path, "rb");
        if (!f) return {};
        unsigned char hdr[12] = {0};
        fread(hdr, 1, 12, f);
        fclose(f);
        const char * ext = memcmp(hdr + 4, "ftyp", 4) == 0 ? ".mp4" : ".m4a";
        tmp_link = std::string("/tmp/edmformer_audio_") + std::to_string(getpid()) + ext;
        unlink(tmp_link.c_str());
        if (symlink(abs_path, tmp_link.c_str()) != 0) return {};
        st = open_ext_audio(tmp_link.c_str(), &ea);
        if (st != noErr) {
            unlink(tmp_link.c_str());
            return {};
        }
    }

    AudioStreamBasicDescription in_fmt;
    UInt32 sz = sizeof(in_fmt);
    ExtAudioFileGetProperty(ea, kExtAudioFileProperty_FileDataFormat, &sz, &in_fmt);

    AudioStreamBasicDescription fmt = {};
    fmt.mSampleRate       = in_fmt.mSampleRate;
    fmt.mFormatID         = kAudioFormatLinearPCM;
    fmt.mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    fmt.mChannelsPerFrame = in_fmt.mChannelsPerFrame;
    fmt.mBitsPerChannel   = 32;
    fmt.mBytesPerFrame    = 4 * fmt.mChannelsPerFrame;
    fmt.mBytesPerPacket   = fmt.mBytesPerFrame;
    fmt.mFramesPerPacket  = 1;
    if (ExtAudioFileSetProperty(ea, kExtAudioFileProperty_ClientDataFormat, sizeof(fmt), &fmt) != noErr) {
        ExtAudioFileDispose(ea);
        return {};
    }

    sr = fmt.mSampleRate;
    channels = fmt.mChannelsPerFrame;

    std::vector<float> data;
    const UInt32 CHUNK = 65536;
    std::vector<float> buf((size_t) CHUNK * channels);
    while (true) {
        AudioBufferList abl;
        abl.mNumberBuffers = 1;
        abl.mBuffers[0].mNumberChannels = channels;
        abl.mBuffers[0].mDataByteSize = CHUNK * fmt.mBytesPerFrame;
        abl.mBuffers[0].mData = buf.data();
        UInt32 n_frames = CHUNK;
        if (ExtAudioFileRead(ea, &n_frames, &abl) != noErr || n_frames == 0) break;
        data.insert(data.end(), buf.begin(), buf.begin() + (size_t) n_frames * channels);
    }
    ExtAudioFileDispose(ea);
    if (!tmp_link.empty()) unlink(tmp_link.c_str());
    return data;
}
#endif

static std::vector<float> downmix(const float * data, uint64_t frames, uint32_t channels) {
    std::vector<float> mono(frames);
    for (uint64_t i = 0; i < frames; i++) {
        double acc = 0.0;
        for (uint32_t c = 0; c < channels; c++) acc += data[i * channels + c];
        mono[i] = (float) (acc / channels);
    }
    return mono;
}

std::vector<float> load_audio(const std::string & path, int target_sr) {
    std::vector<float> mono;
    double sr = target_sr;

    if (path.size() > 4 && path.substr(path.size() - 4) == ".npy") {
        npy_array a;
        if (!npy_load(path, a)) return {};
        return a.data;  // assumed already mono @ target_sr
    }

    bool decoded = false;

    if (path.size() > 4 && path.substr(path.size() - 4) == ".mp3") {
        drmp3 mp3;
        if (drmp3_init_file(&mp3, path.c_str(), nullptr)) {
            const uint64_t frames = drmp3_get_pcm_frame_count(&mp3);
            if (frames > 0) {
                std::vector<float> buf(frames * mp3.channels);
                drmp3_read_pcm_frames_f32(&mp3, frames, buf.data());
                mono = downmix(buf.data(), frames, mp3.channels);
                sr = mp3.sampleRate;
                decoded = true;
            }
            drmp3_uninit(&mp3);
        }
    } else if (path.size() > 4 && path.substr(path.size() - 4) == ".wav") {
        drwav wav;
        if (drwav_init_file(&wav, path.c_str(), nullptr)) {
            std::vector<float> buf((size_t) wav.totalPCMFrameCount * wav.channels);
            drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, buf.data());
            mono = downmix(buf.data(), wav.totalPCMFrameCount, wav.channels);
            sr = wav.sampleRate;
            decoded = true;
            drwav_uninit(&wav);
        }
    }

#ifdef __APPLE__
    if (!decoded) {
        // CoreAudio handles m4a/aac/mp4 (and files with wrong extensions)
        double ca_sr = 0;
        uint32_t ch = 0;
        std::vector<float> buf = decode_coreaudio(path, ca_sr, ch);
        if (!buf.empty()) {
            mono = downmix(buf.data(), buf.size() / ch, ch);
            sr = ca_sr;
            decoded = true;
        }
    }
#endif

    if (!decoded) {
        fprintf(stderr, "failed to decode audio: %s\n", path.c_str());
        return {};
    }

    return resample_sinc(mono, sr, target_sr);
}
