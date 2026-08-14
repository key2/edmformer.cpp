#include "mel.h"

#include <cmath>
#include <cstring>
#include <vector>

// simple iterative radix-2 complex FFT (double precision for accuracy)
static void fft(std::vector<double> & re, std::vector<double> & im) {
    const size_t n = re.size();
    // bit reversal
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * M_PI / (double) len;
        const double wr = cos(ang), wi = sin(ang);
        for (size_t i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (size_t k = 0; k < len / 2; k++) {
                const double ur = re[i + k], ui = im[i + k];
                const double vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                const double vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                const double ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}

int64_t mel_spectrogram(const float * samples, int64_t n, const mel_params & p, std::vector<float> & out) {
    const int n_fft = p.n_fft;
    const int hop   = p.hop;
    const int n_bins = n_fft / 2 + 1;
    const int pad = n_fft / 2;

    // torch.stft(center=True): n_frames = 1 + n // hop ; MusicFM then drops the last frame
    const int64_t n_frames_full = 1 + n / hop;
    const int64_t n_frames = n_frames_full - 1;
    if (n_frames <= 0) return 0;

    // reflect-padded signal
    std::vector<float> x(n + 2 * pad);
    for (int64_t i = 0; i < (int64_t) x.size(); i++) {
        int64_t j = i - pad;
        if (j < 0) j = -j;                       // reflect (without repeating edge)
        if (j >= n) j = 2 * (n - 1) - j;
        x[i] = samples[j];
    }

    std::vector<double> power((size_t) n_frames * n_bins);

    std::vector<double> re(n_fft), im(n_fft);
    for (int64_t t = 0; t < n_frames; t++) {
        const float * frame = x.data() + t * hop;
        for (int i = 0; i < n_fft; i++) {
            re[i] = (double) frame[i] * (double) p.window[i];
            im[i] = 0.0;
        }
        fft(re, im);
        double * pw = power.data() + (size_t) t * n_bins;
        for (int b = 0; b < n_bins; b++) {
            pw[b] = re[b] * re[b] + im[b] * im[b];
        }
    }

    // mel filterbank + dB + normalization; out layout [n_mels][n_frames]
    out.assign((size_t) p.n_mels * n_frames, 0.f);
    for (int m = 0; m < p.n_mels; m++) {
        const float * fb = p.fbank + (size_t) m * n_bins;
        // mel filters are sparse; find the non-zero band once
        int b0 = 0, b1 = n_bins;
        while (b0 < n_bins && fb[b0] == 0.f) b0++;
        while (b1 > b0 && fb[b1 - 1] == 0.f) b1--;
        float * dst = out.data() + (size_t) m * n_frames;
        for (int64_t t = 0; t < n_frames; t++) {
            const double * pw = power.data() + (size_t) t * n_bins;
            double acc = 0.0;
            for (int b = b0; b < b1; b++) acc += pw[b] * (double) fb[b];
            double v = acc < 1e-10 ? 1e-10 : acc;
            v = 10.0 * log10(v);
            dst[t] = (float) ((v - p.mean) / p.std_dev);
        }
    }
    return n_frames;
}
