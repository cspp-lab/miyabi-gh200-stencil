// work.cu — small GPU workloads for the MPS co-location exercises.
//
// Usage:
//   ./work victim ITERS [TAG]   latency of a small kernel, launched and waited
//                               for ITERS times; prints p50/p99/max in us
//   ./work noisy  SECONDS       keeps every SM it can get busy with long
//                               kernels for SECONDS (the "noisy neighbour")
//   ./work fma    KERNELS [TAG] fixed amount of FP32 FMA work (compute bound);
//                               prints elapsed time and GFLOP/s
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)

using clk = std::chrono::steady_clock;

__global__ void spin(long long cycles, int *sink)
{
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
    if (threadIdx.x == 0 && cycles < 0) *sink = 1; // never true; keeps the loop
}

constexpr int FMA_ITERS = 1 << 18;

__global__ void fma_kernel(float *out, float s)
{
    float a = threadIdx.x * 1e-3f, b = a + 1.f, c = a + 2.f, d = a + 3.f;
    for (int i = 0; i < FMA_ITERS; i++) {
        a = fmaf(a, s, 1e-7f);
        b = fmaf(b, s, 1e-7f);
        c = fmaf(c, s, 1e-7f);
        d = fmaf(d, s, 1e-7f);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a + b + c + d;
}

static int run_victim(int iters, const char *tag)
{
    int *sink;
    CK(cudaMalloc(&sink, sizeof(int)));
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    // 32 blocks x ~10 us: tiny next to the GPU, so its latency is dominated
    // by how long it has to wait for SMs, not by its own work.
    const long long cyc = 20000;
    for (int i = 0; i < 20; i++) spin<<<32, 256, 0, st>>>(cyc, sink);
    CK(cudaStreamSynchronize(st));

    std::vector<double> lat(iters);
    auto start = clk::now();
    for (int i = 0; i < iters; i++) {
        auto t0 = clk::now();
        spin<<<32, 256, 0, st>>>(cyc, sink);
        CK(cudaStreamSynchronize(st));
        lat[i] = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
    }
    double total = std::chrono::duration<double>(clk::now() - start).count();
    std::sort(lat.begin(), lat.end());
    printf("RESULT mode=victim tag=%s iters=%d p50_us=%.1f p99_us=%.1f max_us=%.1f total_s=%.3f\n",
           tag, iters, lat[iters / 2], lat[(size_t)(iters * 0.99)], lat.back(), total);
    return 0;
}

static int run_noisy(double seconds)
{
    int *sink;
    CK(cudaMalloc(&sink, sizeof(int)));
    // 132 SMs x 8 blocks x ~5 ms: once these are resident, a newcomer can
    // only start when a block retires (unless it has SMs of its own).
    const long long cyc = 10000000;
    int launched = 0;
    auto start = clk::now();
    while (std::chrono::duration<double>(clk::now() - start).count() < seconds) {
        spin<<<1056, 256>>>(cyc, sink);
        CK(cudaDeviceSynchronize());
        launched++;
    }
    printf("RESULT mode=noisy seconds=%.1f kernels=%d\n", seconds, launched);
    return 0;
}

static int run_fma(int kernels, const char *tag)
{
    const int nblk = 1056, nthr = 256;
    float *out;
    CK(cudaMalloc(&out, sizeof(float) * nblk * nthr));
    fma_kernel<<<nblk, nthr>>>(out, 0.999f); // warm-up
    CK(cudaDeviceSynchronize());
    auto t0 = clk::now();
    for (int k = 0; k < kernels; k++) fma_kernel<<<nblk, nthr>>>(out, 0.999f);
    CK(cudaDeviceSynchronize());
    double s = std::chrono::duration<double>(clk::now() - t0).count();
    double flops = 2.0 * 4 * FMA_ITERS * (double)nblk * nthr * kernels;
    printf("RESULT mode=fma tag=%s kernels=%d time_s=%.3f gflops=%.1f\n",
           tag, kernels, s, flops / s / 1e9);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s victim ITERS [TAG] | noisy SECONDS | fma KERNELS [TAG]\n", argv[0]);
        return 2;
    }
    const char *tag = argc > 3 ? argv[3] : "run";
    if (!strcmp(argv[1], "victim")) return run_victim(atoi(argv[2]), tag);
    if (!strcmp(argv[1], "noisy")) return run_noisy(atof(argv[2]));
    if (!strcmp(argv[1], "fma")) return run_fma(atoi(argv[2]), tag);
    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
