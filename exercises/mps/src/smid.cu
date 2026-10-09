// smid.cu — report how many SMs this process sees and which SMs its blocks
// actually ran on, plus the time of a fixed amount of spin work.
//
// Usage: ./smid [TAG]
//
// Under MPS static partitioning the SM count shrinks to the partition size
// (8 SMs per chunk on GH200) and %smid is renumbered from 0 inside the
// partition, so two partitions both report smid=[0..N-1].
#include <cuda_runtime.h>
#include <cstdio>
#include <set>

__global__ void spin(unsigned *smid, long long cycles)
{
    unsigned id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    if (threadIdx.x == 0) smid[blockIdx.x] = id;
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}

int main(int argc, char **argv)
{
    const char *tag = argc > 1 ? argv[1] : "run";
    const int nblk = 1056; // 8 blocks per SM on a full 132-SM GH200

    int nsm = 0;
    cudaError_t e = cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    if (e != cudaSuccess) {
        printf("%s: error=%s\n", tag, cudaGetErrorString(e));
        return 1;
    }

    unsigned *d, h[nblk];
    cudaMalloc(&d, sizeof h);
    spin<<<nblk, 128>>>(d, 1000); // warm-up

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    spin<<<nblk, 128>>>(d, 2000000);
    cudaEventRecord(b);
    e = cudaEventSynchronize(b);
    if (e != cudaSuccess) {
        printf("%s: error=%s\n", tag, cudaGetErrorString(e));
        return 1;
    }
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);

    std::set<unsigned> s(h, h + nblk);
    printf("RESULT tag=%s attr_SMs=%d used_SMs=%zu smid=[%u..%u] time_ms=%.2f\n",
           tag, nsm, s.size(), *s.begin(), *s.rbegin(), ms);
    return 0;
}
