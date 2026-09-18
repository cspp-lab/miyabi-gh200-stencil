// 3D 7-point Jacobi diffusion stencil, generalized to a 3D process decomposition
// (Px * Py * Pz ranks) on multi-node GH200 (Grace Hopper Superchip).
//
// Miyabi-G has 1 GPU/node, so one MPI rank == one node == one GPU. The process
// grid is built with MPI_Dims_create + a Cartesian communicator, which works for
// any rank count (tested at 4 ranks here; the decomposition itself places no
// upper bound on rank count, so 64-way and beyond is the same code path).
//
// Because the domain is now decomposed along X, Y, *and* Z, none of the 6 face
// halos are contiguous in device memory any more (each local array is padded by
// 1 ghost layer on every side), so every face is packed into a contiguous buffer
// before sending and unpacked into the ghost layer on receipt. Only the 7-point
// (face-neighbor-only) stencil is needed, so edges/corners never need to be
// exchanged.
//
// Boundary condition: zero-Dirichlet on the 6 faces of the *global* domain,
// enforced by never writing those cells (they stay at their initial value of 0).
// A rank that owns a global face has MPI_PROC_NULL as that neighbor; a
// PROC_NULL recv leaves the ghost layer untouched at its pre-zeroed value,
// which reproduces the global Dirichlet face for free.
//
// Runs two execution modes back to back, in a single MPI_Init/MPI_Finalize
// session (some launchers here don't support a clean second MPI_Init in a
// fresh process after the first one exits, so both variants have to share
// one run rather than being two separate invocations), to measure the
// effect of communication/computation overlap:
//   mode 0 (naive)   : pack -> blocking Sendrecv x6 -> unpack -> one kernel
//                       over the whole owned box.
//   mode 1 (overlap) : pack -> non-blocking Isend/Irecv x6 (12 reqs); the
//                       interior (untouched by any halo) is computed
//                       concurrently on another stream; once the halo has
//                       arrived, unpack and update the 1-cell-thick shell
//                       next to each of the 6 faces.
//
// Usage: ./stencil3d NX NY NZ_GLOBAL ITERS [PX PY PZ]
//   PX*PY*PZ must equal the rank count if given; omit (or pass 0 0 0) for
//   automatic factorization via MPI_Dims_create. NX/PX, NY/PY, NZ/PZ must
//   be exact. Launch under whatever this cluster's job wrapper expects one
//   MPI rank per process to come from (here: one rank per node already).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "[rank %d] CUDA error %s at %s:%d\n", g_rank, cudaGetErrorString(err_), __FILE__, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

static int g_rank = 0;

// ---- stencil update over the local sub-box [x0,x1]x[y0,y1]x[z0,z1] (1-based, inclusive) ----
__global__ void stencil_kernel(const double* __restrict__ uold, double* __restrict__ unew,
                                int nxl, int nyl, int nzl,
                                int gx0, int gy0, int gz0,
                                int NXg, int NYg, int NZg,
                                int x0, int x1, int y0, int y1, int z0, int z1,
                                double lambda)
{
    int lx = x0 + blockIdx.x * blockDim.x + threadIdx.x;
    int ly = y0 + blockIdx.y * blockDim.y + threadIdx.y;
    int lz = z0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (lx > x1 || ly > y1 || lz > z1) return;

    int gx = gx0 + lx - 1, gy = gy0 + ly - 1, gz = gz0 + lz - 1;
    if (gx == 0 || gx == NXg - 1 || gy == 0 || gy == NYg - 1 || gz == 0 || gz == NZg - 1)
        return; // global Dirichlet face: leave unchanged (stays 0 forever)

    size_t sx = (size_t)(nxl + 2);
    size_t plane = sx * (size_t)(nyl + 2);
    size_t idx = (size_t)lx + sx * (size_t)ly + plane * (size_t)lz;

    double c  = uold[idx];
    double xm = uold[idx - 1];
    double xp = uold[idx + 1];
    double ym = uold[idx - sx];
    double yp = uold[idx + sx];
    double zm = uold[idx - plane];
    double zp = uold[idx + plane];

    unew[idx] = c + lambda * (xm + xp + ym + yp + zm + zp - 6.0 * c);
}

// ---- pack/unpack the 6 faces into/from contiguous buffers ----
__global__ void pack_x(const double* u, double* buf, int nxl, int nyl, int nzl, int lx) {
    int ly = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int lz = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (ly > nyl || lz > nzl) return;
    size_t sx = nxl + 2, plane = sx * (nyl + 2);
    buf[(ly - 1) + nyl * (lz - 1)] = u[lx + sx * ly + plane * lz];
}
__global__ void unpack_x(double* u, const double* buf, int nxl, int nyl, int nzl, int lx) {
    int ly = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int lz = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (ly > nyl || lz > nzl) return;
    size_t sx = nxl + 2, plane = sx * (nyl + 2);
    u[lx + sx * ly + plane * lz] = buf[(ly - 1) + nyl * (lz - 1)];
}
__global__ void pack_y(const double* u, double* buf, int nxl, int nyl, int nzl, int ly) {
    int lx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int lz = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (lx > nxl || lz > nzl) return;
    size_t sx = nxl + 2, plane = sx * (nyl + 2);
    buf[(lx - 1) + nxl * (lz - 1)] = u[lx + sx * ly + plane * lz];
}
__global__ void unpack_y(double* u, const double* buf, int nxl, int nyl, int nzl, int ly) {
    int lx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int lz = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (lx > nxl || lz > nzl) return;
    size_t sx = nxl + 2, plane = sx * (nyl + 2);
    u[lx + sx * ly + plane * lz] = buf[(lx - 1) + nxl * (lz - 1)];
}
__global__ void pack_z(const double* u, double* buf, int nxl, int nyl, int nzl, int lz) {
    int lx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int ly = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (lx > nxl || ly > nyl) return;
    size_t sx = nxl + 2, plane = sx * (nyl + 2);
    buf[(lx - 1) + nxl * (ly - 1)] = u[lx + sx * ly + plane * lz];
}
__global__ void unpack_z(double* u, const double* buf, int nxl, int nyl, int nzl, int lz) {
    int lx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int ly = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (lx > nxl || ly > nyl) return;
    size_t sx = nxl + 2, plane = sx * (nyl + 2);
    u[lx + sx * ly + plane * lz] = buf[(lx - 1) + nxl * (ly - 1)];
}

// ---- sum of squares over the owned (non-ghost) box, for a cheap stability check ----
__global__ void sumsq_kernel(const double* __restrict__ u, int nxl, int nyl, int nzl, double* result) {
    int lx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int ly = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int lz = blockIdx.z * blockDim.z + threadIdx.z + 1;
    double v = 0.0;
    if (lx <= nxl && ly <= nyl && lz <= nzl) {
        size_t sx = nxl + 2, plane = sx * (nyl + 2);
        double val = u[lx + sx * ly + plane * lz];
        v = val * val;
    }
    extern __shared__ double sdata[];
    int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    sdata[tid] = v;
    __syncthreads();
    for (int s = (blockDim.x * blockDim.y * blockDim.z) / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}

static void pack_all(const double* u, double* sxm, double* sxp, double* sym, double* syp,
                      double* szm, double* szp, int nxl, int nyl, int nzl, cudaStream_t st) {
    dim3 byz((nyl + 15) / 16, (nzl + 15) / 16), bxz((nxl + 15) / 16, (nzl + 15) / 16), bxy((nxl + 15) / 16, (nyl + 15) / 16);
    dim3 t16(16, 16);
    pack_x<<<byz, t16, 0, st>>>(u, sxm, nxl, nyl, nzl, 1);
    pack_x<<<byz, t16, 0, st>>>(u, sxp, nxl, nyl, nzl, nxl);
    pack_y<<<bxz, t16, 0, st>>>(u, sym, nxl, nyl, nzl, 1);
    pack_y<<<bxz, t16, 0, st>>>(u, syp, nxl, nyl, nzl, nyl);
    pack_z<<<bxy, t16, 0, st>>>(u, szm, nxl, nyl, nzl, 1);
    pack_z<<<bxy, t16, 0, st>>>(u, szp, nxl, nyl, nzl, nzl);
}
static void unpack_all(double* u, const double* rxm, const double* rxp, const double* rym, const double* ryp,
                        const double* rzm, const double* rzp, int nxl, int nyl, int nzl, cudaStream_t st) {
    dim3 byz((nyl + 15) / 16, (nzl + 15) / 16), bxz((nxl + 15) / 16, (nzl + 15) / 16), bxy((nxl + 15) / 16, (nyl + 15) / 16);
    dim3 t16(16, 16);
    unpack_x<<<byz, t16, 0, st>>>(u, rxm, nxl, nyl, nzl, 0);
    unpack_x<<<byz, t16, 0, st>>>(u, rxp, nxl, nyl, nzl, nxl + 1);
    unpack_y<<<bxz, t16, 0, st>>>(u, rym, nxl, nyl, nzl, 0);
    unpack_y<<<bxz, t16, 0, st>>>(u, ryp, nxl, nyl, nzl, nyl + 1);
    unpack_z<<<bxy, t16, 0, st>>>(u, rzm, nxl, nyl, nzl, 0);
    unpack_z<<<bxy, t16, 0, st>>>(u, rzp, nxl, nyl, nzl, nzl + 1);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    g_rank = rank;

    if (argc < 5) {
        if (rank == 0)
            fprintf(stderr, "usage: %s NX NY NZ_GLOBAL ITERS [PX PY PZ]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int NXg = atoi(argv[1]), NYg = atoi(argv[2]), NZg = atoi(argv[3]);
    int ITERS = atoi(argv[4]);
    // Runs both the naive and overlap modes back to back, in the same
    // MPI session: this runtime's ORTE/PMIx doesn't support a clean second
    // MPI_Init after MPI_Finalize within the same sandboxed job step, so a
    // separate process per mode (two invocations of this binary) isn't an
    // option here - both comparisons have to happen inside one run.

    int dims[3] = {0, 0, 0};
    if (argc >= 8) {
        dims[0] = atoi(argv[5]); dims[1] = atoi(argv[6]); dims[2] = atoi(argv[7]);
        if ((long)dims[0] * dims[1] * dims[2] != nprocs) {
            if (rank == 0) fprintf(stderr, "PX*PY*PZ (%d*%d*%d) must equal nranks (%d)\n", dims[0], dims[1], dims[2], nprocs);
            MPI_Finalize(); return 1;
        }
    } else {
        MPI_Dims_create(nprocs, 3, dims); // works for any rank count, e.g. 64+
    }

    int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart);
    MPI_Comm_rank(cart, &rank);
    g_rank = rank;
    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);

    if (NXg % dims[0] || NYg % dims[1] || NZg % dims[2]) {
        if (rank == 0)
            fprintf(stderr, "domain %dx%dx%d not evenly divisible by process grid %dx%dx%d\n",
                    NXg, NYg, NZg, dims[0], dims[1], dims[2]);
        MPI_Finalize(); return 1;
    }
    int nxl = NXg / dims[0], nyl = NYg / dims[1], nzl = NZg / dims[2];
    int gx0 = coords[0] * nxl, gy0 = coords[1] * nyl, gz0 = coords[2] * nzl;

    int xm, xp, ym, yp, zm, zp;
    MPI_Cart_shift(cart, 0, 1, &xm, &xp);
    MPI_Cart_shift(cart, 1, 1, &ym, &yp);
    MPI_Cart_shift(cart, 2, 1, &zm, &zp);

    CUDA_CHECK(cudaSetDevice(0)); // 1 GH200 GPU/node with mpiprocs=1: device 0 is always ours

    size_t sx = nxl + 2, plane_pad = sx * (nyl + 2);
    size_t nelem = plane_pad * (size_t)(nzl + 2);
    size_t nbytes = nelem * sizeof(double);

    double *d_u0, *d_u1;
    CUDA_CHECK(cudaMalloc(&d_u0, nbytes));
    CUDA_CHECK(cudaMalloc(&d_u1, nbytes));
    CUDA_CHECK(cudaMemset(d_u0, 0, nbytes));
    CUDA_CHECK(cudaMemset(d_u1, 0, nbytes));

    size_t sizeX = (size_t)nyl * nzl, sizeY = (size_t)nxl * nzl, sizeZ = (size_t)nxl * nyl;
    double *sxm_, *sxp_, *sym_, *syp_, *szm_, *szp_;
    double *rxm_, *rxp_, *rym_, *ryp_, *rzm_, *rzp_;
    CUDA_CHECK(cudaMalloc(&sxm_, sizeX * sizeof(double))); CUDA_CHECK(cudaMalloc(&sxp_, sizeX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&rxm_, sizeX * sizeof(double))); CUDA_CHECK(cudaMalloc(&rxp_, sizeX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&sym_, sizeY * sizeof(double))); CUDA_CHECK(cudaMalloc(&syp_, sizeY * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&rym_, sizeY * sizeof(double))); CUDA_CHECK(cudaMalloc(&ryp_, sizeY * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&szm_, sizeZ * sizeof(double))); CUDA_CHECK(cudaMalloc(&szp_, sizeZ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&rzm_, sizeZ * sizeof(double))); CUDA_CHECK(cudaMalloc(&rzp_, sizeZ * sizeof(double)));

    const double lambda = 1.0 / 7.0; // < 1/6 explicit-scheme stability bound, with margin

    dim3 blk(8, 8, 4);
    cudaStream_t s_interior, s_boundary;
    CUDA_CHECK(cudaStreamCreate(&s_interior));
    CUDA_CHECK(cudaStreamCreate(&s_boundary));

    double *d_sumsq;
    CUDA_CHECK(cudaMalloc(&d_sumsq, sizeof(double)));

    double* h_slab = (double*)malloc((size_t)nxl * nyl * sizeof(double));

  for (int MODE = 0; MODE <= 1; ++MODE) {
    // Initial condition: Gaussian bump at the global domain center, built on the
    // Grace CPU (OpenMP over its cores) one local Z-layer at a time, then staged
    // to the GPU across NVLink C2C with a single cudaMemcpy2D per layer (handles
    // the X ghost padding without a per-row copy). Rebuilt fresh for each mode
    // so naive and overlap start from the identical state.
    CUDA_CHECK(cudaMemset(d_u0, 0, nbytes));
    CUDA_CHECK(cudaMemset(d_u1, 0, nbytes));
    {
        double cx = NXg * 0.5, cy = NYg * 0.5, cz = NZg * 0.5;
        double sigma2 = 0.02 * (double)NXg * (double)NXg;
        for (int lz = 1; lz <= nzl; ++lz) {
            int gz = gz0 + (lz - 1);
            #pragma omp parallel for collapse(2) schedule(static)
            for (int ly = 0; ly < nyl; ++ly) {
                for (int lx = 0; lx < nxl; ++lx) {
                    int gx = gx0 + lx, gy = gy0 + ly;
                    double val = 0.0;
                    if (gx > 0 && gx < NXg - 1 && gy > 0 && gy < NYg - 1 && gz > 0 && gz < NZg - 1) {
                        double dx = gx - cx, dy = gy - cy, dz = gz - cz;
                        val = exp(-(dx * dx + dy * dy + dz * dz) / sigma2);
                    }
                    h_slab[lx + nxl * ly] = val;
                }
            }
            double* dst = d_u0 + (1 + sx * 1 + plane_pad * (size_t)lz); // (lx=1, ly=1, lz)
            CUDA_CHECK(cudaMemcpy2D(dst, sx * sizeof(double), h_slab, nxl * sizeof(double),
                                     nxl * sizeof(double), nyl, cudaMemcpyHostToDevice));
        }
    }

    double *u_old = d_u0, *u_new = d_u1;
    int report_every = (ITERS >= 5) ? ITERS / 5 : 1;

    MPI_Barrier(cart);
    double t_start = MPI_Wtime();
    double t_comm = 0.0;

    for (int it = 0; it < ITERS; ++it) {
        double tc0 = MPI_Wtime();

        if (MODE == 0) {
            pack_all(u_old, sxm_, sxp_, sym_, syp_, szm_, szp_, nxl, nyl, nzl, 0);
            CUDA_CHECK(cudaDeviceSynchronize());

            MPI_Sendrecv(sxm_, sizeX, MPI_DOUBLE, xm, 0, rxp_, sizeX, MPI_DOUBLE, xp, 0, cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(sxp_, sizeX, MPI_DOUBLE, xp, 1, rxm_, sizeX, MPI_DOUBLE, xm, 1, cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(sym_, sizeY, MPI_DOUBLE, ym, 2, ryp_, sizeY, MPI_DOUBLE, yp, 2, cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(syp_, sizeY, MPI_DOUBLE, yp, 3, rym_, sizeY, MPI_DOUBLE, ym, 3, cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(szm_, sizeZ, MPI_DOUBLE, zm, 4, rzp_, sizeZ, MPI_DOUBLE, zp, 4, cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(szp_, sizeZ, MPI_DOUBLE, zp, 5, rzm_, sizeZ, MPI_DOUBLE, zm, 5, cart, MPI_STATUS_IGNORE);

            unpack_all(u_old, rxm_, rxp_, rym_, ryp_, rzm_, rzp_, nxl, nyl, nzl, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            t_comm += MPI_Wtime() - tc0;

            dim3 grid_full((nxl + blk.x - 1) / blk.x, (nyl + blk.y - 1) / blk.y, (nzl + blk.z - 1) / blk.z);
            stencil_kernel<<<grid_full, blk>>>(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg,
                                                1, nxl, 1, nyl, 1, nzl, lambda);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            pack_all(u_old, sxm_, sxp_, sym_, syp_, szm_, szp_, nxl, nyl, nzl, 0);
            CUDA_CHECK(cudaDeviceSynchronize());

            MPI_Request req[12];
            MPI_Irecv(rxp_, sizeX, MPI_DOUBLE, xp, 0, cart, &req[0]);
            MPI_Irecv(rxm_, sizeX, MPI_DOUBLE, xm, 1, cart, &req[1]);
            MPI_Isend(sxm_, sizeX, MPI_DOUBLE, xm, 0, cart, &req[2]);
            MPI_Isend(sxp_, sizeX, MPI_DOUBLE, xp, 1, cart, &req[3]);
            MPI_Irecv(ryp_, sizeY, MPI_DOUBLE, yp, 2, cart, &req[4]);
            MPI_Irecv(rym_, sizeY, MPI_DOUBLE, ym, 3, cart, &req[5]);
            MPI_Isend(sym_, sizeY, MPI_DOUBLE, ym, 2, cart, &req[6]);
            MPI_Isend(syp_, sizeY, MPI_DOUBLE, yp, 3, cart, &req[7]);
            MPI_Irecv(rzp_, sizeZ, MPI_DOUBLE, zp, 4, cart, &req[8]);
            MPI_Irecv(rzm_, sizeZ, MPI_DOUBLE, zm, 5, cart, &req[9]);
            MPI_Isend(szm_, sizeZ, MPI_DOUBLE, zm, 4, cart, &req[10]);
            MPI_Isend(szp_, sizeZ, MPI_DOUBLE, zp, 5, cart, &req[11]);

            int ix0 = 2, ix1 = nxl - 1, iy0 = 2, iy1 = nyl - 1, iz0 = 2, iz1 = nzl - 1;
            if (ix1 >= ix0 && iy1 >= iy0 && iz1 >= iz0) {
                dim3 grid_int((ix1 - ix0 + 1 + blk.x - 1) / blk.x,
                               (iy1 - iy0 + 1 + blk.y - 1) / blk.y,
                               (iz1 - iz0 + 1 + blk.z - 1) / blk.z);
                stencil_kernel<<<grid_int, blk, 0, s_interior>>>(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0,
                                                                  NXg, NYg, NZg, ix0, ix1, iy0, iy1, iz0, iz1, lambda);
                CUDA_CHECK(cudaGetLastError());
            }

            MPI_Waitall(12, req, MPI_STATUSES_IGNORE);
            t_comm += MPI_Wtime() - tc0;

            unpack_all(u_old, rxm_, rxp_, rym_, ryp_, rzm_, rzp_, nxl, nyl, nzl, s_boundary);

            // 1-cell-thick shell next to each of the 6 faces (harmlessly
            // recomputes the 12 edges/8 corners more than once: same
            // deterministic formula, so redundant but not incorrect).
            dim3 gyz((nyl + blk.y - 1) / blk.y, (nzl + blk.z - 1) / blk.z, 1);
            dim3 gxz((nxl + blk.x - 1) / blk.x, (nzl + blk.z - 1) / blk.z, 1);
            dim3 gxy((nxl + blk.x - 1) / blk.x, (nyl + blk.y - 1) / blk.y, 1);
            dim3 b1(blk.x, blk.y, 1);
            stencil_kernel<<<gyz, dim3(1, blk.y, blk.z), 0, s_boundary>>>(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg, 1, 1, 1, nyl, 1, nzl, lambda);
            stencil_kernel<<<gyz, dim3(1, blk.y, blk.z), 0, s_boundary>>>(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg, nxl, nxl, 1, nyl, 1, nzl, lambda);
            stencil_kernel<<<gxz, dim3(blk.x, 1, blk.z), 0, s_boundary>>>(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg, 1, nxl, 1, 1, 1, nzl, lambda);
            stencil_kernel<<<gxz, dim3(blk.x, 1, blk.z), 0, s_boundary>>>(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg, 1, nxl, nyl, nyl, 1, nzl, lambda);
            stencil_kernel<<<gxy, b1, 0, s_boundary>>>(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg, 1, nxl, 1, nyl, 1, 1, lambda);
            stencil_kernel<<<gxy, b1, 0, s_boundary>>>(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg, 1, nxl, 1, nyl, nzl, nzl, lambda);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaStreamSynchronize(s_interior));
            CUDA_CHECK(cudaStreamSynchronize(s_boundary));
        }

        double* tmp = u_old; u_old = u_new; u_new = tmp;

        if ((it + 1) % report_every == 0 || it == ITERS - 1) {
            CUDA_CHECK(cudaMemset(d_sumsq, 0, sizeof(double)));
            dim3 gb((nxl + blk.x - 1) / blk.x, (nyl + blk.y - 1) / blk.y, (nzl + blk.z - 1) / blk.z);
            size_t shmem = blk.x * blk.y * blk.z * sizeof(double);
            sumsq_kernel<<<gb, blk, shmem>>>(u_old, nxl, nyl, nzl, d_sumsq);
            CUDA_CHECK(cudaGetLastError());
            double local_sumsq;
            CUDA_CHECK(cudaMemcpy(&local_sumsq, d_sumsq, sizeof(double), cudaMemcpyDeviceToHost));
            double global_sumsq;
            MPI_Reduce(&local_sumsq, &global_sumsq, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
            if (rank == 0)
                printf("  step %5d  L2(u) = %.6e\n", it + 1, sqrt(global_sumsq));
        }
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;
    double max_time, max_comm;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    MPI_Reduce(&t_comm,     &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    if (rank == 0) {
        double npoints = (double)NXg * NYg * NZg;
        double gflops = 9.0 * npoints * ITERS / max_time / 1e9;   // 9 flops/point/iter
        double gbps   = 16.0 * npoints * ITERS / max_time / 1e9;  // 1 read + 1 write, 8B doubles
        printf("RESULT impl=cuda mode=%s ranks=%d grid=%dx%dx%d domain=%dx%dx%d iters=%d time_s=%.4f comm_s=%.4f gflops=%.2f gbps=%.2f\n",
               MODE == 0 ? "naive" : "overlap", nprocs, dims[0], dims[1], dims[2],
               NXg, NYg, NZg, ITERS, max_time, max_comm, gflops, gbps);
    }
  } // for MODE
    free(h_slab);

    cudaFree(d_sumsq);
    cudaFree(d_u0); cudaFree(d_u1);
    cudaFree(sxm_); cudaFree(sxp_); cudaFree(rxm_); cudaFree(rxp_);
    cudaFree(sym_); cudaFree(syp_); cudaFree(rym_); cudaFree(ryp_);
    cudaFree(szm_); cudaFree(szp_); cudaFree(rzm_); cudaFree(rzp_);
    cudaStreamDestroy(s_interior);
    cudaStreamDestroy(s_boundary);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
