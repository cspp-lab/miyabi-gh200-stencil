// 3D 7-point Jacobi diffusion stencil, OpenACC port of stencil3d.cu, generalized
// to a 3D process decomposition (Px * Py * Pz ranks) on multi-node GH200
// (Grace Hopper Superchip).
//
// This is a directive-based (OpenACC) counterpart to src/stencil3d.cu, built to
// compare against the hand-written CUDA version on identical domains/iteration
// counts. Decomposition, halo layout, boundary handling, both execution modes,
// and the RESULT line format are all intentionally identical to stencil3d.cu so
// the two binaries' output lines are directly comparable; only the on-GPU
// kernels are re-expressed as OpenACC compute constructs instead of CUDA
// kernels, and CUDA streams become OpenACC async queues.
//
// Miyabi-G has 1 GPU/node, so one MPI rank == one node == one GPU. The process
// grid is built with MPI_Dims_create + a Cartesian communicator, which works for
// any rank count (tested at 4 ranks here; the decomposition itself places no
// upper bound on rank count, so 64-way and beyond is the same code path).
//
// Because the domain is decomposed along X, Y, *and* Z, none of the 6 face
// halos are contiguous in device memory (each local array is padded by 1 ghost
// layer on every side), so every face is packed into a contiguous buffer before
// sending and unpacked into the ghost layer on receipt. Only the 7-point
// (face-neighbor-only) stencil is needed, so edges/corners never need to be
// exchanged.
//
// Boundary condition: zero-Dirichlet on the 6 faces of the *global* domain,
// enforced by never writing those cells (they stay at their initial value of 0).
// A rank that owns a global face has MPI_PROC_NULL as that neighbor; a
// PROC_NULL recv leaves the ghost layer untouched at its pre-zeroed value,
// which reproduces the global Dirichlet face for free.
//
// MPI calls are handed device pointers directly (`acc host_data use_device`),
// so this requires GPU-aware MPI (nv-hpcx here), exactly like the CUDA version.
//
// Runs two execution modes back to back, in a single MPI_Init/MPI_Finalize
// session (some launchers here don't support a clean second MPI_Init in a
// fresh process after the first one exits, so both variants have to share
// one run rather than being two separate invocations), to measure the
// effect of communication/computation overlap:
//   mode 0 (naive)   : pack -> blocking Sendrecv x6 -> unpack -> one compute
//                       region over the whole owned box.
//   mode 1 (overlap) : pack -> non-blocking Isend/Irecv x6 (12 reqs); the
//                       interior (untouched by any halo) is computed
//                       concurrently on another async queue; once the halo
//                       has arrived, unpack and update the 1-cell-thick
//                       shell next to each of the 6 faces.
//
// Usage: ./stencil3d_acc NX NY NZ_GLOBAL ITERS [PX PY PZ]
//   PX*PY*PZ must equal the rank count if given; omit (or pass 0 0 0) for
//   automatic factorization via MPI_Dims_create. NX/PX, NY/PY, NZ/PZ must
//   be exact. Launch under whatever this cluster's job wrapper expects one
//   MPI rank per process to come from (here: one rank per node already).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <openacc.h>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Two async queues, mirroring the CUDA version's s_interior / s_boundary
// streams: the interior update overlaps with the halo exchange, while the
// pack/unpack/boundary-shell work is serialized on its own queue.
static const int QUEUE_INTERIOR = 1;
static const int QUEUE_BOUNDARY = 2;

// ---- stencil update over the local sub-box [x0,x1]x[y0,y1]x[z0,z1] (1-based, inclusive) ----
static void stencil_update(const double* uold, double* unew,
                            int nxl, int nyl, int nzl,
                            int gx0, int gy0, int gz0,
                            int NXg, int NYg, int NZg,
                            int x0, int x1, int y0, int y1, int z0, int z1,
                            double lambda, int queue)
{
    size_t sx = (size_t)(nxl + 2);
    size_t plane = sx * (size_t)(nyl + 2);
    size_t nelem = plane * (size_t)(nzl + 2);

    #pragma acc parallel loop collapse(3) present(uold[0:nelem], unew[0:nelem]) async(queue)
    for (int lz = z0; lz <= z1; ++lz) {
        for (int ly = y0; ly <= y1; ++ly) {
            for (int lx = x0; lx <= x1; ++lx) {
                int gx = gx0 + lx - 1, gy = gy0 + ly - 1, gz = gz0 + lz - 1;
                if (gx == 0 || gx == NXg - 1 || gy == 0 || gy == NYg - 1 || gz == 0 || gz == NZg - 1)
                    continue; // global Dirichlet face: leave unchanged (stays 0 forever)

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
        }
    }
}

// ---- pack/unpack the 6 faces into/from contiguous buffers ----
static void pack_x(const double* u, double* buf, int nxl, int nyl, int nzl, int lx, int queue) {
    size_t sx = nxl + 2, plane = sx * (nyl + 2), nelem = plane * (nzl + 2);
    size_t nbuf = (size_t)nyl * nzl;
    #pragma acc parallel loop collapse(2) present(u[0:nelem], buf[0:nbuf]) async(queue)
    for (int lz = 1; lz <= nzl; ++lz)
        for (int ly = 1; ly <= nyl; ++ly)
            buf[(ly - 1) + nyl * (lz - 1)] = u[lx + sx * ly + plane * lz];
}
static void unpack_x(double* u, const double* buf, int nxl, int nyl, int nzl, int lx, int queue) {
    size_t sx = nxl + 2, plane = sx * (nyl + 2), nelem = plane * (nzl + 2);
    size_t nbuf = (size_t)nyl * nzl;
    #pragma acc parallel loop collapse(2) present(u[0:nelem], buf[0:nbuf]) async(queue)
    for (int lz = 1; lz <= nzl; ++lz)
        for (int ly = 1; ly <= nyl; ++ly)
            u[lx + sx * ly + plane * lz] = buf[(ly - 1) + nyl * (lz - 1)];
}
static void pack_y(const double* u, double* buf, int nxl, int nyl, int nzl, int ly, int queue) {
    size_t sx = nxl + 2, plane = sx * (nyl + 2), nelem = plane * (nzl + 2);
    size_t nbuf = (size_t)nxl * nzl;
    #pragma acc parallel loop collapse(2) present(u[0:nelem], buf[0:nbuf]) async(queue)
    for (int lz = 1; lz <= nzl; ++lz)
        for (int lx = 1; lx <= nxl; ++lx)
            buf[(lx - 1) + nxl * (lz - 1)] = u[lx + sx * ly + plane * lz];
}
static void unpack_y(double* u, const double* buf, int nxl, int nyl, int nzl, int ly, int queue) {
    size_t sx = nxl + 2, plane = sx * (nyl + 2), nelem = plane * (nzl + 2);
    size_t nbuf = (size_t)nxl * nzl;
    #pragma acc parallel loop collapse(2) present(u[0:nelem], buf[0:nbuf]) async(queue)
    for (int lz = 1; lz <= nzl; ++lz)
        for (int lx = 1; lx <= nxl; ++lx)
            u[lx + sx * ly + plane * lz] = buf[(lx - 1) + nxl * (lz - 1)];
}
static void pack_z(const double* u, double* buf, int nxl, int nyl, int nzl, int lz, int queue) {
    size_t sx = nxl + 2, plane = sx * (nyl + 2), nelem = plane * (nzl + 2);
    size_t nbuf = (size_t)nxl * nyl;
    #pragma acc parallel loop collapse(2) present(u[0:nelem], buf[0:nbuf]) async(queue)
    for (int ly = 1; ly <= nyl; ++ly)
        for (int lx = 1; lx <= nxl; ++lx)
            buf[(lx - 1) + nxl * (ly - 1)] = u[lx + sx * ly + plane * lz];
}
static void unpack_z(double* u, const double* buf, int nxl, int nyl, int nzl, int lz, int queue) {
    size_t sx = nxl + 2, plane = sx * (nyl + 2), nelem = plane * (nzl + 2);
    size_t nbuf = (size_t)nxl * nyl;
    #pragma acc parallel loop collapse(2) present(u[0:nelem], buf[0:nbuf]) async(queue)
    for (int ly = 1; ly <= nyl; ++ly)
        for (int lx = 1; lx <= nxl; ++lx)
            u[lx + sx * ly + plane * lz] = buf[(lx - 1) + nxl * (ly - 1)];
}

static void pack_all(const double* u, double* sxm, double* sxp, double* sym, double* syp,
                      double* szm, double* szp, int nxl, int nyl, int nzl, int queue) {
    pack_x(u, sxm, nxl, nyl, nzl, 1, queue);
    pack_x(u, sxp, nxl, nyl, nzl, nxl, queue);
    pack_y(u, sym, nxl, nyl, nzl, 1, queue);
    pack_y(u, syp, nxl, nyl, nzl, nyl, queue);
    pack_z(u, szm, nxl, nyl, nzl, 1, queue);
    pack_z(u, szp, nxl, nyl, nzl, nzl, queue);
}
static void unpack_all(double* u, const double* rxm, const double* rxp, const double* rym, const double* ryp,
                        const double* rzm, const double* rzp, int nxl, int nyl, int nzl, int queue) {
    unpack_x(u, rxm, nxl, nyl, nzl, 0, queue);
    unpack_x(u, rxp, nxl, nyl, nzl, nxl + 1, queue);
    unpack_y(u, rym, nxl, nyl, nzl, 0, queue);
    unpack_y(u, ryp, nxl, nyl, nzl, nyl + 1, queue);
    unpack_z(u, rzm, nxl, nyl, nzl, 0, queue);
    unpack_z(u, rzp, nxl, nyl, nzl, nzl + 1, queue);
}

// ---- sum of squares over the owned (non-ghost) box, for a cheap stability check ----
static double sumsq_local(const double* u, int nxl, int nyl, int nzl) {
    size_t sx = nxl + 2, plane = sx * (nyl + 2), nelem = plane * (nzl + 2);
    double sum = 0.0;
    #pragma acc parallel loop collapse(3) present(u[0:nelem]) reduction(+:sum)
    for (int lz = 1; lz <= nzl; ++lz)
        for (int ly = 1; ly <= nyl; ++ly)
            for (int lx = 1; lx <= nxl; ++lx) {
                double val = u[lx + sx * ly + plane * lz];
                sum += val * val;
            }
    return sum;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

    acc_set_device_num(0, acc_device_nvidia); // 1 GH200 GPU/node with mpiprocs=1: device 0 is always ours

    size_t sx = nxl + 2, plane_pad = sx * (size_t)(nyl + 2);
    size_t nelem = plane_pad * (size_t)(nzl + 2);
    size_t nbytes = nelem * sizeof(double);

    double* u0 = (double*)malloc(nbytes);
    double* u1 = (double*)malloc(nbytes);
    #pragma acc enter data create(u0[0:nelem], u1[0:nelem])

    size_t sizeX = (size_t)nyl * nzl, sizeY = (size_t)nxl * nzl, sizeZ = (size_t)nxl * nyl;
    double *sxm_ = (double*)malloc(sizeX * sizeof(double)), *sxp_ = (double*)malloc(sizeX * sizeof(double));
    double *rxm_ = (double*)malloc(sizeX * sizeof(double)), *rxp_ = (double*)malloc(sizeX * sizeof(double));
    double *sym_ = (double*)malloc(sizeY * sizeof(double)), *syp_ = (double*)malloc(sizeY * sizeof(double));
    double *rym_ = (double*)malloc(sizeY * sizeof(double)), *ryp_ = (double*)malloc(sizeY * sizeof(double));
    double *szm_ = (double*)malloc(sizeZ * sizeof(double)), *szp_ = (double*)malloc(sizeZ * sizeof(double));
    double *rzm_ = (double*)malloc(sizeZ * sizeof(double)), *rzp_ = (double*)malloc(sizeZ * sizeof(double));
    #pragma acc enter data create(sxm_[0:sizeX], sxp_[0:sizeX], rxm_[0:sizeX], rxp_[0:sizeX])
    #pragma acc enter data create(sym_[0:sizeY], syp_[0:sizeY], rym_[0:sizeY], ryp_[0:sizeY])
    #pragma acc enter data create(szm_[0:sizeZ], szp_[0:sizeZ], rzm_[0:sizeZ], rzp_[0:sizeZ])

    const double lambda = 1.0 / 7.0; // < 1/6 explicit-scheme stability bound, with margin

    double* h_init = (double*)malloc(nbytes);

  for (int MODE = 0; MODE <= 1; ++MODE) {
    // Initial condition: Gaussian bump at the global domain center, built on the
    // Grace CPU (OpenMP over its cores) into a full host-side padded array
    // (ghosts/global-boundary cells left at 0), then pushed to the GPU with a
    // single `acc update device`. Rebuilt fresh for each mode so naive and
    // overlap start from the identical state.
    memset(h_init, 0, nbytes);
    {
        double cx = NXg * 0.5, cy = NYg * 0.5, cz = NZg * 0.5;
        double sigma2 = 0.02 * (double)NXg * (double)NXg;
        #pragma omp parallel for collapse(3) schedule(static)
        for (int lz = 1; lz <= nzl; ++lz) {
            for (int ly = 1; ly <= nyl; ++ly) {
                for (int lx = 1; lx <= nxl; ++lx) {
                    int gx = gx0 + lx - 1, gy = gy0 + ly - 1, gz = gz0 + lz - 1;
                    double val = 0.0;
                    if (gx > 0 && gx < NXg - 1 && gy > 0 && gy < NYg - 1 && gz > 0 && gz < NZg - 1) {
                        double dx = gx - cx, dy = gy - cy, dz = gz - cz;
                        val = exp(-(dx * dx + dy * dy + dz * dz) / sigma2);
                    }
                    h_init[lx + sx * ly + plane_pad * (size_t)lz] = val;
                }
            }
        }
    }
    memcpy(u0, h_init, nbytes);
    memset(u1, 0, nbytes);
    #pragma acc update device(u0[0:nelem], u1[0:nelem])

    double *u_old = u0, *u_new = u1;
    int report_every = (ITERS >= 5) ? ITERS / 5 : 1;

    MPI_Barrier(cart);
    double t_start = MPI_Wtime();
    double t_comm = 0.0;

    for (int it = 0; it < ITERS; ++it) {
        double tc0 = MPI_Wtime();

        if (MODE == 0) {
            pack_all(u_old, sxm_, sxp_, sym_, syp_, szm_, szp_, nxl, nyl, nzl, acc_async_sync);

            #pragma acc host_data use_device(sxm_, sxp_, sym_, syp_, szm_, szp_, rxm_, rxp_, rym_, ryp_, rzm_, rzp_)
            {
                MPI_Sendrecv(sxm_, sizeX, MPI_DOUBLE, xm, 0, rxp_, sizeX, MPI_DOUBLE, xp, 0, cart, MPI_STATUS_IGNORE);
                MPI_Sendrecv(sxp_, sizeX, MPI_DOUBLE, xp, 1, rxm_, sizeX, MPI_DOUBLE, xm, 1, cart, MPI_STATUS_IGNORE);
                MPI_Sendrecv(sym_, sizeY, MPI_DOUBLE, ym, 2, ryp_, sizeY, MPI_DOUBLE, yp, 2, cart, MPI_STATUS_IGNORE);
                MPI_Sendrecv(syp_, sizeY, MPI_DOUBLE, yp, 3, rym_, sizeY, MPI_DOUBLE, ym, 3, cart, MPI_STATUS_IGNORE);
                MPI_Sendrecv(szm_, sizeZ, MPI_DOUBLE, zm, 4, rzp_, sizeZ, MPI_DOUBLE, zp, 4, cart, MPI_STATUS_IGNORE);
                MPI_Sendrecv(szp_, sizeZ, MPI_DOUBLE, zp, 5, rzm_, sizeZ, MPI_DOUBLE, zm, 5, cart, MPI_STATUS_IGNORE);
            }

            unpack_all(u_old, rxm_, rxp_, rym_, ryp_, rzm_, rzp_, nxl, nyl, nzl, acc_async_sync);
            t_comm += MPI_Wtime() - tc0;

            stencil_update(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg,
                            1, nxl, 1, nyl, 1, nzl, lambda, acc_async_sync);
        } else {
            pack_all(u_old, sxm_, sxp_, sym_, syp_, szm_, szp_, nxl, nyl, nzl, acc_async_sync);

            MPI_Request req[12];
            #pragma acc host_data use_device(sxm_, sxp_, sym_, syp_, szm_, szp_, rxm_, rxp_, rym_, ryp_, rzm_, rzp_)
            {
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
            }

            int ix0 = 2, ix1 = nxl - 1, iy0 = 2, iy1 = nyl - 1, iz0 = 2, iz1 = nzl - 1;
            if (ix1 >= ix0 && iy1 >= iy0 && iz1 >= iz0) {
                stencil_update(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg,
                                ix0, ix1, iy0, iy1, iz0, iz1, lambda, QUEUE_INTERIOR);
            }

            MPI_Waitall(12, req, MPI_STATUSES_IGNORE);
            t_comm += MPI_Wtime() - tc0;

            unpack_all(u_old, rxm_, rxp_, rym_, ryp_, rzm_, rzp_, nxl, nyl, nzl, QUEUE_BOUNDARY);

            // 1-cell-thick shell next to each of the 6 faces (harmlessly
            // recomputes the 12 edges/8 corners more than once: same
            // deterministic formula, so redundant but not incorrect).
            stencil_update(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg, 1, 1, 1, nyl, 1, nzl, lambda, QUEUE_BOUNDARY);
            stencil_update(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg, nxl, nxl, 1, nyl, 1, nzl, lambda, QUEUE_BOUNDARY);
            stencil_update(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg, 1, nxl, 1, 1, 1, nzl, lambda, QUEUE_BOUNDARY);
            stencil_update(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg, 1, nxl, nyl, nyl, 1, nzl, lambda, QUEUE_BOUNDARY);
            stencil_update(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg, 1, nxl, 1, nyl, 1, 1, lambda, QUEUE_BOUNDARY);
            stencil_update(u_old, u_new, nxl, nyl, nzl, gx0, gy0, gz0, NXg, NYg, NZg, 1, nxl, 1, nyl, nzl, nzl, lambda, QUEUE_BOUNDARY);

            #pragma acc wait(QUEUE_INTERIOR, QUEUE_BOUNDARY)
        }

        double* tmp = u_old; u_old = u_new; u_new = tmp;

        if ((it + 1) % report_every == 0 || it == ITERS - 1) {
            double local_sumsq = sumsq_local(u_old, nxl, nyl, nzl);
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
        printf("RESULT impl=acc mode=%s ranks=%d grid=%dx%dx%d domain=%dx%dx%d iters=%d time_s=%.4f comm_s=%.4f gflops=%.2f gbps=%.2f\n",
               MODE == 0 ? "naive" : "overlap", nprocs, dims[0], dims[1], dims[2],
               NXg, NYg, NZg, ITERS, max_time, max_comm, gflops, gbps);
    }
  } // for MODE
    free(h_init);

    #pragma acc exit data delete(sxm_[0:sizeX], sxp_[0:sizeX], rxm_[0:sizeX], rxp_[0:sizeX])
    #pragma acc exit data delete(sym_[0:sizeY], syp_[0:sizeY], rym_[0:sizeY], ryp_[0:sizeY])
    #pragma acc exit data delete(szm_[0:sizeZ], szp_[0:sizeZ], rzm_[0:sizeZ], rzp_[0:sizeZ])
    #pragma acc exit data delete(u0[0:nelem], u1[0:nelem])

    free(u0); free(u1);
    free(sxm_); free(sxp_); free(rxm_); free(rxp_);
    free(sym_); free(syp_); free(rym_); free(ryp_);
    free(szm_); free(szp_); free(rzm_); free(rzp_);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
