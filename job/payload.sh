#!/bin/sh

#------ qsub option --------#
#PBS -q debug-g
#PBS -N stencil3d_gh200
#PBS -l select=4:mpiprocs=1:ompthreads=72
#PBS -l walltime=10:00
#PBS -W group_list=gz00
#PBS -j oe

#PBSWRAP MODULE nvidia/26.3 nv-hpcx

#PBSWRAP SERIAL
# Runs once, on rank 0; the wrapper holds every other rank until this
# block finishes before letting the PARALLEL block below start, so no
# manual $PBSWRAP_RANK branching or marker-file barrier is needed here.
cd "${HOME}"
export TMPDIR="${HOME}/tmp"
mkdir -p "${TMPDIR}"

REPO_URL=https://github.com/cspp-lab/miyabi-gh200-stencil.git
if [ -d repo/.git ]; then
  git -C repo pull --ff-only
else
  git clone --depth 1 "${REPO_URL}" repo
fi
cd repo

echo "=== toolchain: nvidia/26.3 ==="
nvcc --version | tail -1
make clean && make

#PBSWRAP PARALLEL
# One sandboxed process per rank; this binary runs naive then overlap
# internally in a single MPI session (see src/stencil3d.cu). A PARALLEL
# region supports exactly one mpirun-launched program, so the CUDA and
# OpenACC binaries each need their own SERIAL/PARALLEL pair below rather
# than two invocations back to back in one PARALLEL block (a second
# MPI_Init in the same region fails with "getting local rank failed").
cd "${HOME}/repo"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}
echo "=== impl: CUDA ==="
./stencil3d 768 768 768 500

#PBSWRAP SERIAL
cd "${HOME}/repo"

#PBSWRAP PARALLEL
# Same toolchain, domain, and iteration count as the CUDA run above, so
# the RESULT lines (impl=cuda vs impl=acc) are directly comparable.
cd "${HOME}/repo"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}
echo "=== impl: OpenACC ==="
./stencil3d_acc 768 768 768 500
