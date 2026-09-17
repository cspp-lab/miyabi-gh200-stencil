#!/bin/sh

#------ qsub option --------#
#PBS -q debug-g
#PBS -N stencil3d_gh200
#PBS -l select=4:mpiprocs=1:ompthreads=72
#PBS -l walltime=10:00
#PBS -W group_list=gz00
#PBS -j oe

#PBSWRAP SERIAL
# Runs once, on rank 0; the wrapper holds every other rank until this
# block finishes before letting the PARALLEL block below start, so no
# manual $PBSWRAP_RANK branching or marker-file barrier is needed here.
cd "${HOME}"

module purge
module load nvidia/26.3 nv-hpcx
export TMPDIR="${HOME}/tmp"
mkdir -p "${TMPDIR}"

REPO_URL=https://github.com/cspp-lab/miyabi-gh200-stencil.git
if [ -d repo/.git ]; then
  git -C repo pull --ff-only
else
  git clone --depth 1 "${REPO_URL}" repo
fi
cd repo

make clean && make

#PBSWRAP PARALLEL
# One sandboxed process per rank; runs naive then overlap internally in a
# single MPI session (see src/stencil3d.cu).
cd "${HOME}/repo"

module purge
module load nvidia/26.3 nv-hpcx
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}

NX=768
NY=768
NZ=768
ITERS=500

./stencil3d ${NX} ${NY} ${NZ} ${ITERS}
