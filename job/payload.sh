#!/bin/sh

#------ qsub option --------#
#PBS -q debug-g
#PBS -N stencil3d_gh200
#PBS -l select=4:mpiprocs=1:ompthreads=72
#PBS -l walltime=10:00
#PBS -W group_list=gz00
#PBS -j oe

#------- Program execution -------#
# This body runs once per rank, each inside its own bwrap sandbox where
# /work is a per-rank empty tmpfs. $HOME (and $PBSWRAP_WORK under it) is
# bind-mounted from the SAME host directory for every rank of this job
# (/work/gz00/z30105/demo/runs/<jobid>), so it's the one place all ranks
# can actually share files - use it, not /work directly.
cd "${HOME}"
DONE_MARKER="${HOME}/.build_done"

module purge
module load nvidia/26.3 nv-hpcx
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}
export TMPDIR="${HOME}/tmp"

if [ "${PBSWRAP_RANK:-0}" = "0" ]; then
  mkdir -p "${TMPDIR}"
  rm -f "${DONE_MARKER}"

  REPO_URL=https://github.com/cspp-lab/miyabi-gh200-stencil.git
  if [ -d repo/.git ]; then
    git -C repo pull --ff-only
  else
    git clone --depth 1 "${REPO_URL}" repo
  fi
  cd repo

  make clean && make
  touch "${DONE_MARKER}"
else
  until [ -f "${DONE_MARKER}" ]; do sleep 2; done
  cd "${HOME}/repo"
fi

NX=768
NY=768
NZ=768
ITERS=500

# Runs naive then overlap internally, in one MPI session (see src/stencil3d.cu).
./stencil3d ${NX} ${NY} ${NZ} ${ITERS}
