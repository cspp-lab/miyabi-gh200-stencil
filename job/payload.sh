#!/bin/sh

#------ qsub option --------#
#PBS -q debug-g
#PBS -N stencil3d_gh200
#PBS -l select=4:mpiprocs=1:ompthreads=72
#PBS -l walltime=10:00
#PBS -W group_list=gz00
#PBS -j oe

#------- Program execution -------#
# This whole script runs once per rank (sandboxed), so the one-time build
# (git sync + compile) is guarded to rank 0; every other rank just waits on
# a marker file on the shared Lustre /work area before running the binary
# directly (no launcher call needed here - the platform starts one
# sandboxed rank per process already).
WORKDIR=/work/gz00/z30105/miyabi-gh200-stencil-run
DONE_MARKER="${WORKDIR}/.build_done"

module purge
module load nvidia/26.3 nv-hpcx
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}
export TMPDIR="${WORKDIR}/tmp"

if [ "${PBSWRAP_RANK:-0}" = "0" ]; then
  mkdir -p "${WORKDIR}" "${TMPDIR}"
  rm -f "${DONE_MARKER}"
  cd "${WORKDIR}"

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
  cd "${WORKDIR}/repo"
fi

NX=768
NY=768
NZ=768
ITERS=500

if [ "${PBSWRAP_RANK:-0}" = "0" ]; then echo "=== naive (blocking halo exchange) ==="; fi
./stencil3d ${NX} ${NY} ${NZ} ${ITERS} 0

if [ "${PBSWRAP_RANK:-0}" = "0" ]; then echo "=== overlap (comm/compute overlap) ==="; fi
./stencil3d ${NX} ${NY} ${NZ} ${ITERS} 1
