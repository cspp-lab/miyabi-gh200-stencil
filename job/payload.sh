#!/bin/sh

#------ qsub option --------#
#PBS -q debug-g
#PBS -N stencil3d_gh200
#PBS -l select=4:mpiprocs=1:ompthreads=72
#PBS -l walltime=10:00
#PBS -W group_list=gz00
#PBS -j oe

#------- Program execution -------#
# $HOME on this cluster resolves to a node-local /tmp/work path, not the
# shared filesystem, so a binary built there is invisible to every node but
# the one that built it. Use the shared group work area instead.
WORKDIR="/work/gz00/z30105/miyabi-gh200-stencil-run"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# --- sync code from github.com/cspp-lab (Miyabi side only ever runs qsub; the
#     git pull happens here, inside the job payload, at job start) ---
REPO_URL=https://github.com/cspp-lab/miyabi-gh200-stencil.git
if [ -d repo/.git ]; then
  git -C repo pull --ff-only
else
  git clone --depth 1 "${REPO_URL}" repo
fi
cd repo

# --- toolchain ---
module purge
module load nvidia/26.3 nv-hpcx
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}

# The per-job /tmp dir this cluster provides isn't reliably writable here;
# give nvcc a scratch dir we know exists instead.
export TMPDIR="${WORKDIR}/tmp"
mkdir -p "${TMPDIR}"

make clean && make

NX=768
NY=768
NZ=768
ITERS=500

RUNDIR="$(pwd)"
MPIRUN_OPTS="-np 4 --hostfile ${PBS_NODEFILE} --map-by ppr:1:node --wdir ${RUNDIR}"

echo "=== naive (blocking halo exchange) ==="
mpirun ${MPIRUN_OPTS} ./stencil3d ${NX} ${NY} ${NZ} ${ITERS} 0

echo "=== overlap (comm/compute overlap) ==="
mpirun ${MPIRUN_OPTS} ./stencil3d ${NX} ${NY} ${NZ} ${ITERS} 1
