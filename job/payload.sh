#!/bin/sh

#------ qsub option --------#
#PBS -q debug-g
#PBS -N stencil3d_gh200
#PBS -l select=4:mpiprocs=1:ompthreads=72
#PBS -l walltime=10:00
#PBS -W group_list=gz00
#PBS -j oe

#------- Build (plain shell; runs once, not through mpirun) -------#
# /work is Lustre and is genuinely shared/identical across every node of the
# job (confirmed via diagnostic job); $HOME is not, so the build output must
# live here for the "./stencil3d ..." launch below to see it on every node.
WORKDIR=/work/gz00/z30105/miyabi-gh200-stencil-run
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

module purge
module load nvidia/26.3 nv-hpcx
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}

# The per-job /tmp dir this cluster provides isn't reliably writable here;
# give nvcc a scratch dir we know exists instead.
export TMPDIR="${WORKDIR}/tmp"
mkdir -p "${TMPDIR}"

make clean && make

#------- Program execution -------#
# Writing "./stencil3d ..." here runs as `mpirun bwrap ... ./stencil3d ...`
# across all allocated nodes; no explicit --hostfile/--map-by/--wdir needed.
NX=768
NY=768
NZ=768
ITERS=500

echo "=== naive (blocking halo exchange) ==="
./stencil3d ${NX} ${NY} ${NZ} ${ITERS} 0

echo "=== overlap (comm/compute overlap) ==="
./stencil3d ${NX} ${NY} ${NZ} ${ITERS} 1
