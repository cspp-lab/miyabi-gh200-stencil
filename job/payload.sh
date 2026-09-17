#!/bin/sh

#------ qsub option --------#
#PBS -q debug-g
#PBS -N stencil3d_gh200
#PBS -l select=4:mpiprocs=1:ompthreads=72
#PBS -l walltime=10:00
#PBS -W group_list=gz00
#PBS -j oe

#------- Program execution -------#
cd ${PBS_O_WORKDIR}

# --- sync code from github.com/cspp-lab (Miyabi side only ever runs qsub; the
#     git pull happens here, inside the job payload, at job start) ---
REPO_URL=https://github.com/cspp-lab/miyabi-gh200-stencil.git
if [ -d repo/.git ]; then
  git -C repo pull --ff-only
else
  git clone --depth 1 "${REPO_URL}" repo
fi
cd repo

# --- toolchain: adjust to whatever `module avail` shows on this system ---
module load nvidia nvmpi 2>/dev/null || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}

make clean && make

NX=768
NY=768
NZ=768
ITERS=500

echo "=== naive (blocking halo exchange) ==="
mpirun -np 4 ./stencil3d ${NX} ${NY} ${NZ} ${ITERS} 0

echo "=== overlap (comm/compute overlap) ==="
mpirun -np 4 ./stencil3d ${NX} ${NY} ${NZ} ${ITERS} 1
