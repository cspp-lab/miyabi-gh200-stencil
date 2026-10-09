#!/bin/sh
# Exercise 1: how does the visible SM count and the run time change with the
# partition size? Creates one partition at a time, runs ./smid in it, and
# removes it again.

#------ qsub option --------#
#PBS -q debug-g
#PBS -N mps_ex1
#PBS -l select=1:mpiprocs=1:ompthreads=72
#PBS -l walltime=05:00
#PBS -W group_list=gz00
#PBS -j oe

#PBSWRAP MODULE nvidia/26.3 nv-hpcx

#PBSWRAP SERIAL
cd "${HOME}"
export TMPDIR="${HOME}/tmp"; mkdir -p "${TMPDIR}"
git clone --depth 1 -b "${REPO_BRANCH:-mps-test}" https://github.com/cspp-lab/miyabi-gh200-stencil.git repo
cd repo/exercises/mps && make
. job/mps_env.sh

./smid full-gpu-no-mps

mps_start -S
for n in 1 2 3 4 6 8 12 15; do
    P=$(mps_addpart "$n")
    CUDA_MPS_SM_PARTITION="$P" ./smid "chunks=$n"
    mps_rmpart "$P" > /dev/null
done

echo "== can we have all 16.5 chunks' worth of SMs?"
P=$(mps_addpart 16) && echo "16 chunks: $P" || echo "16 chunks: refused"
mps_lspart
mps_stop
