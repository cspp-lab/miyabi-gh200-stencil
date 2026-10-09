#!/bin/sh
# Exercise 0: start an MPS daemon with static SM partitioning, give each of
# two ranks on one node its own 7-chunk (56-SM) partition, and stop it again.

#------ qsub option --------#
#PBS -q debug-g
#PBS -N mps_ex0
#PBS -l select=1:mpiprocs=2:ompthreads=36
#PBS -l walltime=05:00
#PBS -W group_list=gz00
#PBS -j oe

#PBSWRAP MODULE nvidia/26.3 nv-hpcx

#PBSWRAP SERIAL
cd "${HOME}"
export TMPDIR="${HOME}/tmp"; mkdir -p "${TMPDIR}"
git clone --depth 1 https://github.com/cspp-lab/miyabi-gh200-stencil.git repo
cd repo/exercises/mps && make
. job/mps_env.sh

mps_start -S
mps_addpart 7 >  "${HOME}/parts.txt"
mps_addpart 7 >> "${HOME}/parts.txt"
mps_lspart

echo "== a client that does not choose a partition:"
./smid no-partition

#PBSWRAP PARALLEL
# Rank r takes line r+1 of parts.txt
cd "${HOME}/repo/exercises/mps"
. job/mps_env.sh
export CUDA_MPS_SM_PARTITION=$(sed -n "$((PBSWRAP_RANK + 1))p" "${HOME}/parts.txt")
./smid "rank${PBSWRAP_RANK}"

#PBSWRAP SERIAL
cd "${HOME}/repo/exercises/mps"
. job/mps_env.sh
mps_stop && echo "MPS stopped"
