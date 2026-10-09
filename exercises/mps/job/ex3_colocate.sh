#!/bin/sh
# Exercise 3: how many SMs does a memory-bound code actually need, and can
# the rest of the GPU do something useful at the same time?
#   1. the 3D stencil (memory bound) in partitions of 1..15 chunks
#   2. a compute-bound FMA kernel likewise
#   3. stencil and FMA one after the other on the whole partitionable GPU,
#      versus side by side in K_STENCIL + (15 - K_STENCIL) chunks
# Pick K_STENCIL from your step-1 curve and resubmit with -v K_STENCIL=<k>.

#------ qsub option --------#
#PBS -q debug-g
#PBS -N mps_ex3
#PBS -l select=1:mpiprocs=1:ompthreads=72
#PBS -l walltime=10:00
#PBS -W group_list=gz00
#PBS -j oe

#PBSWRAP MODULE nvidia/26.3 nv-hpcx

#PBSWRAP SERIAL
cd "${HOME}"
export TMPDIR="${HOME}/tmp"; mkdir -p "${TMPDIR}"
git clone --depth 1 https://github.com/cspp-lab/miyabi-gh200-stencil.git repo
make -C repo stencil3d
cd repo/exercises/mps && make
. job/mps_env.sh

K_STENCIL=${K_STENCIL:-4}
STENCIL="../../stencil3d 512 512 512 200"   # single rank: 1x1x1 grid
FMA="./work fma 100"

now() { date +%s.%N; }

echo "== full GPU, no MPS"
$STENCIL | grep RESULT
$FMA full

mps_start -S

echo "== step 1/2: chunk scan"
for n in 1 2 4 6 8 12 15; do
    P=$(mps_addpart "$n")
    echo "-- chunks=$n"
    CUDA_MPS_SM_PARTITION="$P" $STENCIL | grep RESULT
    CUDA_MPS_SM_PARTITION="$P" $FMA "chunks=$n"
    mps_rmpart "$P" > /dev/null
done

echo "== step 3a: one after the other, 15 chunks"
P=$(mps_addpart 15)
t0=$(now)
CUDA_MPS_SM_PARTITION="$P" $STENCIL | grep RESULT
CUDA_MPS_SM_PARTITION="$P" $FMA seq
t1=$(now)
echo "SEQUENTIAL makespan_s=$(echo "$t1 - $t0" | bc)"
mps_rmpart "$P" > /dev/null

echo "== step 3b: side by side, stencil ${K_STENCIL} + fma $((15 - K_STENCIL)) chunks"
PS=$(mps_addpart "$K_STENCIL")
PF=$(mps_addpart $((15 - K_STENCIL)))
t0=$(now)
CUDA_MPS_SM_PARTITION="$PS" $STENCIL | grep RESULT &
CUDA_MPS_SM_PARTITION="$PF" $FMA side
wait
t1=$(now)
echo "COLOCATED k=${K_STENCIL} makespan_s=$(echo "$t1 - $t0" | bc)"

mps_stop
