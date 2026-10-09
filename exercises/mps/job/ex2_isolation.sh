#!/bin/sh
# Exercise 2: a latency-sensitive "victim" next to a "noisy neighbour" that
# keeps the GPU full, under five ways of sharing one GPU:
#   A  victim alone (reference)
#   B  no MPS: the two processes time-slice the GPU
#   C  MPS, no limits: SMs are shared dynamically
#   D  MPS with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 for both: a cap, not a reservation
#   E  MPS static partitioning, 7 chunks each: exclusive SMs
# Compare the victim's p50/p99/max latency and the noisy kernel count.

#------ qsub option --------#
#PBS -q debug-g
#PBS -N mps_ex2
#PBS -l select=1:mpiprocs=1:ompthreads=72
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

VICTIM_ITERS=2000
NOISE_SEC=8

# pair <noisy-env> <victim-env> <tag>: noisy in the background, victim once
# the noisy one has had time to fill the GPU
pair() {
    env $1 ./work noisy "$NOISE_SEC" &
    sleep 2
    env $2 ./work victim "$VICTIM_ITERS" "$3"
    wait
}

echo "== A: victim alone"
./work victim "$VICTIM_ITERS" alone

echo "== B: no MPS"
pair "" "" no-mps

echo "== C: MPS, dynamic sharing"
mps_start
pair "" "" mps-dynamic
mps_stop

echo "== D: MPS, ACTIVE_THREAD_PERCENTAGE=50"
mps_start
pair CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 mps-pct50
mps_stop

echo "== E: MPS static partitioning, 7 + 7 chunks"
mps_start -S
PN=$(mps_addpart 7)
PV=$(mps_addpart 7)
pair CUDA_MPS_SM_PARTITION="$PN" CUDA_MPS_SM_PARTITION="$PV" mps-static

echo "== check: ACTIVE_THREAD_PERCENTAGE inside a static partition"
CUDA_MPS_SM_PARTITION="$PV" CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 ./smid static+pct50
mps_stop
