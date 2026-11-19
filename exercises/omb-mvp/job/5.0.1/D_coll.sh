# D: collectives, select=N:mpiprocs=1 (N=2,4,8)
#PBSWRAP SINGLE
mkdir -p summary
cat > omb.env <<'ENV'
O=/work/share/benchmarks/Miyabi-G/osu-microbenchmarks-7.5.3-MVAPICH-Plus-5.0.1/libexec/osu-micro-benchmarks/mpi
# run <name> <subdir/bench> [args...]: rank 0 output -> summary/<name>.txt, other ranks' errors kept only if non-empty
run() {
  local name=$1; shift
  if [ "$PBSWRAP_RANK" = 0 ]; then
    timeout 600 $O/"$@" > summary/$name.txt 2>&1; echo "rc=$?" >> summary/$name.txt
  else
    local e=summary/$name.r$PBSWRAP_RANK.err
    timeout 600 $O/"$@" > $e 2>&1; [ -s $e ] || rm -f $e
  fi
}
ENV
{ module list 2>&1; env | grep -E '^(MVP|UCX|HYDRA)_' | sort; mpichversion | head -1
  ldd $(. ./omb.env; echo $O/pt2pt/osu_bw) | grep -E 'mpi|uc[pst]'; nvidia-smi -L; true; } > summary/env.txt 2>&1
#PBSWRAP PARALLEL
echo "rank=$PBSWRAP_RANK host=$(hostname) cpus=$(taskset -pc $$ | sed 's/.*: //')" > summary/place-$PBSWRAP_RANK.txt
#PBSWRAP PARALLEL
. ./omb.env; run allreduce_D collective/osu_allreduce -d cuda -m 4:16777216
#PBSWRAP PARALLEL
. ./omb.env; run allreduce_H collective/osu_allreduce -m 4:16777216
#PBSWRAP PARALLEL
. ./omb.env; run reduce_scatter_D collective/osu_reduce_scatter -d cuda -m 4:16777216
#PBSWRAP PARALLEL
. ./omb.env; run allgather_D collective/osu_allgather -d cuda -m 1:4194304
#PBSWRAP PARALLEL
. ./omb.env; run alltoall_D collective/osu_alltoall -d cuda -m 1:4194304
#PBSWRAP PARALLEL
. ./omb.env; run alltoall_H collective/osu_alltoall -m 1:4194304
#PBSWRAP PARALLEL
. ./omb.env; run bcast_D collective/osu_bcast -d cuda -m 1:16777216
#PBSWRAP PARALLEL
. ./omb.env; run barrier collective/osu_barrier
