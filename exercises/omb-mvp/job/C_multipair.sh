# C: multi-pair / congestion (host buffers), select=2:mpiprocs=8
#PBSWRAP SINGLE
mkdir -p summary
cat > omb.env <<'ENV'
O=/work/share/benchmarks/Miyabi-G/osu-microbenchmarks-7.5.3-MVAPICH-Plus-5.1.0rc/libexec/osu-micro-benchmarks/mpi
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
. ./omb.env; run mbw_mr_HH pt2pt/osu_mbw_mr
#PBSWRAP PARALLEL
. ./omb.env; run multi_lat_HH pt2pt/osu_multi_lat
#PBSWRAP PARALLEL
. ./omb.env; run fan_in_HH congestion/osu_bw_fan_in
#PBSWRAP PARALLEL
. ./omb.env; run fan_out_HH congestion/osu_bw_fan_out
