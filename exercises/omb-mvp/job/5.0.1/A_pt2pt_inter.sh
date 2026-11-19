# A: inter-node pt2pt, select=2:mpiprocs=1
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
. ./omb.env; run lat_HH pt2pt/osu_latency
#PBSWRAP PARALLEL
. ./omb.env; run lat_DD pt2pt/osu_latency -d cuda D D
#PBSWRAP PARALLEL
. ./omb.env; run bw_HH pt2pt/osu_bw
#PBSWRAP PARALLEL
. ./omb.env; run bw_HD pt2pt/osu_bw -d cuda H D
#PBSWRAP PARALLEL
. ./omb.env; run bw_DH pt2pt/osu_bw -d cuda D H
#PBSWRAP PARALLEL
. ./omb.env; run bw_DD pt2pt/osu_bw -d cuda D D
#PBSWRAP PARALLEL
. ./omb.env; run bibw_HH pt2pt/osu_bibw
#PBSWRAP PARALLEL
. ./omb.env; run bibw_DD pt2pt/osu_bibw -d cuda D D
#PBSWRAP PARALLEL
. ./omb.env; export UCX_RNDV_THRESH=intra:auto,inter:8192; run lat_DD_rndv8k pt2pt/osu_latency -d cuda D D
#PBSWRAP PARALLEL
. ./omb.env; export UCX_RNDV_THRESH=intra:auto,inter:8192; run bw_DD_rndv8k pt2pt/osu_bw -d cuda D D
#PBSWRAP PARALLEL
. ./omb.env; run putbw_DD one-sided/osu_put_bw -d cuda D D
#PBSWRAP PARALLEL
. ./omb.env; run putlat_DD one-sided/osu_put_latency -d cuda D D
