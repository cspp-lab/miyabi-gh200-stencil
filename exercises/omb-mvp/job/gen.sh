#!/bin/bash
# Generate pbswrap job bodies for OMB on MVAPICH-Plus 5.1.0rc (Miyabi-G).
# One PARALLEL section per benchmark (one launch per section).
# Usage: ./gen.sh [MVP version, default 5.1.0rc] -> writes <ver>/{A_pt2pt_inter,B_pt2pt_intra,C_multipair,D_coll}.sh
# The MVAPICH install itself is chosen by the pbswrap template; this only picks the matching OMB build.
set -e
cd "$(dirname "$0")"
VER=${1:-5.1.0rc}
mkdir -p $VER

prelude() {
sed "s/@VER@/$VER/" <<'EOF'
#PBSWRAP SINGLE
mkdir -p summary
cat > omb.env <<'ENV'
O=/work/share/benchmarks/Miyabi-G/osu-microbenchmarks-7.5.3-MVAPICH-Plus-@VER@/libexec/osu-micro-benchmarks/mpi
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
EOF
}

sec() { printf '#PBSWRAP PARALLEL\n. ./omb.env; %s\n' "$*"; }

{
  echo "# A: inter-node pt2pt, select=2:mpiprocs=1"
  prelude
  sec run lat_HH pt2pt/osu_latency
  sec run lat_DD pt2pt/osu_latency -d cuda D D
  sec run bw_HH pt2pt/osu_bw
  sec run bw_HD pt2pt/osu_bw -d cuda H D
  sec run bw_DH pt2pt/osu_bw -d cuda D H
  sec run bw_DD pt2pt/osu_bw -d cuda D D
  sec run bibw_HH pt2pt/osu_bibw
  sec run bibw_DD pt2pt/osu_bibw -d cuda D D
  sec 'export UCX_RNDV_THRESH=intra:auto,inter:8192; run lat_DD_rndv8k pt2pt/osu_latency -d cuda D D'
  sec 'export UCX_RNDV_THRESH=intra:auto,inter:8192; run bw_DD_rndv8k pt2pt/osu_bw -d cuda D D'
  sec run putbw_DD one-sided/osu_put_bw -d cuda D D
  sec run putlat_DD one-sided/osu_put_latency -d cuda D D
} > $VER/A_pt2pt_inter.sh

{
  echo "# B: intra-node pt2pt (host buffers), select=1:mpiprocs=2"
  prelude
  sec run lat_HH_intra pt2pt/osu_latency
  sec run bw_HH_intra pt2pt/osu_bw
  sec run bibw_HH_intra pt2pt/osu_bibw
} > $VER/B_pt2pt_intra.sh

{
  echo "# C: multi-pair / congestion (host buffers), select=2:mpiprocs=8"
  prelude
  sec run mbw_mr_HH pt2pt/osu_mbw_mr
  sec run multi_lat_HH pt2pt/osu_multi_lat
  sec run fan_in_HH congestion/osu_bw_fan_in
  sec run fan_out_HH congestion/osu_bw_fan_out
} > $VER/C_multipair.sh

{
  echo "# D: collectives, select=N:mpiprocs=1 (N=2,4,8)"
  prelude
  sec run allreduce_D collective/osu_allreduce -d cuda -m 4:16777216
  sec run allreduce_H collective/osu_allreduce -m 4:16777216
  sec run reduce_scatter_D collective/osu_reduce_scatter -d cuda -m 4:16777216
  sec run allgather_D collective/osu_allgather -d cuda -m 1:4194304
  sec run alltoall_D collective/osu_alltoall -d cuda -m 1:4194304
  sec run alltoall_H collective/osu_alltoall -m 1:4194304
  sec run bcast_D collective/osu_bcast -d cuda -m 1:16777216
  sec run barrier collective/osu_barrier
} > $VER/D_coll.sh
