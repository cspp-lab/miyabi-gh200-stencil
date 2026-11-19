# pbswrap v33 (mvapich template): environment + 2-rank wire-up check.
# Submit: queue=debug-g or regular-g select=2:mpiprocs=1 walltime=00:10:00
#PBSWRAP SINGLE
mkdir -p summary
{
  module list 2>&1
  echo "PATH=$PATH" | tr ':' '\n'
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | tr ':' '\n'
  env | grep -E '^(MVP|UCX|HYDRA|PMI)' | sort
  echo "== mpichversion"; mpichversion 2>&1 | head -3
  true
} > summary/single.txt 2>&1
#PBSWRAP PARALLEL
OMB=/work/share/benchmarks/Miyabi-G/osu-microbenchmarks-7.5.3-MVAPICH-Plus-5.1.0rc/libexec/osu-micro-benchmarks/mpi
mkdir -p summary
{
  echo "rank=$PBSWRAP_RANK host=$(hostname)"
  env | grep -E '^(PMI|MVP|UCX)' | sort
  echo "== ldd"; ldd $OMB/pt2pt/osu_bw | grep -E 'mpi|uc[pst]'
  echo "== osu_hello"; timeout 60 $OMB/startup/osu_hello; echo "exit=$?"
  echo "== osu_bw D D"; timeout 120 $OMB/pt2pt/osu_bw -d cuda D D; echo "exit=$?"
} > summary/rank-$PBSWRAP_RANK.txt 2>&1
true
