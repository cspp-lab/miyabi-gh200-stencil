# pbswrap job body: do MVAPICH-Plus binaries wire up under the template's
# default (HPC-X Open MPI) launcher, or does each rank come up as a singleton?
# Submit: queue=regular-g or debug-g select=2:mpiprocs=1 walltime=00:10:00
#PBSWRAP PARALLEL
export T_UCX_DIR=/work/share/ucx/1.19.0-gcc11
export MVP_PATH=$(echo /work/share/MVAPICH-Plus/5.1.0rc/opt/mvapich/plus/5.1.0rc/cuda13.3/ucx/*/gcc11.3.1)
export LD_LIBRARY_PATH=$T_UCX_DIR/lib:$MVP_PATH/lib:$LD_LIBRARY_PATH
export MVP_ENABLE_GPU=1
OMB=/work/share/benchmarks/Miyabi-G/osu-microbenchmarks-7.5.3-MVAPICH-Plus-5.1.0rc/libexec/osu-micro-benchmarks/mpi
mkdir -p summary
{
  echo "rank=$PBSWRAP_RANK host=$(hostname)"
  env | grep -E '^(PMI|PMIX|OMPI_COMM_WORLD|HYDRA)' | sort
  echo "== ldd"; ldd $OMB/startup/osu_hello | grep -E 'mpi|ucx|ucp|pmi'
  echo "== osu_hello"; timeout 60 $OMB/startup/osu_hello; echo "exit=$?"
  echo "== osu_bw H H"; timeout 60 $OMB/pt2pt/osu_bw -m 1:4194304; echo "exit=$?"
} > summary/rank-$PBSWRAP_RANK.txt 2>&1
