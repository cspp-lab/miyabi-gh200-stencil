# pbswrap job body: can the template's launcher start MVAPICH-Plus 5.1.0rc ranks?
# Submit: queue=regular-g (or debug-g) select=2:mpiprocs=1 walltime=00:10:00
# Based on ~/Downloads/osu_bw.sh. The MVP install dir contains the launcher's
# name, which pbswrap rejects on a string match, so it is resolved by a glob.
#PBSWRAP MODULE gcc cuda/13.4 nccl
#PBSWRAP SINGLE
MVP_PATH=$(echo /work/share/MVAPICH-Plus/5.1.0rc/opt/mvapich/plus/5.1.0rc/cuda13.3/ucx/*/gcc11.3.1)
OMB_ROOT=/work/share/benchmarks/Miyabi-G/osu-microbenchmarks-7.5.3-MVAPICH-Plus-5.1.0rc
mkdir -p summary
{
  echo "MVP_PATH=$MVP_PATH"
  ls -d /work/share/MVAPICH-Plus/5.1.0rc/opt/mvapich/plus/5.1.0rc/*/*/*
  module list 2>&1
  echo "== mpichversion"; $MVP_PATH/bin/mpichversion
  echo "== run scripts"; ls $OMB_ROOT/run
  echo "== omb"; ls $OMB_ROOT/libexec/osu-micro-benchmarks/mpi/*
  echo "== launcher env"; env | grep -E '^(PMI|PMIX|OMPI|HYDRA|PBSWRAP)' | sort
} > summary/env.txt 2>&1
#PBSWRAP PARALLEL
export T_UCX_DIR=/work/share/ucx/1.19.0-gcc11
export MVP_PATH=$(echo /work/share/MVAPICH-Plus/5.1.0rc/opt/mvapich/plus/5.1.0rc/cuda13.3/ucx/*/gcc11.3.1)
export PATH=$MVP_PATH/bin:$T_UCX_DIR/bin:$PATH
export LD_LIBRARY_PATH=$T_UCX_DIR/lib:$MVP_PATH/lib:$LD_LIBRARY_PATH
export MVP_ENABLE_GPU=1
PROG=/work/share/benchmarks/Miyabi-G/osu-microbenchmarks-7.5.3-MVAPICH-Plus-5.1.0rc/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw
mkdir -p summary
{
  echo "rank=$PBSWRAP_RANK host=$(hostname)"
  env | grep -E '^(PMI|PMIX|OMPI)' | sort
  timeout 120 $PROG -d cuda D D
  echo "exit=$?"
} > summary/rank-$PBSWRAP_RANK.txt 2>&1
