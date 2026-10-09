# mps_env.sh — helpers for running an MPS daemon inside one Miyabi-G job.
# Source it at the top of every SERIAL/PARALLEL region:  . ./mps_env.sh
#
# The pipe/log directories live under /tmp and are named after the job's
# private $HOME (/tmp/pbswrap-<jobid>/out), so every region and every rank
# of the same job on the same node finds the same daemon, and two jobs never
# collide.

_mps_tag=$(basename "$(dirname "$HOME")")
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe-${_mps_tag}
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log-${_mps_tag}

gpu_uuid() {
    nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1
}

# mps_start [-S]   start the control daemon (-S: static SM partitioning)
mps_start() {
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d "$@" || return 1
    for _ in $(seq 1 50); do
        [ -p "$CUDA_MPS_PIPE_DIRECTORY/control" ] && return 0
        sleep 0.1
    done
    echo "mps_start: control pipe did not appear" >&2
    return 1
}

# mps_stop   quit the daemon and wait until it is really gone, so that a
# following mps_start (possibly with different options) starts clean
mps_stop() {
    echo quit | nvidia-cuda-mps-control
    for _ in $(seq 1 100); do
        pgrep -u "$(id -u)" -x nvidia-cuda-mps-control > /dev/null || return 0
        sleep 0.1
    done
    echo "mps_stop: daemon still running" >&2
    return 1
}

# mps_addpart N   create an N-chunk partition (8 SMs/chunk on GH200) and
# print its full ID, i.e. the value for CUDA_MPS_SM_PARTITION.
# The ID is taken from the "created" line: lspart abbreviates the UUID.
mps_addpart() {
    _uuid=$(gpu_uuid)
    echo "sm_partition add $_uuid $1" | nvidia-cuda-mps-control |
        grep -o "$_uuid/[A-Za-z0-9+/=]*"
}

# mps_rmpart ID   remove a partition given its full ID
mps_rmpart() {
    echo "sm_partition rm ${1%%/*} ${1#*/}" | nvidia-cuda-mps-control
}

mps_lspart() {
    echo lspart | nvidia-cuda-mps-control
}
