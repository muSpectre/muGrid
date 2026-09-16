# Shared environment for the muGrid homogenization benchmark scripts.
#
# Sourced by the three job submission scripts (job_*.sh) and by make_docs.sh.
# Sets up the compiled extension on PYTHONPATH, the UCX transport workaround,
# and the GPU-count override the benchmark drivers use on this AMD node, then
# cds into the repository. Does not set `set -e` — the caller owns that.

# --------------------------------------------------------------------------- #
# Cluster module / toolchain environment (compilers, MPI, Python, matplotlib).
# --------------------------------------------------------------------------- #
source /work/classic/fr_lp1029-IMTEK-Simulation/mi300a/env.sh

REPO="$HOME/Software/muGrid"

# Make the compiled extension and the pure-Python bindings importable.
export PYTHONPATH="$REPO/build_mi300a/language_bindings/python:$REPO/language_bindings/python${PYTHONPATH:+:$PYTHONPATH}"

# UCX_TLS=^rocm_ipc: the ROCm IPC rendezvous transport triggers an rkey-size
# assertion failure (rkey_size=9 exp=79) between ranks; disable it so UCX falls
# back to rocm_copy / shared-memory transfers. Needed for the multi-GPU MPI runs.
export UCX_TLS="^rocm_ipc"

# The benchmark drivers count GPUs to plan the GPU curves. This is an AMD node
# but still ships a stub `nvidia-smi`, so pin the count to the SLURM allocation
# rather than relying on vendor auto-detection. When run outside a GPU
# allocation (e.g. rendering on the head node) this stays unset and the GPU
# sweeps are skipped automatically.
GPUS_ALLOC="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-}}"
if [[ -n "$GPUS_ALLOC" ]]; then
    export MUGRID_BENCH_GPU_COUNT="$GPUS_ALLOC"
    export MUGRID_BENCH_GPU_NAME="AMD Instinct MI300A"
fi

cd "$REPO"
