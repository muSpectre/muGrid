#!/bin/bash
#SBATCH --job-name=muGrid-homog-bench
#SBATCH --partition=mi300a
#SBATCH --nodes=1
#SBATCH --ntasks=92
#SBATCH --cpus-per-task=1
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --account=bw17d009
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Benchmark of the muGrid FEM-elasticity homogenization example, run on an
# MI300A compute node (4 APUs). Two studies are appended to the shared benchmark
# database (benchmarks/results.csv) and the documentation pages are regenerated:
#
#   1. examples/benchmark_homogenization.py
#      WITHOUT preconditioner: time vs. grid size (cpu1 / cpuN / gpu1 / gpuN) and
#      MPI strong scaling on CPU cores and on the GPUs.
#
#   2. examples/benchmark_homogenization_preconditioner.py
#      WITH the reference-material (Ladecky et al. 2023) Fourier preconditioner:
#      iteration count vs. grid size (none vs. reference), reference-solve time
#      across device/MPI configs, and MPI strong scaling on CPU cores and GPUs.
#
# MPI strong scaling is measured on the CPU (1,2,4,8,16,32,64,92 ranks) and on
# the GPUs (1,2,4 devices — the node has 4 MI300A APUs; homogenization.py binds
# one rank per GPU, round-robin).
#
# NOTE: the login/head node has no MI300A GPU, so submit this with `sbatch` to a
# compute node. The GPU sweeps are skipped automatically wherever no GPU is
# visible, so a CPU-only dry run on the head node still works.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Environment
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
# allocation (e.g. a CPU-only dry run on the head node) this stays unset and the
# GPU sweeps are skipped automatically.
GPUS_ALLOC="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-}}"
if [[ -n "$GPUS_ALLOC" ]]; then
    export MUGRID_BENCH_GPU_COUNT="$GPUS_ALLOC"
    export MUGRID_BENCH_GPU_NAME="AMD Instinct MI300A"
fi

cd "$REPO"

# --------------------------------------------------------------------------- #
# Scaling configuration (shared by both studies)
# --------------------------------------------------------------------------- #
CPU_RANKS="1 2 4 8 16 32 64 92"   # CPU strong-scaling ladder (full node = 92)
GPU_RANKS="1 2 4"                  # GPU strong-scaling sweep (4 MI300A APUs)
SCALING_SIZES="64 96"              # grid sizes used for the scaling sweeps

# --------------------------------------------------------------------------- #
# Study 1 — homogenization WITHOUT preconditioner
# --------------------------------------------------------------------------- #
echo "=== Homogenization benchmark (no preconditioner) ==="
python3 examples/benchmark_homogenization.py \
    --sizes 16 24 32 48 64 96 128 \
    --mpi-cpu-ranks 92 \
    --scaling-sizes $SCALING_SIZES \
    --scaling-ranks $CPU_RANKS \
    --scaling-gpu-ranks $GPU_RANKS \
    --maxiter 100 \
    --doc-out docs/benchmark_homogenization.md

# --------------------------------------------------------------------------- #
# Study 2 — homogenization WITH the reference Fourier preconditioner
# --------------------------------------------------------------------------- #
echo "=== Homogenization benchmark (reference Fourier preconditioner) ==="
python3 examples/benchmark_homogenization_preconditioner.py \
    --sizes 16 24 32 48 64 96 128 \
    --iter-sizes 16 24 32 48 64 \
    --mpi-cpu-ranks 92 \
    --scaling-sizes $SCALING_SIZES \
    --scaling-ranks $CPU_RANKS \
    --scaling-gpu-ranks $GPU_RANKS \
    --maxiter 20000 \
    --tol 1e-6 \
    --doc-out docs/benchmark_homogenization_preconditioner.md

echo
echo "Done. New rows appended to benchmarks/results.csv and pages regenerated:"
echo "  docs/benchmark_homogenization.md"
echo "  docs/benchmark_homogenization_preconditioner.md"
echo
echo "Commit the database and pages with:"
echo "  git add benchmarks/results.csv docs/benchmark_homogenization*.md docs/benchmark_homogenization*.png"
