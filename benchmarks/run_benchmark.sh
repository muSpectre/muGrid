#!/bin/bash
#SBATCH --job-name=muGrid-homog-bench
#SBATCH --partition=mi300a
#SBATCH --nodes=1
#SBATCH --ntasks=92
#SBATCH --cpus-per-task=1
#SBATCH --gpus=4
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --account=bw17d009
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Benchmark of the muGrid FEM-elasticity homogenization example, run on an
# MI300A compute node (4 APUs). Two studies are appended to the shared benchmark
# database (benchmarks/results.csv) and the documentation pages are regenerated:
#
#   1. examples/benchmark_homogenization.py
#      WITHOUT preconditioner: solve time vs. grid size for a single CPU core,
#      the full node via MPI (92 ranks), a single GPU, and all 4 GPUs via MPI.
#
#   2. examples/benchmark_homogenization_preconditioner.py
#      WITH the reference-material (Ladecky et al. 2023) Fourier preconditioner:
#      iteration count vs. grid size (none vs. reference), and the same
#      single-core / full-node-MPI / 1-GPU / 4-GPU solve-time comparison.
#
# Each configuration is swept in grid size up to 2048^3 (single CPU core only to
# 128^3) and stops at the first size that runs OUT OF MEMORY, which is flagged
# `OOM` in the table and dropped from the plot. The realistic ceiling is far
# below 2048^3 (see the README); the large cap just lets each config run until it
# actually exhausts memory.
#
# NOTE: the login/head node has no MI300A GPU, so submit this with `sbatch` to a
# compute node. The GPU curves are skipped automatically wherever no GPU is
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
# Grid-size sweep (shared by both studies)
# --------------------------------------------------------------------------- #
SIZES="16 24 32 48 64 96 128 192 256 384 512 768 1024 1536 2048"
MAX_SIZE=2048        # cap for the full-CPU (MPI) and GPU configs
CPU1_MAX_SIZE=128    # cap for the single CPU core (hopeless beyond this)

# --------------------------------------------------------------------------- #
# Study 1 — homogenization WITHOUT preconditioner
# --------------------------------------------------------------------------- #
echo "=== Homogenization benchmark (no preconditioner) ==="
python3 examples/benchmark_homogenization.py \
    --sizes $SIZES \
    --max-size $MAX_SIZE \
    --cpu1-max-size $CPU1_MAX_SIZE \
    --mpi-cpu-ranks 92 \
    --maxiter 100 \
    --doc-out docs/benchmark_homogenization.md

# --------------------------------------------------------------------------- #
# Study 2 — homogenization WITH the reference Fourier preconditioner
# --------------------------------------------------------------------------- #
echo "=== Homogenization benchmark (reference Fourier preconditioner) ==="
python3 examples/benchmark_homogenization_preconditioner.py \
    --sizes $SIZES \
    --max-size $MAX_SIZE \
    --cpu1-max-size $CPU1_MAX_SIZE \
    --iter-sizes 16 24 32 48 64 \
    --mpi-cpu-ranks 92 \
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
