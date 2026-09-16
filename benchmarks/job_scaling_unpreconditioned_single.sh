#!/bin/bash
#SBATCH --job-name=muGrid-homog-scaling-none-fp32
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

# SINGLE PRECISION (fp32). The double-precision (fp64) counterpart is
# job_scaling_unpreconditioned_double.sh; both append to the same DB and the
# docs overlay the two precisions (fp64 solid, fp32 dashed).
#
# Case 2 of 3 — UNPRECONDITIONED scaling: solve time vs grid size for a single
# CPU core, the full node via MPI (92 ranks), a single GPU, and all 4 GPUs via
# MPI, at a fixed CG-iteration budget (identical arithmetic per configuration).
# In single precision each field is half the size, so the memory ceiling (and
# the `OOM` cutoff) is pushed to larger grids than in fp64.
#
# Each config is swept in grid size up to its cap (single CPU core only to
# 128^3) and stops at the first size that runs OUT OF MEMORY, flagged `OOM` in
# the table and dropped from the plot.
#
# Measures only: appends the `time_vs_size` study of the `homogenization`
# benchmark to benchmarks/results.csv. Generate the documentation page
# afterwards with `benchmarks/make_docs.sh`.

set -euo pipefail
source "$HOME/Software/muGrid/benchmarks/_env.sh"

SIZES="16 24 32 48 64 96 128 192 256 384 512 768 1024 1536 2048"
MAX_SIZE=2048        # cap for the full-CPU (MPI) and GPU configs
CPU1_MAX_SIZE=128    # cap for the single CPU core (hopeless beyond this)

echo "=== Case 2: unpreconditioned scaling, fp32 (solve time vs grid size) ==="
python3 examples/benchmark_homogenization.py \
    --collect-only \
    --jobs 8 \
    --precision single \
    --sizes $SIZES \
    --max-size $MAX_SIZE \
    --cpu1-max-size $CPU1_MAX_SIZE \
    --mpi-cpu-ranks 92 \
    --maxiter 100

echo
echo "Done. Rows appended to benchmarks/results.csv (study: time_vs_size)."
echo "If the job timed out, just resubmit — finished points replay from the cache."
echo "Render the page with: benchmarks/make_docs.sh"
