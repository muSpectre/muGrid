#!/bin/bash
#SBATCH --job-name=muGrid-homog-scaling-ref-fp32
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
# job_scaling_preconditioned_double.sh; both append to the same DB and the docs
# overlay the two precisions (fp64 solid, fp32 dashed).
#
# Case 3 of 3 — reference-PRECONDITIONED scaling: reference-preconditioned solve
# time vs grid size for a single CPU core, the full node via MPI (92 ranks), a
# single GPU, and all 4 GPUs via MPI, run to convergence. Because the iteration
# count is grid-independent under the preconditioner, this isolates the
# per-iteration cost — dominated by the forward/inverse FFT pair, so it exercises
# the FFT-engine paths (single-precision cuFFT/rocFFT N-D transform on the GPU,
# slab MPI decomposition).
#
# Each config is swept in grid size up to its cap (single CPU core only to
# 128^3) and stops at the first size that runs OUT OF MEMORY, flagged `OOM` in
# the table and dropped from the plot.
#
# Measures only: appends the `reference_timing` study of the
# `homogenization_preconditioner` benchmark to benchmarks/results.csv. Generate
# the documentation page afterwards with `benchmarks/make_docs.sh`.

set -euo pipefail
source "$HOME/Software/muGrid/benchmarks/_env.sh"

SIZES="16 24 32 48 64 96 128 192 256 384 512 768 1024 1536 2048"
MAX_SIZE=2048        # cap for the full-CPU (MPI) and GPU configs
CPU1_MAX_SIZE=128    # cap for the single CPU core (hopeless beyond this)

echo "=== Case 3: reference-preconditioned scaling, fp32 (time vs grid size) ==="
python3 examples/benchmark_homogenization_preconditioner.py \
    --collect-only \
    --jobs 8 \
    --precision single \
    --studies reference_timing \
    --sizes $SIZES \
    --max-size $MAX_SIZE \
    --cpu1-max-size $CPU1_MAX_SIZE \
    --mpi-cpu-ranks 92 \
    --maxiter 10 \
    --tol 1e-6

echo
echo "Done. Rows appended to benchmarks/results.csv (study: reference_timing)."
echo "If the job timed out, just resubmit — finished points replay from the cache."
echo "Render the page with: benchmarks/make_docs.sh"
