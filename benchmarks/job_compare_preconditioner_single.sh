#!/bin/bash
#SBATCH --job-name=muGrid-homog-compare-fp32
#SBATCH --partition=mi300a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=bw17d009
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# SINGLE PRECISION (fp32). The double-precision (fp64) counterpart is
# job_compare_preconditioner_double.sh; both append to the same DB and the docs
# overlay the two precisions (fp64 solid, fp32 dashed).
#
# Case 1 of 3 — preconditioner COMPARISON: unpreconditioned vs reference
# (Ladecký et al. 2023) Fourier preconditioner, CG iteration count vs grid size,
# both run to convergence. The count depends only on the operator and
# preconditioner — not the device or MPI decomposition — so this runs on a
# single CPU core and needs no GPUs and little memory.
#
# Note: the same relative tolerance (1e-6) is used as in fp64 so the comparison
# is like-for-like. Single-precision CG may stagnate near the fp32 residual
# floor before reaching it; if so it runs to --maxiter, which is exactly the
# precision effect this overlay is meant to reveal.
#
# Measures only: appends the `iterations` study of the
# `homogenization_preconditioner` benchmark to benchmarks/results.csv. Generate
# the documentation page afterwards with `benchmarks/make_docs.sh`.

set -euo pipefail
source "$HOME/Software/muGrid/benchmarks/_env.sh"

echo "=== Case 1: preconditioner comparison, fp32 (CG iteration count) ==="
python3 examples/benchmark_homogenization_preconditioner.py \
    --collect-only \
    --jobs "${SLURM_CPUS_PER_TASK:-1}" \
    --precision single \
    --studies iterations \
    --iter-sizes 16 24 32 48 64 \
    --maxiter 20000 \
    --tol 1e-6

echo
echo "Done. Rows appended to benchmarks/results.csv (study: iterations)."
echo "If the job timed out, just resubmit — finished points replay from the cache."
echo "Render the page with: benchmarks/make_docs.sh"
