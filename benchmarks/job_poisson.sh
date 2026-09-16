#!/bin/bash
#SBATCH --job-name=muGrid-poisson-scaling
#SBATCH --partition=mi300a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --account=bw17d009
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Poisson CG solve-time scaling: a single CPU core vs. a single GPU, across 3D
# grid sizes. Feeds the docs/benchmark.md page.
#
# Every finished point is cached (benchmarks/cache/, keyed by commit + point), so
# if this job hits the wall-clock limit you can simply resubmit it: the completed
# points replay from the cache in seconds and the sweep continues where it left
# off. Only ever benchmark a COMMITTED (clean) tree — dirty runs are not cached.
#
# The single-CPU-core sweep is parallelised across the allocated cores with
# --jobs (each point uses one core); the GPU sweep runs serially on the one GPU.
#
# Measures only: appends the `time_vs_size` study of the `poisson` benchmark to
# benchmarks/results.csv. Render the page afterwards with `benchmarks/make_docs.sh`.

set -euo pipefail
source "$HOME/Software/muGrid/benchmarks/_env.sh"

SIZES="32 48 64 96 128 192 256 384 512 768 1024"
MAX_SIZE=1024        # cap for the GPU config
CPU1_MAX_SIZE=256    # cap for the single CPU core (slow beyond this)

echo "=== Poisson scaling (solve time vs grid size), CPU core vs GPU ==="
python3 examples/benchmark.py \
    --collect-only \
    --sizes $SIZES \
    --max-size $MAX_SIZE \
    --cpu1-max-size $CPU1_MAX_SIZE \
    --jobs "${SLURM_CPUS_PER_TASK:-1}" \
    --maxiter 5000

echo
echo "Done. Rows appended to benchmarks/results.csv (benchmark: poisson)."
echo "If the job timed out, just resubmit — finished points replay from the cache."
echo "Render the page with: benchmarks/make_docs.sh"
