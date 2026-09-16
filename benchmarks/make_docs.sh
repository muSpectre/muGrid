#!/bin/bash
# Regenerate ALL benchmark documentation — every page and plot — from the shared
# database (benchmarks/results.csv). This only renders; it does not measure, so
# it runs anywhere (including the login/head node, no GPU or MPI needed). Run the
# job_*.sh submission scripts first to (re)collect the data.
#
#   Poisson scaling            -> docs/benchmark.md (+ .png)
#   Homogenization (Case 2)    -> docs/benchmark_homogenization.md (+ .png)
#   Preconditioner (Cases 1&3) -> docs/benchmark_homogenization_preconditioner.md
#                                 (+ *_iters.png, *_time.png)
#
# Each page is rendered from the latest run of its study/studies AND each
# precision independently, so the cases (and the fp64/fp32 variants) can be
# measured by separate jobs at separate times and still combine on one page.

set -euo pipefail
source "$HOME/Software/muGrid/benchmarks/_env.sh"

echo "=== Rendering benchmark pages from benchmarks/results.csv ==="

# Poisson CG solve — CPU core vs GPU. Tolerant: until job_poisson.sh has appended
# `poisson` rows to the DB, leave the existing committed page in place.
python3 examples/benchmark.py --render-only --doc-out docs/benchmark.md \
    || echo "  (no poisson rows in the DB yet — leaving docs/benchmark.md as is)"

# Homogenization — unpreconditioned scaling (Case 2).
python3 examples/benchmark_homogenization.py --render-only \
    --doc-out docs/benchmark_homogenization.md

# Preconditioner — comparison + reference-preconditioned scaling (Cases 1 & 3;
# one page, the latest run of each study selected independently).
python3 examples/benchmark_homogenization_preconditioner.py --render-only \
    --doc-out docs/benchmark_homogenization_preconditioner.md

echo
echo "Done. Pages regenerated:"
echo "  docs/benchmark.md"
echo "  docs/benchmark_homogenization.md"
echo "  docs/benchmark_homogenization_preconditioner.md"
echo
echo "Commit the database and pages with:"
echo "  git add benchmarks/results.csv docs/benchmark*.md docs/benchmark*.png"
