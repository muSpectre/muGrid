#!/bin/bash
#
# Benchmark script for Poisson solver
#
# This script runs the Poisson solver with different grid sizes and stencil
# implementations, comparing performance between generic and hard-coded stencils.
#
# Usage:
#   ./benchmark_poisson.sh [host|device] [2d|3d] [maxiter]
#
# Arguments:
#   host|device  - Memory location (default: host)
#   2d|3d        - Grid dimensionality (default: 3d)
#   maxiter      - Maximum CG iterations (default: 50)
#
# Environment variables:
#   PYTHON       - Python interpreter to use (default: python3)
#   PYTHONPATH   - Python path (set automatically if not defined)
#
# Requirements:
#   - jq (for JSON processing)
#   - Python with muGrid installed
#

set -e

# Default parameters
MEMORY="${1:-host}"
DIM="${2:-3d}"
MAXITER="${3:-50}"  # Limit iterations for consistent benchmarking

# Use PYTHON environment variable or default to python3
PYTHON="${PYTHON:-python3}"

# Validate arguments
if [[ "$MEMORY" != "host" && "$MEMORY" != "device" ]]; then
    echo "Error: First argument must be 'host' or 'device'"
    echo "Usage: $0 [host|device] [2d|3d]"
    exit 1
fi

if [[ "$DIM" != "2d" && "$DIM" != "3d" ]]; then
    echo "Error: Second argument must be '2d' or '3d'"
    echo "Usage: $0 [host|device] [2d|3d]"
    exit 1
fi

# Check for jq
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed."
    echo "Install with: brew install jq (macOS) or apt-get install jq (Linux)"
    exit 1
fi

# Find the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POISSON_PY="$SCRIPT_DIR/poisson.py"

if [[ ! -f "$POISSON_PY" ]]; then
    echo "Error: poisson.py not found at $POISSON_PY"
    exit 1
fi

# Define grid sizes based on dimensionality
if [[ "$DIM" == "2d" ]]; then
    GRID_SIZES=("32,32" "64,64" "128,128" "256,256" "512,512" "1024,1024")
else
    GRID_SIZES=("16,16,16" "32,32,32" "48,48,48" "64,64,64" "96,96,96" "128,128,128")
fi

# Stencil implementations to compare
STENCILS=("generic" "hardcoded")

# Output file for results
RESULTS_FILE="/tmp/poisson_benchmark_results.json"
echo "[]" > "$RESULTS_FILE"

echo "============================================================"
echo "Poisson Solver Benchmark"
echo "============================================================"
echo "Python:      $PYTHON"
echo "Memory:      $MEMORY"
echo "Dimensions:  $DIM"
echo "Max iter:    $MAXITER"
echo "Grid sizes:  ${GRID_SIZES[*]}"
echo "============================================================"
echo ""

# Run benchmarks
for grid in "${GRID_SIZES[@]}"; do
    for stencil in "${STENCILS[@]}"; do
        echo -n "Running: grid=$grid, stencil=$stencil ... "

        # Run the solver and capture JSON output
        result=$("$PYTHON" "$POISSON_PY" \
            -n "$grid" \
            -m "$MEMORY" \
            -s "$stencil" \
            -i "$MAXITER" \
            --json 2>&1)

        if [[ $? -ne 0 ]]; then
            echo "FAILED"
            echo "$result"
            continue
        fi

        # Extract key metrics
        apply_time=$(echo "$result" | jq -r '.results.apply_time_seconds')
        apply_throughput=$(echo "$result" | jq -r '.results.apply_throughput_GBps')
        apply_flops=$(echo "$result" | jq -r '.results.apply_flops_rate_GFLOPs')

        echo "done (apply: ${apply_time}s, ${apply_throughput} GB/s, ${apply_flops} GFLOP/s)"

        # Append to results file
        jq --argjson new "$result" '. += [$new]' "$RESULTS_FILE" > "${RESULTS_FILE}.tmp"
        mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"
    done
done

echo ""
echo "============================================================"
echo "Summary: Apply Kernel Performance"
echo "============================================================"
echo ""

# Print header
printf "%-15s %12s %12s %10s %10s %10s\n" \
    "Grid Size" "Generic" "Hardcoded" "Speedup" "GB/s" "GFLOP/s"
printf "%-15s %12s %12s %10s %10s %10s\n" \
    "---------------" "------------" "------------" "----------" "----------" "----------"

# Process results and compute speedup
for grid in "${GRID_SIZES[@]}"; do
    # Get apply times for both stencils
    generic_time=$(jq -r --arg g "$grid" \
        '[.[] | select(.config.nb_grid_pts == ($g | split(",") | map(tonumber)))
              | select(.config.stencil == "generic")] | .[0].results.apply_time_seconds // "N/A"' \
        "$RESULTS_FILE")

    hardcoded_time=$(jq -r --arg g "$grid" \
        '[.[] | select(.config.nb_grid_pts == ($g | split(",") | map(tonumber)))
              | select(.config.stencil == "hardcoded")] | .[0].results.apply_time_seconds // "N/A"' \
        "$RESULTS_FILE")

    hardcoded_throughput=$(jq -r --arg g "$grid" \
        '[.[] | select(.config.nb_grid_pts == ($g | split(",") | map(tonumber)))
              | select(.config.stencil == "hardcoded")] | .[0].results.apply_throughput_GBps // "N/A"' \
        "$RESULTS_FILE")

    hardcoded_flops=$(jq -r --arg g "$grid" \
        '[.[] | select(.config.nb_grid_pts == ($g | split(",") | map(tonumber)))
              | select(.config.stencil == "hardcoded")] | .[0].results.apply_flops_rate_GFLOPs // "N/A"' \
        "$RESULTS_FILE")

    # Compute speedup
    if [[ "$generic_time" != "N/A" && "$hardcoded_time" != "N/A" ]]; then
        speedup=$(echo "scale=2; $generic_time / $hardcoded_time" | bc)
    else
        speedup="N/A"
    fi

    # Format times for display
    if [[ "$generic_time" != "N/A" ]]; then
        generic_ms=$(printf "%.3f ms" $(echo "$generic_time * 1000" | bc))
    else
        generic_ms="N/A"
    fi

    if [[ "$hardcoded_time" != "N/A" ]]; then
        hardcoded_ms=$(printf "%.3f ms" $(echo "$hardcoded_time * 1000" | bc))
    else
        hardcoded_ms="N/A"
    fi

    if [[ "$hardcoded_throughput" != "N/A" ]]; then
        throughput_str=$(printf "%.2f" "$hardcoded_throughput")
    else
        throughput_str="N/A"
    fi

    if [[ "$hardcoded_flops" != "N/A" ]]; then
        flops_str=$(printf "%.2f" "$hardcoded_flops")
    else
        flops_str="N/A"
    fi

    printf "%-15s %12s %12s %9sx %10s %10s\n" \
        "$grid" "$generic_ms" "$hardcoded_ms" "$speedup" "$throughput_str" "$flops_str"
done

echo ""
echo "============================================================"
echo "Scaling Analysis (Hardcoded Stencil)"
echo "============================================================"
echo ""

printf "%-15s %12s %12s %10s %10s\n" \
    "Grid Size" "Grid Points" "Time/iter" "GB/s" "GFLOP/s"
printf "%-15s %12s %12s %10s %10s\n" \
    "---------------" "------------" "------------" "----------" "----------"

for grid in "${GRID_SIZES[@]}"; do
    # Get metrics for hardcoded stencil
    result=$(jq -r --arg g "$grid" \
        '[.[] | select(.config.nb_grid_pts == ($g | split(",") | map(tonumber)))
              | select(.config.stencil == "hardcoded")] | .[0] // empty' \
        "$RESULTS_FILE")

    if [[ -n "$result" ]]; then
        nb_pts=$(echo "$result" | jq -r '.config.nb_grid_pts_total')
        apply_throughput=$(echo "$result" | jq -r '.results.apply_throughput_GBps')
        apply_flops=$(echo "$result" | jq -r '.results.apply_flops_rate_GFLOPs')
        apply_time=$(echo "$result" | jq -r '.results.apply_time_seconds')
        iterations=$(echo "$result" | jq -r '.results.iterations')

        # Time per iteration
        time_per_iter_ms=$(echo "scale=3; $apply_time * 1000 / $iterations" | bc)

        printf "%-15s %12s %11s ms %10.2f %10.2f\n" \
            "$grid" "$nb_pts" "$time_per_iter_ms" "$apply_throughput" "$apply_flops"
    fi
done

echo ""
echo "============================================================"
echo "Notes"
echo "============================================================"
echo ""
echo "Bandwidth (GB/s): Estimated assuming 8 memory accesses per point"
echo "                  (7 stencil reads + 1 write). With cache reuse,"
echo "                  actual DRAM bandwidth is ~4x lower."
echo ""
echo "GFLOP/s:          Floating-point operations (14 FLOPs per point"
echo "                  for 7-point stencil with scaling)."
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
