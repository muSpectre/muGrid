#!/bin/bash
#
# Benchmark script for Homogenization solver
#
# This script runs the homogenization solver with different grid sizes,
# comparing performance between host (CPU) and device (GPU) execution.
#
# Usage:
#   ./benchmark_homogenization.sh [host|device] [maxiter]
#
# Arguments:
#   host|device  - Memory location (default: host)
#   maxiter      - Maximum CG iterations per load case (default: 100)
#
# Environment variables:
#   PYTHON       - Python interpreter to use (default: python3)
#   PYTHONPATH   - Python path (set automatically if not defined)
#
# Requirements:
#   - jq (for JSON processing)
#   - Python with muGrid installed
#   - CuPy (for device execution)
#

set -e

# Default parameters
MEMORY="${1:-host}"
MAXITER="${2:-100}"

# Use PYTHON environment variable or default to python3
PYTHON="${PYTHON:-python3}"

# Validate arguments
if [[ "$MEMORY" != "host" && "$MEMORY" != "device" ]]; then
    echo "Error: First argument must be 'host' or 'device'"
    echo "Usage: $0 [host|device] [maxiter]"
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
HOMOG_PY="$SCRIPT_DIR/homogenization.py"

if [[ ! -f "$HOMOG_PY" ]]; then
    echo "Error: homogenization.py not found at $HOMOG_PY"
    exit 1
fi

# Define grid sizes (2D only for homogenization)
GRID_SIZES=("16,16" "32,32" "64,64" "128,128" "256,256")

# Inclusion types to test
INCLUSION_TYPES=("single" "checkerboard")

# Output file for results
RESULTS_FILE="/tmp/homogenization_benchmark_results.json"
echo "[]" > "$RESULTS_FILE"

echo "============================================================"
echo "Homogenization Benchmark"
echo "============================================================"
echo "Python:      $PYTHON"
echo "Memory:      $MEMORY"
echo "Max iter:    $MAXITER"
echo "Grid sizes:  ${GRID_SIZES[*]}"
echo "============================================================"
echo ""

# Run benchmarks
for grid in "${GRID_SIZES[@]}"; do
    for incl_type in "${INCLUSION_TYPES[@]}"; do
        echo -n "Running: grid=$grid, inclusion=$incl_type ... "

        # Run the solver and capture JSON output
        result=$("$PYTHON" "$HOMOG_PY" \
            -n "$grid" \
            -m "$MEMORY" \
            -i "$MAXITER" \
            --inclusion-type "$incl_type" \
            --json 2>&1)

        if [[ $? -ne 0 ]]; then
            echo "FAILED"
            echo "$result"
            continue
        fi

        # Extract key metrics
        total_time=$(echo "$result" | jq -r '.results.total_time_seconds')
        throughput=$(echo "$result" | jq -r '.results.memory_throughput_GBps')
        flops=$(echo "$result" | jq -r '.results.flops_rate_GFLOPs')
        iterations=$(echo "$result" | jq -r '.results.total_cg_iterations')

        echo "done (time: ${total_time}s, ${iterations} iters, ${throughput} GB/s, ${flops} GFLOP/s)"

        # Append to results file
        jq --argjson new "$result" '. += [$new]' "$RESULTS_FILE" > "${RESULTS_FILE}.tmp"
        mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"
    done
done

echo ""
echo "============================================================"
echo "Summary: Performance by Grid Size (single inclusion)"
echo "============================================================"
echo ""

# Print header
printf "%-12s %12s %10s %12s %10s %10s\n" \
    "Grid Size" "Grid Points" "CG Iters" "Time (s)" "GB/s" "GFLOP/s"
printf "%-12s %12s %10s %12s %10s %10s\n" \
    "------------" "------------" "----------" "------------" "----------" "----------"

# Process results
for grid in "${GRID_SIZES[@]}"; do
    result=$(jq -r --arg g "$grid" \
        '[.[] | select(.config.nb_grid_pts == ($g | split(",") | map(tonumber)))
              | select(.config.inclusion_type == "single")] | .[0] // empty' \
        "$RESULTS_FILE")

    if [[ -n "$result" ]]; then
        nb_pts=$(echo "$result" | jq -r '.config.nb_grid_pts_total')
        total_time=$(echo "$result" | jq -r '.results.total_time_seconds')
        iterations=$(echo "$result" | jq -r '.results.total_cg_iterations')
        throughput=$(echo "$result" | jq -r '.results.memory_throughput_GBps')
        flops=$(echo "$result" | jq -r '.results.flops_rate_GFLOPs')

        printf "%-12s %12s %10s %12.4f %10.2f %10.2f\n" \
            "$grid" "$nb_pts" "$iterations" "$total_time" "$throughput" "$flops"
    fi
done

echo ""
echo "============================================================"
echo "Summary: Homogenized Properties (single inclusion)"
echo "============================================================"
echo ""

printf "%-12s %12s %12s %12s %12s\n" \
    "Grid Size" "E_eff" "E_voigt" "E_reuss" "Vol. Frac."
printf "%-12s %12s %12s %12s %12s\n" \
    "------------" "------------" "------------" "------------" "------------"

for grid in "${GRID_SIZES[@]}"; do
    result=$(jq -r --arg g "$grid" \
        '[.[] | select(.config.nb_grid_pts == ($g | split(",") | map(tonumber)))
              | select(.config.inclusion_type == "single")] | .[0] // empty' \
        "$RESULTS_FILE")

    if [[ -n "$result" ]]; then
        E_eff=$(echo "$result" | jq -r '.results.E_effective_approx')
        E_voigt=$(echo "$result" | jq -r '.results.E_voigt_bound')
        E_reuss=$(echo "$result" | jq -r '.results.E_reuss_bound')
        v_f=$(echo "$result" | jq -r '.config.volume_fraction')

        printf "%-12s %12.4f %12.4f %12.4f %12.4f\n" \
            "$grid" "$E_eff" "$E_voigt" "$E_reuss" "$v_f"
    fi
done

echo ""
echo "============================================================"
echo "Host vs Device Comparison (if both available)"
echo "============================================================"
echo ""

# If we have both host and device results, compare them
if [[ -f "/tmp/homogenization_benchmark_host.json" && -f "/tmp/homogenization_benchmark_device.json" ]]; then
    printf "%-12s %12s %12s %10s\n" \
        "Grid Size" "Host (s)" "Device (s)" "Speedup"
    printf "%-12s %12s %12s %10s\n" \
        "------------" "------------" "------------" "----------"

    for grid in "${GRID_SIZES[@]}"; do
        host_time=$(jq -r --arg g "$grid" \
            '[.[] | select(.config.nb_grid_pts == ($g | split(",") | map(tonumber)))
                  | select(.config.inclusion_type == "single")] | .[0].results.total_time_seconds // "N/A"' \
            "/tmp/homogenization_benchmark_host.json")

        device_time=$(jq -r --arg g "$grid" \
            '[.[] | select(.config.nb_grid_pts == ($g | split(",") | map(tonumber)))
                  | select(.config.inclusion_type == "single")] | .[0].results.total_time_seconds // "N/A"' \
            "/tmp/homogenization_benchmark_device.json")

        if [[ "$host_time" != "N/A" && "$device_time" != "N/A" ]]; then
            speedup=$(echo "scale=2; $host_time / $device_time" | bc)
            printf "%-12s %12.4f %12.4f %9sx\n" \
                "$grid" "$host_time" "$device_time" "$speedup"
        fi
    done
else
    echo "Run benchmark with both 'host' and 'device' to see comparison."
    echo "Results are saved to:"
    echo "  Host:   /tmp/homogenization_benchmark_host.json"
    echo "  Device: /tmp/homogenization_benchmark_device.json"
fi

# Save results with memory type suffix for later comparison
cp "$RESULTS_FILE" "/tmp/homogenization_benchmark_${MEMORY}.json"

echo ""
echo "============================================================"
echo "Notes"
echo "============================================================"
echo ""
echo "Memory throughput and FLOP rates are estimates based on:"
echo "  - FEM gradient/divergence operations"
echo "  - Stress tensor computations"
echo "  - Ghost communication overhead"
echo ""
echo "The homogenization solves 3 load cases (xx, yy, xy strain)."
echo "Total CG iterations is the sum across all cases."
echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Comparison file:  /tmp/homogenization_benchmark_${MEMORY}.json"
echo ""
