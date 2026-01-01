#!/bin/bash
#
# Benchmark script for Homogenization solver
#
# This script runs the homogenization solver with different grid sizes,
# comparing performance between fused and generic kernels.
#
# Usage:
#   ./benchmark_homogenization.sh [cpu|gpu] [2d|3d] [maxiter]
#
# Arguments:
#   cpu|gpu      - Device for computation (default: cpu)
#   2d|3d        - Grid dimensionality (default: 2d)
#   maxiter      - Maximum CG iterations per load case (default: 100)
#
# Environment variables:
#   PYTHON       - Python interpreter to use (default: python3)
#   PYTHONPATH   - Python path (set automatically if not defined)
#
# Requirements:
#   - jq (for JSON processing)
#   - Python with muGrid installed
#   - CuPy (for GPU execution)
#

set -e

# Default parameters
DEVICE="${1:-cpu}"
DIM="${2:-2d}"
MAXITER="${3:-100}"

# Use PYTHON environment variable or default to python3
PYTHON="${PYTHON:-python3}"

# Validate arguments
if [[ "$DEVICE" != "cpu" && "$DEVICE" != "gpu" ]]; then
    echo "Error: First argument must be 'cpu' or 'gpu'"
    echo "Usage: $0 [cpu|gpu] [2d|3d] [maxiter]"
    exit 1
fi

if [[ "$DIM" != "2d" && "$DIM" != "3d" ]]; then
    echo "Error: Second argument must be '2d' or '3d'"
    echo "Usage: $0 [cpu|gpu] [2d|3d] [maxiter]"
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

# Define grid sizes based on dimensionality
if [[ "$DIM" == "2d" ]]; then
    GRID_SIZES=("16,16" "32,32" "64,64" "128,128" "256,256")
else
    GRID_SIZES=("8,8,8" "16,16,16" "24,24,24" "32,32,32" "48,48,48")
fi

# Kernel implementations to compare
KERNELS=("fused" "generic")

# Inclusion types to test
INCLUSION_TYPES=("single" "checkerboard")

# Output file for results
RESULTS_FILE="/tmp/homogenization_benchmark_results.json"
echo "[]" > "$RESULTS_FILE"

echo "============================================================"
echo "Homogenization Benchmark"
echo "============================================================"
echo "Python:      $PYTHON"
echo "Device:      $DEVICE"
echo "Dimensions:  $DIM"
echo "Max iter:    $MAXITER"
echo "Grid sizes:  ${GRID_SIZES[*]}"
echo "Kernels:     ${KERNELS[*]}"
echo "============================================================"
echo ""

# Run benchmarks
for grid in "${GRID_SIZES[@]}"; do
    for kernel in "${KERNELS[@]}"; do
        for incl_type in "${INCLUSION_TYPES[@]}"; do
            echo -n "Running: grid=$grid, kernel=$kernel, inclusion=$incl_type ... "

            # Run the solver and capture JSON output
            result=$("$PYTHON" "$HOMOG_PY" \
                -n "$grid" \
                -d "$DEVICE" \
                -k "$kernel" \
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
            flops=$(echo "$result" | jq -r '.results.flops_rate_GFLOPs_estimated')
            iterations=$(echo "$result" | jq -r '.results.total_cg_iterations')

            echo "done (time: ${total_time}s, ${iterations} iters, ${throughput} GB/s, ${flops} GFLOP/s)"

            # Append to results file
            jq --argjson new "$result" '. += [$new]' "$RESULTS_FILE" > "${RESULTS_FILE}.tmp"
            mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"
        done
    done
done

echo ""
echo "============================================================"
echo "Summary: Performance by Grid Size (single inclusion, fused kernel)"
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
              | select(.config.inclusion_type == "single")
              | select(.config.kernel == "fused")] | .[0] // empty' \
        "$RESULTS_FILE")

    if [[ -n "$result" ]]; then
        nb_pts=$(echo "$result" | jq -r '.config.nb_grid_pts_total')
        total_time=$(echo "$result" | jq -r '.results.total_time_seconds')
        iterations=$(echo "$result" | jq -r '.results.total_cg_iterations')
        throughput=$(echo "$result" | jq -r '.results.memory_throughput_GBps')
        flops=$(echo "$result" | jq -r '.results.flops_rate_GFLOPs_estimated')

        printf "%-12s %12s %10s %12.4f %10.2f %10.2f\n" \
            "$grid" "$nb_pts" "$iterations" "$total_time" "$throughput" "$flops"
    fi
done

echo ""
echo "============================================================"
echo "Summary: Fused vs Generic Kernel (single inclusion)"
echo "============================================================"
echo ""

printf "%-12s %12s %12s %10s\n" \
    "Grid Size" "Fused (s)" "Generic (s)" "Speedup"
printf "%-12s %12s %12s %10s\n" \
    "------------" "------------" "------------" "----------"

for grid in "${GRID_SIZES[@]}"; do
    fused_time=$(jq -r --arg g "$grid" \
        '[.[] | select(.config.nb_grid_pts == ($g | split(",") | map(tonumber)))
              | select(.config.inclusion_type == "single")
              | select(.config.kernel == "fused")] | .[0].results.total_time_seconds // "N/A"' \
        "$RESULTS_FILE")

    generic_time=$(jq -r --arg g "$grid" \
        '[.[] | select(.config.nb_grid_pts == ($g | split(",") | map(tonumber)))
              | select(.config.inclusion_type == "single")
              | select(.config.kernel == "generic")] | .[0].results.total_time_seconds // "N/A"' \
        "$RESULTS_FILE")

    if [[ "$fused_time" != "N/A" && "$generic_time" != "N/A" && "$fused_time" != "0" ]]; then
        speedup=$(echo "scale=2; $generic_time / $fused_time" | bc)
        printf "%-12s %12.4f %12.4f %9sx\n" \
            "$grid" "$fused_time" "$generic_time" "$speedup"
    else
        printf "%-12s %12s %12s %10s\n" \
            "$grid" "$fused_time" "$generic_time" "N/A"
    fi
done

echo ""
echo "============================================================"
echo "Summary: Homogenized Properties (single inclusion, fused)"
echo "============================================================"
echo ""

printf "%-12s %12s %12s %12s %12s\n" \
    "Grid Size" "E_eff" "E_voigt" "E_reuss" "Vol. Frac."
printf "%-12s %12s %12s %12s %12s\n" \
    "------------" "------------" "------------" "------------" "------------"

for grid in "${GRID_SIZES[@]}"; do
    result=$(jq -r --arg g "$grid" \
        '[.[] | select(.config.nb_grid_pts == ($g | split(",") | map(tonumber)))
              | select(.config.inclusion_type == "single")
              | select(.config.kernel == "fused")] | .[0] // empty' \
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

# Save results with device suffix for later comparison
cp "$RESULTS_FILE" "/tmp/homogenization_benchmark_${DEVICE}.json"

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
if [[ "$DIM" == "2d" ]]; then
    echo "The homogenization solves 3 load cases (xx, yy, xy strain)."
else
    echo "The homogenization solves 6 load cases (xx, yy, zz, yz, xz, xy strain)."
fi
echo "Total CG iterations is the sum across all cases."
echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Comparison file:  /tmp/homogenization_benchmark_${DEVICE}.json"
echo ""
