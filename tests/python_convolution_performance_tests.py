#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_convolution_performance_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    12 Nov 2025

@brief   Performance tests for convolution operators

Copyright © 2025 Lars Pastewka

µGrid is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µGrid is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µGrid; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import time
import unittest
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse import coo_array

import muGrid
from muGrid import wrap_field
from muGrid.Solvers import conjugate_gradients

# Try to import pypapi for hardware counter access
try:
    import pypapi
    from pypapi import events as papi_events

    PAPI_AVAILABLE = True
except ImportError:
    PAPI_AVAILABLE = False
    print("Warning: pypapi not available. Install with: pip install pypapi")
    print("Performance tests will run without hardware counters.")

# Try to import CuPy for GPU tests
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# Use compile-time feature flag for GPU availability
GPU_AVAILABLE = muGrid.has_gpu


def get_gpu_skip_reason():
    """Get detailed skip reason for GPU tests."""
    if not GPU_AVAILABLE:
        return "GPU backend not available (muGrid not compiled with CUDA/HIP)"
    return None


def get_gpu_cupy_skip_reason():
    """Get detailed skip reason for GPU+CuPy tests."""
    reasons = []
    if not GPU_AVAILABLE:
        reasons.append("GPU backend not available (muGrid not compiled with CUDA/HIP)")
    if not HAS_CUPY:
        reasons.append("CuPy not installed (pip install cupy-cuda*)")
    if reasons:
        return "; ".join(reasons)
    return None


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""

    wall_time_ms: float
    total_flops: int
    grid_size: tuple
    stencil_size: tuple
    nb_components: int
    nb_operators: int
    nb_quad_pts: int

    # Device info
    device: str = "cpu"

    # Hardware counters (optional)
    papi_cycles: Optional[int] = None
    papi_instructions: Optional[int] = None
    papi_fp_ops: Optional[int] = None
    papi_l1_dcm: Optional[int] = None
    papi_l2_dcm: Optional[int] = None
    papi_l3_tcm: Optional[int] = None

    @property
    def gflops(self) -> float:
        """Calculate GFLOPS from theoretical FLOPs"""
        return (self.total_flops / 1e9) / (self.wall_time_ms / 1000.0)

    @property
    def actual_gflops(self) -> float:
        """Calculate GFLOPS from PAPI measured FP operations"""
        if self.papi_fp_ops is not None:
            return (self.papi_fp_ops / 1e9) / (self.wall_time_ms / 1000.0)
        return self.gflops

    @property
    def ipc(self) -> Optional[float]:
        """Calculate instructions per cycle"""
        if self.papi_cycles is not None and self.papi_instructions is not None:
            return self.papi_instructions / self.papi_cycles
        return None

    @property
    def l1_miss_rate(self) -> Optional[float]:
        """Calculate L1 cache miss rate as percentage"""
        if self.papi_l1_dcm is not None and self.papi_instructions is not None:
            return (self.papi_l1_dcm / self.papi_instructions) * 100.0
        return None

    @property
    def l2_miss_rate(self) -> Optional[float]:
        """Calculate L2 cache miss rate as percentage"""
        if self.papi_l2_dcm is not None and self.papi_instructions is not None:
            return (self.papi_l2_dcm / self.papi_instructions) * 100.0
        return None

    @property
    def l3_miss_rate(self) -> Optional[float]:
        """Calculate L3 cache miss rate as percentage"""
        if self.papi_l3_tcm is not None and self.papi_instructions is not None:
            return (self.papi_l3_tcm / self.papi_instructions) * 100.0
        return None

    def print_report(self):
        """Print a formatted performance report"""
        print("\n" + "=" * 70)
        print("PERFORMANCE METRICS")
        print("=" * 70)
        print(f"Device:             {self.device}")
        print(
            f"Grid size:          {self.grid_size[0]} x {self.grid_size[1]} "
            f"= {np.prod(self.grid_size)} pixels"
        )
        print(f"Stencil size:       {self.stencil_size[0]} x {self.stencil_size[1]}")
        print(f"Field components:   {self.nb_components}")
        print(f"Operators:          {self.nb_operators}")
        print(f"Quadrature points:  {self.nb_quad_pts}")
        print("-" * 70)
        print(f"Wall time:          {self.wall_time_ms:.3f} ms")
        print(f"Theoretical FLOPs:  {self.total_flops:,}")
        print(f"Theoretical GFLOPS: {self.gflops:.3f}")

        if self.papi_fp_ops is not None:
            print("-" * 70)
            print("PAPI Hardware Counters:")
            print(f"  Actual FP Ops:    {self.papi_fp_ops:,}")
            print(f"  Actual GFLOPS:    {self.actual_gflops:.3f}")
            print(f"  Total cycles:     {self.papi_cycles:,}")
            print(f"  Total instr:      {self.papi_instructions:,}")
            print(f"  IPC:              {self.ipc:.2f}")
            print(
                f"  L1 cache misses:  {self.papi_l1_dcm:,} "
                f"({self.l1_miss_rate:.2f}% of instructions)"
            )
            print(
                f"  L2 cache misses:  {self.papi_l2_dcm:,} "
                f"({self.l2_miss_rate:.2f}% of instructions)"
            )
            print(
                f"  L3 cache misses:  {self.papi_l3_tcm:,} "
                f"({self.l3_miss_rate:.2f}% of instructions)"
            )
        print("=" * 70 + "\n")


def count_theoretical_flops(conv_op, nodal_field, quad_field, grid_size):
    """
    Count theoretical FLOPs for a convolution operation.

    The convolution operation at each pixel involves:
    - Looping over stencil points (nb_conv_pts)
    - For each stencil point, looping over nodal sub-points (nb_pixelnodal_pts)
    - Computing: quad_vals += alpha * nodal_vals * operator_vals

    Each outer product (vector * vector) involves:
    - nb_components multiplications (for the column vector)
    - nb_operators * nb_quad_pts multiplications (for broadcasting)
    - Same number of additions

    Total per pixel: 2 * nb_components * nb_operators * nb_quad_pts *
        nb_conv_pts * nb_nodal_pts
    """
    nb_pixels = np.prod(grid_size)
    nb_components = nodal_field.nb_components
    nb_operators = conv_op.nb_operators
    nb_quad_pts = conv_op.nb_quad_pts
    nb_nodal_pts = conv_op.nb_nodal_pts

    # Get stencil shape from the operator
    # This is a bit tricky - we need to compute nb_conv_pts from the stencil
    # For now, estimate from the operator matrix dimensions
    # pixel_operator returns a flat list, so we compute nb_conv_pts from total size
    operator_list = conv_op.pixel_operator
    total_elements = len(operator_list)
    # Total elements = nb_operators * nb_quad_pts * nb_nodal_pts * nb_conv_pts
    nb_conv_pts = total_elements // (nb_operators * nb_quad_pts * nb_nodal_pts)

    # Each convolution point contributes:
    # - nb_components values (from nodal field)
    # - multiplied by nb_operators * nb_quad_pts values (from operator)
    # - 2 FLOPs per element (multiply + add)
    flops_per_pixel = (
        2 * nb_components * nb_operators * nb_quad_pts * nb_conv_pts * nb_nodal_pts
    )

    total_flops = flops_per_pixel * nb_pixels

    return total_flops


def measure_convolution_performance(
    conv_op,
    nodal_field,
    quad_field,
    grid_size,
    stencil_size,
    num_iterations=10,
    use_papi=True,
    device="cpu",
):
    """
    Measure performance of convolution operator.

    Parameters
    ----------
    conv_op : muGrid.ConvolutionOperator
        The convolution operator to benchmark
    nodal_field : muGrid.Field
        Input nodal field
    quad_field : muGrid.Field
        Output quadrature field
    grid_size : tuple
        Size of the grid (nx, ny)
    stencil_size : tuple
        Size of the stencil (sx, sy)
    num_iterations : int
        Number of iterations for averaging
    use_papi : bool
        Whether to use PAPI hardware counters
    device : str
        Device string (e.g., "cpu", "cuda:0", "rocm:0")

    Returns
    -------
    PerformanceMetrics
        Container with all performance metrics
    """
    is_on_device = nodal_field.is_on_gpu

    # Count theoretical FLOPs
    total_flops = count_theoretical_flops(conv_op, nodal_field, quad_field, grid_size)

    # Warm-up iteration
    conv_op.apply(nodal_field, quad_field)

    # Synchronize if on device to ensure warm-up is complete
    if is_on_device and HAS_CUPY:
        cp.cuda.Device().synchronize()

    # Initialize PAPI if available, requested, and on host (PAPI doesn't work for GPU)
    papi_values = {}
    if PAPI_AVAILABLE and use_papi and not is_on_device:
        try:
            # Define events to monitor
            papi_events_list = [
                papi_events.PAPI_TOT_CYC,  # Total cycles
                papi_events.PAPI_TOT_INS,  # Total instructions
                papi_events.PAPI_FP_OPS,  # Floating point operations
                papi_events.PAPI_L1_DCM,  # L1 data cache misses
                papi_events.PAPI_L2_DCM,  # L2 data cache misses
                papi_events.PAPI_L3_TCM,  # L3 total cache misses
            ]

            pypapi.papi_high.start_counters(papi_events_list)
            papi_enabled = True
        except Exception as e:
            print(f"Warning: Could not initialize PAPI: {e}")
            papi_enabled = False
    else:
        papi_enabled = False

    # Time the operation
    if is_on_device and HAS_CUPY:
        # Use CUDA events for accurate GPU timing
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        for _ in range(num_iterations):
            conv_op.apply(nodal_field, quad_field)
        end_event.record()
        end_event.synchronize()

        # Get elapsed time in milliseconds
        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        wall_time_ms = elapsed_ms / num_iterations
    else:
        # Use wall-clock time for CPU
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            conv_op.apply(nodal_field, quad_field)

        end_time = time.perf_counter()
        wall_time_ms = ((end_time - start_time) / num_iterations) * 1000.0

    # Read PAPI counters
    if papi_enabled:
        try:
            counters = pypapi.papi_high.stop_counters()
            papi_values = {
                "cycles": counters[0] // num_iterations,
                "instructions": counters[1] // num_iterations,
                "fp_ops": counters[2] // num_iterations,
                "l1_dcm": counters[3] // num_iterations,
                "l2_dcm": counters[4] // num_iterations,
                "l3_tcm": counters[5] // num_iterations,
            }
        except Exception as e:
            print(f"Warning: Could not read PAPI counters: {e}")
            papi_values = {}

    # Create metrics object
    metrics = PerformanceMetrics(
        wall_time_ms=wall_time_ms,
        total_flops=total_flops,
        grid_size=grid_size,
        stencil_size=stencil_size,
        nb_components=nodal_field.nb_components,
        nb_operators=conv_op.nb_operators,
        nb_quad_pts=conv_op.nb_quad_pts,
        device=device,
        papi_cycles=papi_values.get("cycles"),
        papi_instructions=papi_values.get("instructions"),
        papi_fp_ops=papi_values.get("fp_ops"),
        papi_l1_dcm=papi_values.get("l1_dcm"),
        papi_l2_dcm=papi_values.get("l2_dcm"),
        papi_l3_tcm=papi_values.get("l3_tcm"),
    )

    return metrics


class ConvolutionPerformanceTests(unittest.TestCase):
    """Test suite for convolution operator performance"""

    def test_performance_small_grid(self):
        """Benchmark convolution on a small 32x32 grid"""
        print("\n=== Testing 2D Convolution Performance (Small Grid) ===")

        # Parameters
        nb_x_pts = 32
        nb_y_pts = 32
        nb_stencil_x = 3
        nb_stencil_y = 3
        nb_operators = 2
        nb_quad_pts = 4
        nb_field_components = 3

        # Create stencil
        stencil = np.random.rand(nb_operators, nb_quad_pts, nb_stencil_x, nb_stencil_y)

        conv_op = muGrid.ConvolutionOperator([-1, -1], stencil)

        # Create field collection
        nb_ghosts = (1, 1)
        fc = muGrid.GlobalFieldCollection(
            (nb_x_pts, nb_y_pts),
            sub_pts={"quad": nb_quad_pts},
            nb_ghosts_left=nb_ghosts,
            nb_ghosts_right=nb_ghosts,
        )

        # Create fields
        nodal_cpp = fc.real_field("nodal", nb_field_components)
        quad_cpp = fc.real_field("quad", (nb_field_components, nb_operators), "quad")
        nodal = wrap_field(nodal_cpp)

        # Initialize with random data
        nodal.p[...] = np.random.rand(*nodal.p.shape)

        # Measure performance
        metrics = measure_convolution_performance(
            conv_op,
            nodal_cpp,
            quad_cpp,
            grid_size=(nb_x_pts, nb_y_pts),
            stencil_size=(nb_stencil_x, nb_stencil_y),
            num_iterations=100,
        )

        metrics.print_report()

        # Sanity checks
        self.assertGreater(metrics.total_flops, 0)
        self.assertGreater(metrics.gflops, 0.0)

    def test_performance_large_grid(self):
        """Benchmark convolution on a large 256x256 grid"""
        print("\n=== Testing 2D Convolution Performance (Large Grid) ===")

        # Parameters
        nb_x_pts = 256
        nb_y_pts = 256
        nb_stencil_x = 3
        nb_stencil_y = 3
        nb_operators = 2
        nb_quad_pts = 4
        nb_field_components = 3

        # Create stencil
        stencil = np.random.rand(
            nb_operators, nb_quad_pts, 1, nb_stencil_x, nb_stencil_y
        )

        conv_op = muGrid.ConvolutionOperator([-1, -1], stencil)

        # Create field collection
        comm = muGrid.Communicator()
        subdivisions = (1, 1)
        decomposition = muGrid.CartesianDecomposition(
            comm, (nb_x_pts, nb_y_pts), subdivisions, (1, 1), (1, 1)
        )
        fc = decomposition
        fc.set_nb_sub_pts("quad", nb_quad_pts)

        # Create fields
        nodal_cpp = fc.real_field("nodal", nb_field_components)
        quad_cpp = fc.real_field("quad", (nb_field_components, nb_operators), "quad")
        nodal = wrap_field(nodal_cpp)

        # Initialize with random data
        nodal.p[...] = np.random.rand(*nodal.p.shape)

        # Measure performance (fewer iterations for large grid)
        metrics = measure_convolution_performance(
            conv_op,
            nodal_cpp,
            quad_cpp,
            grid_size=(nb_x_pts, nb_y_pts),
            stencil_size=(nb_stencil_x, nb_stencil_y),
            num_iterations=10,
        )

        metrics.print_report()

        # Sanity checks
        self.assertGreater(metrics.total_flops, 0)
        self.assertGreater(metrics.gflops, 0.0)

        if metrics.l1_miss_rate is not None:
            # Cache miss rate should be reasonable
            self.assertLess(metrics.l1_miss_rate, 50.0)

    def test_performance_comparison_grid_sizes(self):
        """Compare performance across different grid sizes"""
        print("\n=== Comparing Performance Across Grid Sizes ===")

        grid_sizes = [16, 32, 64, 128, 256]
        results = []

        for grid_size in grid_sizes:
            nb_stencil = 3
            nb_operators = 2
            nb_quad_pts = 4
            nb_components = 3

            # Create stencil
            stencil = np.random.rand(nb_operators, nb_quad_pts, nb_stencil, nb_stencil)

            conv_op = muGrid.ConvolutionOperator([-1, -1], stencil)

            # Create field collection
            nb_ghosts = (1, 1)
            fc = muGrid.GlobalFieldCollection(
                (grid_size, grid_size),
                sub_pts={"quad": nb_quad_pts},
                nb_ghosts_left=nb_ghosts,
                nb_ghosts_right=nb_ghosts,
            )

            # Create fields
            nodal_cpp = fc.real_field("nodal", nb_components)
            quad_cpp = fc.real_field("quad", (nb_components, nb_operators), "quad")
            nodal = wrap_field(nodal_cpp)

            # Initialize with random data
            nodal.p[...] = np.random.rand(*nodal.p.shape)

            # Measure performance
            iterations = 100 if grid_size <= 64 else 10
            metrics = measure_convolution_performance(
                conv_op,
                nodal_cpp,
                quad_cpp,
                grid_size=(grid_size, grid_size),
                stencil_size=(nb_stencil, nb_stencil),
                num_iterations=iterations,
            )

            results.append(metrics)

        # Print comparison table
        print("\n" + "=" * 90)
        print(
            f"{'Grid Size':>10} {'GFLOPS':>12} {'L1 Miss %':>12} "
            f"{'L2 Miss %':>12} {'L3 Miss %':>12} {'IPC':>12}"
        )
        print("-" * 90)

        for metrics in results:
            grid_pixels = np.prod(metrics.grid_size)
            l1_str = f"{metrics.l1_miss_rate:.2f}" if metrics.l1_miss_rate else "N/A"
            l2_str = f"{metrics.l2_miss_rate:.2f}" if metrics.l2_miss_rate else "N/A"
            l3_str = f"{metrics.l3_miss_rate:.2f}" if metrics.l3_miss_rate else "N/A"
            ipc_str = f"{metrics.ipc:.2f}" if metrics.ipc else "N/A"

            print(
                f"{grid_pixels:>10} {metrics.gflops:>12.3f} {l1_str:>12} "
                f"{l2_str:>12} {l3_str:>12} {ipc_str:>12}"
            )

        print("=" * 90 + "\n")


def run_laplace_solver(nb_grid_pts, use_device=False):
    """
    Run Laplace solver using muGrid convolution operator.

    Parameters
    ----------
    nb_grid_pts : tuple
        Grid size (nx, ny)
    use_device : bool
        If True, use GPU memory; otherwise use CPU

    Returns
    -------
    tuple
        (solution array, elapsed time in seconds, device string)
    """
    comm = muGrid.Communicator()
    subdivisions = (1, 1)

    # Determine memory location
    if use_device:
        memory_location = muGrid.GlobalFieldCollection.MemoryLocation.Device
    else:
        memory_location = muGrid.GlobalFieldCollection.MemoryLocation.Host

    # Setup problem
    decomposition = muGrid.CartesianDecomposition(
        comm, nb_grid_pts, subdivisions, (1, 1), (1, 1), memory_location=memory_location
    )
    fc = decomposition
    grid_spacing = 1 / np.array(nb_grid_pts)

    x, y = decomposition.coords
    i, j = decomposition.icoords

    rhs = decomposition.real_field("rhs")
    solution = decomposition.real_field("solution")

    # Initialize RHS - need to handle device arrays differently
    if use_device and HAS_CUPY:
        x_cp = cp.asarray(x)
        y_cp = cp.asarray(y)
        rhs_data = (1 + cp.cos(2 * cp.pi * x_cp) * cp.cos(2 * cp.pi * y_cp)) ** 10
        rhs_data -= cp.mean(rhs_data)
        rhs.p[...] = rhs_data
        solution.s[...] = 0
    else:
        rhs.p[...] = (1 + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)) ** 10
        rhs.p[...] -= np.mean(rhs.p)
        solution.s[...] = 0

    # Create Laplace operator
    stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplace = muGrid.ConvolutionOperator([-1, -1], stencil)

    # Get device string for reporting
    device_str = rhs.device

    # Define Hessian-vector product
    scale_factor = -np.mean(grid_spacing) ** 2

    def hessp_mugrid(x_field, Ax_field):
        decomposition.communicate_ghosts(x_field)
        laplace.apply(x_field, Ax_field)
        Ax = wrap_field(Ax_field)
        Ax.s[...] /= scale_factor
        return Ax_field

    # Time the solver
    if use_device and HAS_CUPY:
        cp.cuda.Device().synchronize()
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()

    t_start = time.perf_counter()
    conjugate_gradients(
        comm,
        fc,
        hessp_mugrid,
        rhs,
        solution,
        tol=1e-6,
        maxiter=1000,
    )
    t_elapsed = time.perf_counter() - t_start

    if use_device and HAS_CUPY:
        end_event.record()
        end_event.synchronize()
        t_elapsed = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0

    # Get solution back to host for comparison
    if use_device and HAS_CUPY:
        solution_arr = cp.asnumpy(solution.s)
    else:
        solution_arr = solution.s.copy()

    return solution_arr, t_elapsed, device_str


def test_laplace_mugrid_vs_scipy(nb_grid_pts=(512, 512)):
    """Test Laplace solver: muGrid convolution vs SciPy sparse matrix."""
    comm = muGrid.Communicator()
    subdivisions = (1, 1)

    # Run muGrid solution on host
    mugrid_solution, t_mugrid, device_str = run_laplace_solver(
        nb_grid_pts, use_device=False
    )

    # Setup for scipy comparison
    decomposition = muGrid.CartesianDecomposition(
        comm, nb_grid_pts, subdivisions, (1, 1), (1, 1)
    )
    fc = decomposition
    grid_spacing = 1 / np.array(nb_grid_pts)
    x, y = decomposition.coords
    i, j = decomposition.icoords

    solution = decomposition.real_field("solution")
    rhs = decomposition.real_field("rhs")
    rhs.p[...] = (1 + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)) ** 10
    rhs.p[...] -= np.mean(rhs.p)
    solution.s[...] = 0

    # scipy Laplace sparse matrix
    nb = np.prod(nb_grid_pts)

    def grid_index(i, j):
        return (i % nb_grid_pts[0]).reshape(-1) * nb_grid_pts[1] + (
            j % nb_grid_pts[1]
        ).reshape(-1)

    laplace_sparse = (
        coo_array(
            (-4 * np.ones(nb), (grid_index(i, j), grid_index(i, j))), shape=(nb, nb)
        )
        + coo_array(
            (np.ones(nb), (grid_index(i, j), grid_index(i + 1, j))), shape=(nb, nb)
        )
        + coo_array(
            (np.ones(nb), (grid_index(i, j), grid_index(i - 1, j))), shape=(nb, nb)
        )
        + coo_array(
            (np.ones(nb), (grid_index(i, j), grid_index(i, j + 1))), shape=(nb, nb)
        )
        + coo_array(
            (np.ones(nb), (grid_index(i, j), grid_index(i, j - 1))), shape=(nb, nb)
        )
    ).tocsr()

    def hessp_scipy(x_field, Ax_field):
        x_arr = wrap_field(x_field)
        Ax = wrap_field(Ax_field)
        Ax.p[...] = (laplace_sparse @ x_arr.p.reshape(-1)).reshape(nb_grid_pts)
        Ax.s[...] /= -np.mean(grid_spacing) ** 2
        return Ax_field

    t_scipy = -time.perf_counter()
    conjugate_gradients(
        comm,
        fc,
        hessp_scipy,
        rhs,
        solution,
        tol=1e-6,
        maxiter=1000,
    )
    t_scipy += time.perf_counter()

    # Check solutions agree
    np.testing.assert_allclose(solution.s, mugrid_solution)

    # Print timing result
    print(f"muGrid operator time ({device_str}):  {t_mugrid:.6f} s")
    print(f"SciPy Sparse time:  {t_scipy:.6f} s")

    # Check that the speed is at least comparable
    # assert t_mugrid < 1.05 * t_scipy, (
    #     f"muGrid slower than SciPy sparse: "
    #     f"muGrid {t_mugrid:.6f}s vs. SciPy sparse {t_scipy:.6f}s"
    # )


@unittest.skipUnless(GPU_AVAILABLE and HAS_CUPY, get_gpu_cupy_skip_reason() or "")
def test_laplace_device_vs_host():
    """Compare Laplace solver performance between CPU and GPU.

    This test uses CUDA-aware MPI for ghost communication on device fields.
    """
    nb_grid_pts = (256, 256)

    # Run on host
    host_solution, t_host, host_device = run_laplace_solver(
        nb_grid_pts, use_device=False
    )

    # Run on device
    device_solution, t_device, device_str = run_laplace_solver(
        nb_grid_pts, use_device=True
    )

    # Check solutions match
    # Note: CPU and GPU may have small differences due to:
    # - Different floating-point operation ordering
    # - FMA (fused multiply-add) vs separate mul+add
    # - Parallel reduction accumulation order
    np.testing.assert_allclose(
        device_solution,
        host_solution,
        rtol=1e-4,
        atol=1e-5,
        err_msg="Device and host Laplace solutions differ",
    )

    # Print timing comparison
    speedup = t_host / t_device if t_device > 0 else 0
    print(f"\nLaplace solver comparison ({nb_grid_pts[0]}x{nb_grid_pts[1]}):")
    print(f"  Host ({host_device}):   {t_host:.6f} s")
    print(f"  Device ({device_str}): {t_device:.6f} s")
    print(f"  Speedup: {speedup:.2f}x")


@unittest.skipUnless(GPU_AVAILABLE and HAS_CUPY, get_gpu_cupy_skip_reason() or "")
def test_laplace_device_scaling():
    """Test Laplace solver scaling on GPU across different grid sizes.

    This test uses CUDA-aware MPI for ghost communication on device fields.
    """
    grid_sizes = [64, 128, 256, 512]
    results = []

    print("\n=== Laplace Solver GPU Scaling ===")
    print(f"{'Grid':>10} {'Host (s)':>12} {'Device (s)':>12} {'Speedup':>10}")
    print("-" * 50)

    for size in grid_sizes:
        nb_grid_pts = (size, size)

        # Host timing
        _, t_host, _ = run_laplace_solver(nb_grid_pts, use_device=False)

        # Device timing
        _, t_device, device_str = run_laplace_solver(nb_grid_pts, use_device=True)

        speedup = t_host / t_device if t_device > 0 else 0
        results.append((size, t_host, t_device, speedup))

        print(f"{size}x{size:>4} {t_host:>12.6f} {t_device:>12.6f} {speedup:>10.2f}x")

    print("-" * 50)
    print(f"Device: {device_str}")


def get_quad_triangle_kernel():
    """Get the kernel for quadrature on triangles with 6 quadrature points."""
    return np.array(
        [
            [  # operator 1
                [[[2 / 3, 1 / 6], [1 / 6, 0]]],  # quadrature point 1
                [[[1 / 6, 1 / 6], [2 / 3, 0]]],  # quadrature point 2
                [[[1 / 6, 2 / 3], [1 / 6, 0]]],  # quadrature point 3
                [[[0, 1 / 6], [1 / 6, 2 / 3]]],  # quadrature point 4
                [[[0, 2 / 3], [1 / 6, 1 / 6]]],  # quadrature point 5
                [[[0, 1 / 6], [2 / 3, 1 / 6]]],  # quadrature point 6
            ]
        ]
    )


def get_quad_triangle_manual_matrix():
    """Get the manual matrix for quadrature on triangles."""
    return np.array(
        [
            [2 / 3, 1 / 6, 1 / 6, 0],  # q1
            [1 / 6, 2 / 3, 1 / 6, 0],  # q2
            [1 / 6, 1 / 6, 2 / 3, 0],  # q3
            [0, 1 / 6, 1 / 6, 2 / 3],  # q4
            [0, 1 / 6, 2 / 3, 1 / 6],  # q5
            [0, 2 / 3, 1 / 6, 1 / 6],  # q6
        ]
    )


def run_quad_triangle_mugrid(nb_grid_pts, use_device=False):
    """
    Run quadrature triangle computation using muGrid convolution.

    Parameters
    ----------
    nb_grid_pts : tuple
        Grid size (Nx, Ny)
    use_device : bool
        If True, use GPU memory

    Returns
    -------
    tuple
        (result array on host, elapsed time, device string)
    """
    Nx, Ny = nb_grid_pts

    # Determine memory location
    if use_device:
        memory_location = muGrid.GlobalFieldCollection.MemoryLocation.Device
    else:
        memory_location = muGrid.GlobalFieldCollection.MemoryLocation.Host

    # Create field collection
    fc = muGrid.GlobalFieldCollection(
        nb_grid_pts,
        sub_pts={"quad": 6},
        nb_ghosts_right=(1, 1),
        memory_location=memory_location,
    )
    nodal_field_cpp = fc.real_field("nodal")
    quad_field_cpp = fc.real_field("quad", 1, "quad")
    nodal_field = wrap_field(nodal_field_cpp)

    # Random nodal values
    init_field = np.random.rand(Nx, Ny)
    padded_field = np.pad(init_field, (0, 1), mode="wrap")

    # Initialize field
    if use_device and HAS_CUPY:
        nodal_field.pg[...] = cp.asarray(padded_field)
    else:
        nodal_field.pg[...] = padded_field

    # Create operator
    kernel = get_quad_triangle_kernel()
    op_mugrid = muGrid.ConvolutionOperator([0, 0], kernel)

    device_str = nodal_field_cpp.device

    # Time the operation
    if use_device and HAS_CUPY:
        cp.cuda.Device().synchronize()
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()

    t_start = time.perf_counter()
    op_mugrid.apply(nodal_field_cpp, quad_field_cpp)
    t_elapsed = time.perf_counter() - t_start

    if use_device and HAS_CUPY:
        end_event.record()
        end_event.synchronize()
        t_elapsed = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0

    # Get result back to host
    quad_field = wrap_field(quad_field_cpp)
    if use_device and HAS_CUPY:
        result = cp.asnumpy(quad_field.p)
    else:
        result = quad_field.p.copy()

    return result, t_elapsed, device_str, init_field


def quad_manual_combined(a, Nx, Ny, m_quad):
    """Compute quadrature manually using tensor-matrix multiplication."""
    F = a.reshape((Nx, Ny))

    # periodic wrap
    F = np.vstack([F, F[0:1, :]])
    F = np.hstack([F, F[:, 0:1]])

    # stack square-adjacent nodal values
    bl = F[:-1, :-1]
    br = F[:-1, 1:]
    tl = F[1:, :-1]
    tr = F[1:, 1:]

    stack = np.stack([bl, tl, br, tr], axis=-1)
    res = np.einsum("ij,xyj->xyi", m_quad, stack)
    return res  # shape (Nx, Ny, 6)


def test_quad_triangle_3_mugrid_vs_manual():
    """
    Compares wall-clock time and correctness of computing quadrature points on
    triangles using muGrid convolution vs. 'manual' tensor-matrix multiplication.
    """
    Nx, Ny = 1000, 1000
    nb_grid_pts = (Nx, Ny)

    # Run muGrid
    quad_mugrid, t_mugrid, device_str, init_field = run_quad_triangle_mugrid(
        nb_grid_pts, use_device=False
    )

    # Run manual
    m_quad = get_quad_triangle_manual_matrix()
    t_manual_start = time.perf_counter()
    quad_manual = quad_manual_combined(init_field, Nx, Ny, m_quad)
    t_manual = time.perf_counter() - t_manual_start

    quad_manual = quad_manual.transpose(2, 0, 1)  # -> shape (6, Nx, Ny)

    # Print timing
    print(f"muGrid operator time ({device_str}):  {t_mugrid:.6f} s")
    print(f"Manual operator time:  {t_manual:.6f} s")

    # Correctness check
    np.testing.assert_allclose(quad_mugrid, quad_manual, rtol=1e-10, atol=1e-12)

    # Compare wall-clock time
    # assert t_mugrid < 1.05 * t_manual, (
    #     f"muGrid slower than manual quadrature: "
    #     f"muGrid {t_mugrid:.6f}s vs. tensor-matrix mul {t_manual:.6f}s"
    # )


@unittest.skipUnless(GPU_AVAILABLE and HAS_CUPY, get_gpu_cupy_skip_reason() or "")
def test_quad_triangle_device_vs_host():
    """Compare quadrature triangle computation between CPU and GPU."""
    Nx, Ny = 1000, 1000
    nb_grid_pts = (Nx, Ny)

    # We need to use the same random seed for both runs
    np.random.seed(42)

    # Run on host
    host_result, t_host, host_device, init_field_host = run_quad_triangle_mugrid(
        nb_grid_pts, use_device=False
    )

    # Run on device with same input
    # Recreate the field collection and use same init_field
    if HAS_CUPY:
        memory_location = muGrid.GlobalFieldCollection.MemoryLocation.Device
    else:
        memory_location = muGrid.GlobalFieldCollection.MemoryLocation.Host

    fc = muGrid.GlobalFieldCollection(
        nb_grid_pts,
        sub_pts={"quad": 6},
        nb_ghosts_right=(1, 1),
        memory_location=memory_location,
    )
    nodal_field_cpp = fc.real_field("nodal")
    quad_field_cpp = fc.real_field("quad", 1, "quad")
    nodal_field = wrap_field(nodal_field_cpp)

    padded_field = np.pad(init_field_host, (0, 1), mode="wrap")
    nodal_field.pg[...] = cp.asarray(padded_field)

    kernel = get_quad_triangle_kernel()
    op_mugrid = muGrid.ConvolutionOperator([0, 0], kernel)
    device_str = nodal_field_cpp.device

    # Time device operation
    cp.cuda.Device().synchronize()
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()
    op_mugrid.apply(nodal_field_cpp, quad_field_cpp)
    end_event.record()
    end_event.synchronize()
    t_device = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0

    quad_field = wrap_field(quad_field_cpp)
    device_result = cp.asnumpy(quad_field.p)

    # Check results match
    np.testing.assert_allclose(
        device_result,
        host_result,
        rtol=1e-10,
        atol=1e-12,
        err_msg="Device and host quadrature results differ",
    )

    # Print comparison
    speedup = t_host / t_device if t_device > 0 else 0
    print(f"\nQuadrature triangle comparison ({Nx}x{Ny}):")
    print(f"  Host ({host_device}):   {t_host:.6f} s")
    print(f"  Device ({device_str}): {t_device:.6f} s")
    print(f"  Speedup: {speedup:.2f}x")


@unittest.skipUnless(GPU_AVAILABLE and HAS_CUPY, get_gpu_cupy_skip_reason() or "")
def test_quad_triangle_device_scaling():
    """Test quadrature triangle scaling on GPU across different grid sizes."""
    grid_sizes = [256, 512, 1000, 2000]

    print("\n=== Quadrature Triangle GPU Scaling ===")
    print(f"{'Grid':>12} {'Host (s)':>12} {'Device (s)':>12} {'Speedup':>10}")
    print("-" * 50)

    for size in grid_sizes:
        nb_grid_pts = (size, size)

        # Host timing
        _, t_host, _, _ = run_quad_triangle_mugrid(nb_grid_pts, use_device=False)

        # Device timing
        _, t_device, device_str, _ = run_quad_triangle_mugrid(
            nb_grid_pts, use_device=True
        )

        speedup = t_host / t_device if t_device > 0 else 0
        print(f"{size}x{size:>5} {t_host:>12.6f} {t_device:>12.6f} {speedup:>10.2f}x")

    print("-" * 50)
    print(f"Device: {device_str}")


@unittest.skipUnless(GPU_AVAILABLE, get_gpu_skip_reason() or "")
class DeviceConvolutionPerformanceTests(unittest.TestCase):
    """Performance test suite for device (GPU) convolution operator"""

    def test_device_performance_small_grid(self):
        """Benchmark convolution on a small 32x32 grid on GPU"""
        print("\n=== Testing 2D Device Convolution Performance (Small Grid) ===")

        # Parameters
        nb_x_pts = 32
        nb_y_pts = 32
        nb_stencil_x = 3
        nb_stencil_y = 3
        nb_operators = 2
        nb_quad_pts = 4
        nb_field_components = 3

        # Create stencil
        stencil = np.random.rand(nb_operators, nb_quad_pts, nb_stencil_x, nb_stencil_y)

        conv_op = muGrid.ConvolutionOperator([-1, -1], stencil)

        # Create device field collection
        nb_ghosts = (1, 1)
        fc = muGrid.GlobalFieldCollection(
            (nb_x_pts, nb_y_pts),
            sub_pts={"quad": nb_quad_pts},
            nb_ghosts_left=nb_ghosts,
            nb_ghosts_right=nb_ghosts,
            memory_location=muGrid.GlobalFieldCollection.MemoryLocation.Device,
        )

        # Create fields
        nodal_cpp = fc.real_field("nodal", nb_field_components)
        quad_cpp = fc.real_field("quad", (nb_field_components, nb_operators), "quad")

        # Initialize with random data (transfer to device via CuPy if available)
        if HAS_CUPY:
            nodal_arr = cp.from_dlpack(nodal_cpp)
            nodal_shape = nodal_arr.shape
            # Create random data with correct shape
            random_data = cp.random.rand(*nodal_shape).astype(cp.float64)
            nodal_arr[...] = random_data

        device_str = nodal_cpp.device

        # Measure performance
        metrics = measure_convolution_performance(
            conv_op,
            nodal_cpp,
            quad_cpp,
            grid_size=(nb_x_pts, nb_y_pts),
            stencil_size=(nb_stencil_x, nb_stencil_y),
            num_iterations=100,
            device=device_str,
        )

        metrics.print_report()

        # Sanity checks
        self.assertGreater(metrics.total_flops, 0)
        self.assertGreater(metrics.gflops, 0.0)

    def test_device_performance_large_grid(self):
        """Benchmark convolution on a large 256x256 grid on GPU"""
        print("\n=== Testing 2D Device Convolution Performance (Large Grid) ===")

        # Parameters
        nb_x_pts = 256
        nb_y_pts = 256
        nb_stencil_x = 3
        nb_stencil_y = 3
        nb_operators = 2
        nb_quad_pts = 4
        nb_field_components = 3

        # Create stencil
        stencil = np.random.rand(
            nb_operators, nb_quad_pts, 1, nb_stencil_x, nb_stencil_y
        )

        conv_op = muGrid.ConvolutionOperator([-1, -1], stencil)

        # Create device field collection
        nb_ghosts = (1, 1)
        fc = muGrid.GlobalFieldCollection(
            (nb_x_pts, nb_y_pts),
            sub_pts={"quad": nb_quad_pts},
            nb_ghosts_left=nb_ghosts,
            nb_ghosts_right=nb_ghosts,
            memory_location=muGrid.GlobalFieldCollection.MemoryLocation.Device,
        )

        # Create fields
        nodal_cpp = fc.real_field("nodal", nb_field_components)
        quad_cpp = fc.real_field("quad", (nb_field_components, nb_operators), "quad")

        # Initialize with random data
        if HAS_CUPY:
            nodal_arr = cp.from_dlpack(nodal_cpp)
            nodal_arr[...] = cp.random.rand(*nodal_arr.shape).astype(cp.float64)

        device_str = nodal_cpp.device

        # Measure performance (fewer iterations for large grid)
        metrics = measure_convolution_performance(
            conv_op,
            nodal_cpp,
            quad_cpp,
            grid_size=(nb_x_pts, nb_y_pts),
            stencil_size=(nb_stencil_x, nb_stencil_y),
            num_iterations=100,
            device=device_str,
        )

        metrics.print_report()

        # Sanity checks
        self.assertGreater(metrics.total_flops, 0)
        self.assertGreater(metrics.gflops, 0.0)


@unittest.skipUnless(GPU_AVAILABLE and HAS_CUPY, get_gpu_cupy_skip_reason() or "")
class HostDeviceComparisonTests(unittest.TestCase):
    """Compare performance between host and device convolution"""

    def test_host_vs_device_performance(self):
        """Compare convolution performance between CPU and GPU"""
        print("\n=== Host vs Device Performance Comparison ===")

        grid_sizes = [32, 64, 128, 256]
        host_results = []
        device_results = []

        for grid_size in grid_sizes:
            nb_stencil = 3
            nb_operators = 2
            nb_quad_pts = 4
            nb_components = 3
            nb_ghosts = (1, 1)

            # Create stencil
            stencil = np.random.rand(nb_operators, nb_quad_pts, nb_stencil, nb_stencil)
            conv_op = muGrid.ConvolutionOperator([-1, -1], stencil)

            # Host field collection
            fc_host = muGrid.GlobalFieldCollection(
                (grid_size, grid_size),
                sub_pts={"quad": nb_quad_pts},
                nb_ghosts_left=nb_ghosts,
                nb_ghosts_right=nb_ghosts,
            )
            nodal_host = fc_host.real_field("nodal", nb_components)
            quad_host = fc_host.real_field(
                "quad", (nb_components, nb_operators), "quad"
            )

            # Initialize host field
            nodal_arr = np.from_dlpack(nodal_host)
            nodal_arr[...] = np.random.rand(*nodal_arr.shape)

            # Measure host performance
            iterations = 100 if grid_size <= 64 else 50
            host_metrics = measure_convolution_performance(
                conv_op,
                nodal_host,
                quad_host,
                grid_size=(grid_size, grid_size),
                stencil_size=(nb_stencil, nb_stencil),
                num_iterations=iterations,
                device="cpu",
            )
            host_results.append(host_metrics)

            # Device field collection
            fc_device = muGrid.GlobalFieldCollection(
                (grid_size, grid_size),
                sub_pts={"quad": nb_quad_pts},
                nb_ghosts_left=nb_ghosts,
                nb_ghosts_right=nb_ghosts,
                memory_location=muGrid.GlobalFieldCollection.MemoryLocation.Device,
            )
            nodal_device = fc_device.real_field("nodal", nb_components)
            quad_device = fc_device.real_field(
                "quad", (nb_components, nb_operators), "quad"
            )

            # Initialize device field
            nodal_device_arr = cp.from_dlpack(nodal_device)
            nodal_device_arr[...] = cp.random.rand(*nodal_device_arr.shape).astype(
                cp.float64
            )

            device_str = nodal_device.device

            # Measure device performance
            device_metrics = measure_convolution_performance(
                conv_op,
                nodal_device,
                quad_device,
                grid_size=(grid_size, grid_size),
                stencil_size=(nb_stencil, nb_stencil),
                num_iterations=iterations,
                device=device_str,
            )
            device_results.append(device_metrics)

        # Print comparison table
        print("\n" + "=" * 80)
        print(f"{'Grid':>10} {'Host GFLOPS':>15} {'Device GFLOPS':>15} {'Speedup':>12}")
        print("-" * 80)

        for host_m, device_m in zip(host_results, device_results):
            grid_pixels = np.prod(host_m.grid_size)
            speedup = device_m.gflops / host_m.gflops if host_m.gflops > 0 else 0

            print(
                f"{grid_pixels:>10} {host_m.gflops:>15.3f} "
                f"{device_m.gflops:>15.3f} {speedup:>12.2f}x"
            )

        print("=" * 80 + "\n")

    def test_device_convolution_correctness(self):
        """Verify that device convolution produces same results as host"""
        print("\n=== Device Convolution Correctness Check ===")

        # Parameters
        nb_x_pts = 16
        nb_y_pts = 16
        nb_stencil = 3
        nb_operators = 2
        nb_quad_pts = 4
        nb_components = 3
        nb_ghosts = (1, 1)

        # Create stencil
        stencil = np.random.rand(nb_operators, nb_quad_pts, nb_stencil, nb_stencil)
        conv_op = muGrid.ConvolutionOperator([-1, -1], stencil)

        # Create host fields
        fc_host = muGrid.GlobalFieldCollection(
            (nb_x_pts, nb_y_pts),
            sub_pts={"quad": nb_quad_pts},
            nb_ghosts_left=nb_ghosts,
            nb_ghosts_right=nb_ghosts,
        )
        nodal_host = fc_host.real_field("nodal", nb_components)
        quad_host = fc_host.real_field("quad", (nb_components, nb_operators), "quad")

        # Initialize host nodal field with known values
        nodal_host_arr = np.from_dlpack(nodal_host)
        test_data = np.random.rand(*nodal_host_arr.shape)
        nodal_host_arr[...] = test_data

        # Apply on host
        conv_op.apply(nodal_host, quad_host)
        host_result = np.from_dlpack(quad_host).copy()

        # Create device fields
        fc_device = muGrid.GlobalFieldCollection(
            (nb_x_pts, nb_y_pts),
            sub_pts={"quad": nb_quad_pts},
            nb_ghosts_left=nb_ghosts,
            nb_ghosts_right=nb_ghosts,
            memory_location=muGrid.GlobalFieldCollection.MemoryLocation.Device,
        )
        nodal_device = fc_device.real_field("nodal", nb_components)
        quad_device = fc_device.real_field(
            "quad", (nb_components, nb_operators), "quad"
        )

        # Copy same data to device
        nodal_device_arr = cp.from_dlpack(nodal_device)
        nodal_device_arr[...] = cp.asarray(test_data)

        # Apply on device
        conv_op.apply(nodal_device, quad_device)

        # Get device result back to host
        quad_device_arr = cp.from_dlpack(quad_device)
        device_result = cp.asnumpy(quad_device_arr)

        # Compare results
        np.testing.assert_allclose(
            device_result,
            host_result,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Device convolution result differs from host result",
        )
        print("Device and host results match!")
