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
    operator_matrix = conv_op.pixel_operator
    nb_conv_pts = operator_matrix.shape[1] // nb_nodal_pts

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

    Returns
    -------
    PerformanceMetrics
        Container with all performance metrics
    """

    # Count theoretical FLOPs
    total_flops = count_theoretical_flops(conv_op, nodal_field, quad_field, grid_size)

    # Warm-up iteration
    conv_op.apply(nodal_field, quad_field)

    # Initialize PAPI if available and requested
    papi_values = {}
    if PAPI_AVAILABLE and use_papi:
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
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        conv_op.apply(nodal_field, quad_field)

    end_time = time.perf_counter()

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

    # Calculate average wall time
    wall_time_ms = ((end_time - start_time) / num_iterations) * 1000.0

    # Create metrics object
    metrics = PerformanceMetrics(
        wall_time_ms=wall_time_ms,
        total_flops=total_flops,
        grid_size=grid_size,
        stencil_size=stencil_size,
        nb_components=nodal_field.nb_components,
        nb_operators=conv_op.nb_operators,
        nb_quad_pts=conv_op.nb_quad_pts,
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

        conv_op = muGrid.ConvolutionOperator([0, 0], stencil)

        # Create field collection
        fc = muGrid.GlobalFieldCollection(
            (nb_x_pts, nb_y_pts), sub_pts={"quad": nb_quad_pts}
        )

        # Create fields
        nodal = fc.real_field("nodal", nb_field_components)
        quad = fc.real_field("quad", (nb_field_components, nb_operators), "quad")

        # Initialize with random data
        nodal.p = np.random.rand(*nodal.p.shape)

        # Measure performance
        metrics = measure_convolution_performance(
            conv_op,
            nodal,
            quad,
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
        stencil = np.random.rand(nb_operators, nb_quad_pts, nb_stencil_x, nb_stencil_y)

        conv_op = muGrid.ConvolutionOperator([0, 0], stencil)

        # Create field collection
        fc = muGrid.GlobalFieldCollection(
            (nb_x_pts, nb_y_pts), sub_pts={"quad": nb_quad_pts}
        )

        # Create fields
        nodal = fc.real_field("nodal", nb_field_components)
        quad = fc.real_field("quad", (nb_field_components, nb_operators), "quad")

        # Initialize with random data
        nodal.p = np.random.rand(*nodal.p.shape)

        # Measure performance (fewer iterations for large grid)
        metrics = measure_convolution_performance(
            conv_op,
            nodal,
            quad,
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

            conv_op = muGrid.ConvolutionOperator([0, 0], stencil)

            # Create field collection
            fc = muGrid.GlobalFieldCollection(
                (grid_size, grid_size), sub_pts={"quad": nb_quad_pts}
            )

            # Create fields
            nodal = fc.real_field("nodal", nb_components)
            quad = fc.real_field("quad", (nb_components, nb_operators), "quad")

            # Initialize with random data
            nodal.p = np.random.rand(*nodal.p.shape)

            # Measure performance
            iterations = 100 if grid_size <= 64 else 10
            metrics = measure_convolution_performance(
                conv_op,
                nodal,
                quad,
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


def test_laplace_mugrid_vs_scipy(nb_grid_pts=(128, 128)):
    comm = muGrid.Communicator()
    subdivisions = (1, 1)

    # For debug reporting
    def callback(it, x, r, p):
        """
        Callback function to print the current solution, residual, and search direction.
        """
        print(f"{it:5} {np.dot(r.ravel(), r.ravel()):.5}")

    # Setup problem
    decomposition = muGrid.CartesianDecomposition(
        comm, nb_grid_pts, subdivisions, (1, 1), (1, 1)
    )
    fc = decomposition.collection
    grid_spacing = 1 / np.array(nb_grid_pts)  # Grid spacing

    x, y = decomposition.coords  # Domain-local coords for each pixel
    i, j = decomposition.icoords

    rhs = fc.real_field("rhs")
    solution = fc.real_field("solution")

    rhs.p = (1 + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)) ** 10
    rhs.p -= np.mean(rhs.p)

    solution.s[...] = 0

    # muGrid solution
    stencil = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    )  # FD-stencil for the Laplacian
    laplace = muGrid.ConvolutionOperator([-1, -1], stencil)

    def hessp_mugrid(x, Ax):
        """
        Function to compute the product of the Hessian matrix with a vector.
        The Hessian is represented by the convolution operator.
        """
        decomposition.communicate_ghosts(x)
        laplace.apply(x, Ax)
        # We need the minus sign because the Laplace operator is negative
        # definite, but the conjugate-gradients solver assumes a
        # positive-definite operator.
        Ax.s /= -np.mean(grid_spacing) ** 2  # Scale by grid spacing
        return Ax

    conjugate_gradients(
        comm,
        fc,
        hessp_mugrid,  # linear operator
        rhs,
        solution,
        tol=1e-6,
        maxiter=1000,
        callback=callback,
    )
    mugrid_solution = solution.s.copy()

    # scipy Laplace sparse matrix
    nb = np.prod(nb_grid_pts)

    def grid_index(i, j):
        return (i % nb_grid_pts[0]).reshape(-1) * nb_grid_pts[1] + (
            j % nb_grid_pts[1]
        ).reshape(-1)

    laplace = (
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
    )

    laplace = laplace.tocsr()

    # scipy solution
    solution.s[...] = 0

    def hessp_scipy(x, Ax):
        """
        Function to compute the product of the Hessian matrix with a vector.
        The Hessian is represented by the convolution operator.
        """
        Ax.p = (laplace @ x.p.reshape(-1)).reshape(nb_grid_pts)
        # We need the minus sign because the Laplace operator is negative
        # definite, but the conjugate-gradients solver assumes a
        # positive-definite operator.
        Ax.s /= -np.mean(grid_spacing) ** 2  # Scale by grid spacing
        return Ax

    conjugate_gradients(
        comm,
        fc,
        hessp_scipy,  # linear operator
        rhs,
        solution,
        tol=1e-6,
        maxiter=1000,
        callback=callback,
    )

    # Check that both solutions agree
    np.testing.assert_allclose(solution.s, mugrid_solution)
