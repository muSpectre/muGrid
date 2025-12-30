/**
 * @file   test_linalg.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   30 Dec 2025
 *
 * @brief  Testing linear algebra operations, especially ghost subtraction
 *
 * These tests verify that linalg operations (norm_sq, vecdot, axpy_norm_sq)
 * correctly exclude ghost regions from their computations, which is essential
 * for MPI-parallel computations where ghost values are duplicated.
 *
 * The tests specifically cover the SoA (Structure of Arrays) memory layout
 * used on GPUs, which requires different memory access patterns than the
 * AoS (Array of Structures) layout used on CPUs.
 *
 * Copyright © 2025 Lars Pastewka
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "tests.hh"
#include "linalg/linalg.hh"
#include "mpi/communicator.hh"
#include "mpi/cartesian_decomposition.hh"
#include "field/field_typed.hh"
#include "memory/device.hh"

#include <cmath>

namespace muGrid {

BOOST_AUTO_TEST_SUITE(linalg_tests);

/* ---------------------------------------------------------------------- */
/* Test norm_sq on host with ghosts                                        */
/* ---------------------------------------------------------------------- */
BOOST_AUTO_TEST_CASE(norm_sq_host_with_ghosts_2d) {
    constexpr Index_t len{8};
    constexpr Index_t nb_components{2};

    Communicator comm{};
    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

    CartesianDecomposition decomp{comm, nb_domain_grid_pts, nb_subdivisions,
                                  nb_ghosts_left, nb_ghosts_right};

    auto & collection{decomp.get_collection()};
    auto & field{dynamic_cast<RealField &>(
        collection.real_field("test", nb_components))};

    // Fill the entire field (including ghosts) with 1.0
    auto * data = field.data();
    for (Index_t i = 0; i < field.get_buffer_size(); ++i) {
        data[i] = 1.0;
    }

    // Expected: only interior pixels should be counted
    // Interior = len * len pixels, each with nb_components values of 1.0
    // norm_sq = len * len * nb_components * 1.0^2
    const Real expected = static_cast<Real>(len * len * nb_components);

    Real result = linalg::norm_sq(field);

    BOOST_CHECK_CLOSE(result, expected, 1e-10);
}

/* ---------------------------------------------------------------------- */
/* Test vecdot on host with ghosts                                         */
/* ---------------------------------------------------------------------- */
BOOST_AUTO_TEST_CASE(vecdot_host_with_ghosts_2d) {
    constexpr Index_t len{8};
    constexpr Index_t nb_components{2};

    Communicator comm{};
    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

    CartesianDecomposition decomp{comm, nb_domain_grid_pts, nb_subdivisions,
                                  nb_ghosts_left, nb_ghosts_right};

    auto & collection{decomp.get_collection()};
    auto & field_a{dynamic_cast<RealField &>(
        collection.real_field("test_a", nb_components))};
    auto & field_b{dynamic_cast<RealField &>(
        collection.real_field("test_b", nb_components))};

    // Fill fields with 1.0 and 2.0
    auto * data_a = field_a.data();
    auto * data_b = field_b.data();
    for (Index_t i = 0; i < field_a.get_buffer_size(); ++i) {
        data_a[i] = 1.0;
        data_b[i] = 2.0;
    }

    // Expected: only interior pixels should be counted
    // vecdot = len * len * nb_components * (1.0 * 2.0)
    const Real expected = static_cast<Real>(len * len * nb_components * 2.0);

    Real result = linalg::vecdot(field_a, field_b);

    BOOST_CHECK_CLOSE(result, expected, 1e-10);
}

/* ---------------------------------------------------------------------- */
/* Test norm_sq with multi-component field (vector field)                  */
/* ---------------------------------------------------------------------- */
BOOST_AUTO_TEST_CASE(norm_sq_host_vector_field_2d) {
    constexpr Index_t len{8};
    constexpr Index_t nb_components{3};  // 3-component vector field

    Communicator comm{};
    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

    CartesianDecomposition decomp{comm, nb_domain_grid_pts, nb_subdivisions,
                                  nb_ghosts_left, nb_ghosts_right};

    auto & collection{decomp.get_collection()};
    auto & field{dynamic_cast<RealField &>(
        collection.real_field("vector_field", nb_components))};

    // Fill with different values per component to verify correct access
    auto * data = field.data();
    for (Index_t i = 0; i < field.get_buffer_size(); ++i) {
        data[i] = 1.0;
    }

    const Real expected = static_cast<Real>(len * len * nb_components);
    Real result = linalg::norm_sq(field);

    BOOST_CHECK_CLOSE(result, expected, 1e-10);
}

/* ---------------------------------------------------------------------- */
/* Test axpy_norm_sq on host with ghosts                                   */
/* ---------------------------------------------------------------------- */
BOOST_AUTO_TEST_CASE(axpy_norm_sq_host_with_ghosts_2d) {
    constexpr Index_t len{8};
    constexpr Index_t nb_components{2};
    const Real alpha{0.5};
    const Real fill_x{2.0};
    const Real fill_y{3.0};

    Communicator comm{};
    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

    CartesianDecomposition decomp{comm, nb_domain_grid_pts, nb_subdivisions,
                                  nb_ghosts_left, nb_ghosts_right};

    auto & collection{decomp.get_collection()};
    auto & field_x{dynamic_cast<RealField &>(
        collection.real_field("x", nb_components))};
    auto & field_y{dynamic_cast<RealField &>(
        collection.real_field("y", nb_components))};

    // Fill fields
    auto * data_x = field_x.data();
    auto * data_y = field_y.data();
    for (Index_t i = 0; i < field_x.get_buffer_size(); ++i) {
        data_x[i] = fill_x;
        data_y[i] = fill_y;
    }

    // y_new = alpha * x + y = 0.5 * 2 + 3 = 4
    // norm_sq(y_new) = len * len * nb_components * 4^2
    const Real y_new = alpha * fill_x + fill_y;
    const Real expected = static_cast<Real>(len * len * nb_components) *
                          y_new * y_new;

    Real result = linalg::axpy_norm_sq(alpha, field_x, field_y);

    BOOST_CHECK_CLOSE(result, expected, 1e-10);
}

/* ---------------------------------------------------------------------- */
/* GPU tests (only when GPU backend is enabled)                            */
/* ---------------------------------------------------------------------- */
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)

/**
 * Test norm_sq on GPU with ghosts - this specifically tests the SoA layout
 * ghost subtraction which was fixed in the GPU kernels.
 */
BOOST_AUTO_TEST_CASE(norm_sq_device_with_ghosts_2d) {
    constexpr Index_t len{8};
    constexpr Index_t nb_components{2};

    Communicator comm{};
    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

    // Create GPU decomposition
#if defined(MUGRID_ENABLE_CUDA)
    auto device = Device::cuda();
#elif defined(MUGRID_ENABLE_HIP)
    auto device = Device::rocm();
#endif

    CartesianDecomposition decomp{comm, nb_domain_grid_pts, nb_subdivisions,
                                  nb_ghosts_left, nb_ghosts_right, {}, device};

    auto & collection{decomp.get_collection()};
    auto & field{dynamic_cast<RealFieldDevice &>(
        collection.real_field("test", nb_components))};

    // Create a host field for initialization
    Communicator comm_host{};
    CartesianDecomposition decomp_host{comm_host, nb_domain_grid_pts,
                                       nb_subdivisions, nb_ghosts_left,
                                       nb_ghosts_right};
    auto & host_collection{decomp_host.get_collection()};
    auto & host_field{dynamic_cast<RealField &>(
        host_collection.real_field("host_test", nb_components))};

    // Fill host field with 1.0
    auto * host_data = host_field.data();
    for (Index_t i = 0; i < host_field.get_buffer_size(); ++i) {
        host_data[i] = 1.0;
    }

    // Copy to device
    field.deep_copy_from(host_field);

    // Expected: only interior pixels should be counted
    const Real expected = static_cast<Real>(len * len * nb_components);

    Real result = linalg::norm_sq(field);

    BOOST_CHECK_CLOSE(result, expected, 1e-10);
}

/**
 * Test vecdot on GPU with ghosts - tests SoA ghost subtraction in dot product.
 */
BOOST_AUTO_TEST_CASE(vecdot_device_with_ghosts_2d) {
    constexpr Index_t len{8};
    constexpr Index_t nb_components{2};

    Communicator comm{};
    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

#if defined(MUGRID_ENABLE_CUDA)
    auto device = Device::cuda();
#elif defined(MUGRID_ENABLE_HIP)
    auto device = Device::rocm();
#endif

    CartesianDecomposition decomp{comm, nb_domain_grid_pts, nb_subdivisions,
                                  nb_ghosts_left, nb_ghosts_right, {}, device};

    auto & collection{decomp.get_collection()};
    auto & field_a{dynamic_cast<RealFieldDevice &>(
        collection.real_field("test_a", nb_components))};
    auto & field_b{dynamic_cast<RealFieldDevice &>(
        collection.real_field("test_b", nb_components))};

    // Create host fields for initialization
    Communicator comm_host{};
    CartesianDecomposition decomp_host{comm_host, nb_domain_grid_pts,
                                       nb_subdivisions, nb_ghosts_left,
                                       nb_ghosts_right};
    auto & host_collection{decomp_host.get_collection()};
    auto & host_field_a{dynamic_cast<RealField &>(
        host_collection.real_field("host_a", nb_components))};
    auto & host_field_b{dynamic_cast<RealField &>(
        host_collection.real_field("host_b", nb_components))};

    // Fill host fields
    auto * data_a = host_field_a.data();
    auto * data_b = host_field_b.data();
    for (Index_t i = 0; i < host_field_a.get_buffer_size(); ++i) {
        data_a[i] = 1.0;
        data_b[i] = 2.0;
    }

    // Copy to device
    field_a.deep_copy_from(host_field_a);
    field_b.deep_copy_from(host_field_b);

    const Real expected = static_cast<Real>(len * len * nb_components * 2.0);
    Real result = linalg::vecdot(field_a, field_b);

    BOOST_CHECK_CLOSE(result, expected, 1e-10);
}

/**
 * Test norm_sq with multi-component field (vector field) on GPU.
 * GPU counterpart of norm_sq_host_vector_field_2d.
 */
BOOST_AUTO_TEST_CASE(norm_sq_device_vector_field_2d) {
    constexpr Index_t len{8};
    constexpr Index_t nb_components{3};  // 3-component vector field

    Communicator comm{};
    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

#if defined(MUGRID_ENABLE_CUDA)
    auto device = Device::cuda();
#elif defined(MUGRID_ENABLE_HIP)
    auto device = Device::rocm();
#endif

    CartesianDecomposition decomp{comm, nb_domain_grid_pts, nb_subdivisions,
                                  nb_ghosts_left, nb_ghosts_right, {}, device};

    auto & collection{decomp.get_collection()};
    auto & field{dynamic_cast<RealFieldDevice &>(
        collection.real_field("vector_field", nb_components))};

    // Create host field for initialization
    Communicator comm_host{};
    CartesianDecomposition decomp_host{comm_host, nb_domain_grid_pts,
                                       nb_subdivisions, nb_ghosts_left,
                                       nb_ghosts_right};
    auto & host_collection{decomp_host.get_collection()};
    auto & host_field{dynamic_cast<RealField &>(
        host_collection.real_field("host_vector_field", nb_components))};

    auto * host_data = host_field.data();
    for (Index_t i = 0; i < host_field.get_buffer_size(); ++i) {
        host_data[i] = 1.0;
    }

    field.deep_copy_from(host_field);

    const Real expected = static_cast<Real>(len * len * nb_components);
    Real result = linalg::norm_sq(field);

    BOOST_CHECK_CLOSE(result, expected, 1e-10);
}

/**
 * Test axpy_norm_sq on GPU only - GPU counterpart of axpy_norm_sq_host_with_ghosts_2d.
 */
BOOST_AUTO_TEST_CASE(axpy_norm_sq_device_with_ghosts_2d) {
    constexpr Index_t len{8};
    constexpr Index_t nb_components{2};
    const Real alpha{0.5};
    const Real fill_x{2.0};
    const Real fill_y{3.0};

    Communicator comm{};
    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

#if defined(MUGRID_ENABLE_CUDA)
    auto device = Device::cuda();
#elif defined(MUGRID_ENABLE_HIP)
    auto device = Device::rocm();
#endif

    CartesianDecomposition decomp{comm, nb_domain_grid_pts, nb_subdivisions,
                                  nb_ghosts_left, nb_ghosts_right, {}, device};

    auto & collection{decomp.get_collection()};
    auto & field_x{dynamic_cast<RealFieldDevice &>(
        collection.real_field("x", nb_components))};
    auto & field_y{dynamic_cast<RealFieldDevice &>(
        collection.real_field("y", nb_components))};

    // Create host fields for initialization
    Communicator comm_host{};
    CartesianDecomposition decomp_host{comm_host, nb_domain_grid_pts,
                                       nb_subdivisions, nb_ghosts_left,
                                       nb_ghosts_right};
    auto & host_collection{decomp_host.get_collection()};
    auto & host_x{dynamic_cast<RealField &>(
        host_collection.real_field("host_x", nb_components))};
    auto & host_y{dynamic_cast<RealField &>(
        host_collection.real_field("host_y", nb_components))};

    auto * data_x = host_x.data();
    auto * data_y = host_y.data();
    for (Index_t i = 0; i < host_x.get_buffer_size(); ++i) {
        data_x[i] = fill_x;
        data_y[i] = fill_y;
    }

    field_x.deep_copy_from(host_x);
    field_y.deep_copy_from(host_y);

    const Real y_new = alpha * fill_x + fill_y;
    const Real expected = static_cast<Real>(len * len * nb_components) *
                          y_new * y_new;

    Real result = linalg::axpy_norm_sq(alpha, field_x, field_y);

    BOOST_CHECK_CLOSE(result, expected, 1e-10);
}

/* ---------------------------------------------------------------------- */
/* Deep copy layout conversion tests                                       */
/* ---------------------------------------------------------------------- */

/**
 * Test deep_copy from CPU (AoS) to GPU (SoA) with layout conversion.
 *
 * deep_copy now performs automatic layout conversion between AoS (CPU) and
 * SoA (GPU) formats, so logical field values are preserved across the copy.
 */
BOOST_AUTO_TEST_CASE(deep_copy_cpu_to_gpu_constant_values) {
    constexpr Index_t len{8};
    constexpr Index_t nb_components{3};
    const Real fill_value{2.5};

    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

    // Create CPU decomposition (AoS layout)
    Communicator comm_cpu{};
    CartesianDecomposition decomp_cpu{comm_cpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right};
    auto & cpu_collection{decomp_cpu.get_collection()};
    auto & cpu_field{dynamic_cast<RealField &>(
        cpu_collection.real_field("cpu_field", nb_components))};

    // Create GPU decomposition (SoA layout)
#if defined(MUGRID_ENABLE_CUDA)
    auto device = Device::cuda();
#elif defined(MUGRID_ENABLE_HIP)
    auto device = Device::rocm();
#endif
    Communicator comm_gpu{};
    CartesianDecomposition decomp_gpu{comm_gpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right, {}, device};
    auto & gpu_collection{decomp_gpu.get_collection()};
    auto & gpu_field{dynamic_cast<RealFieldDevice &>(
        gpu_collection.real_field("gpu_field", nb_components))};

    // Fill CPU field with constant value (works for both layouts)
    auto * cpu_data = cpu_field.data();
    for (Index_t i = 0; i < cpu_field.get_buffer_size(); ++i) {
        cpu_data[i] = fill_value;
    }

    // Deep copy to GPU
    gpu_field.deep_copy_from(cpu_field);

    // Both should produce same norm_sq since all values are identical
    Real cpu_norm = linalg::norm_sq(cpu_field);
    Real gpu_norm = linalg::norm_sq(gpu_field);

    BOOST_CHECK_CLOSE(cpu_norm, gpu_norm, 1e-10);

    // Verify expected value
    const Real expected = static_cast<Real>(len * len * nb_components) *
                          fill_value * fill_value;
    BOOST_CHECK_CLOSE(cpu_norm, expected, 1e-10);
}

/**
 * Test round-trip deep_copy: CPU -> GPU -> CPU preserves values.
 *
 * This test uses non-constant values to verify that layout conversion
 * correctly transforms between AoS (CPU) and SoA (GPU) formats and back.
 */
BOOST_AUTO_TEST_CASE(deep_copy_round_trip_cpu_gpu_cpu) {
    constexpr Index_t len{8};
    constexpr Index_t nb_components{2};

    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

    // Create CPU decomposition
    Communicator comm_cpu{};
    CartesianDecomposition decomp_cpu{comm_cpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right};
    auto & cpu_collection{decomp_cpu.get_collection()};
    auto & cpu_field_src{dynamic_cast<RealField &>(
        cpu_collection.real_field("cpu_src", nb_components))};
    auto & cpu_field_dst{dynamic_cast<RealField &>(
        cpu_collection.real_field("cpu_dst", nb_components))};

    // Create GPU decomposition
#if defined(MUGRID_ENABLE_CUDA)
    auto device = Device::cuda();
#elif defined(MUGRID_ENABLE_HIP)
    auto device = Device::rocm();
#endif
    Communicator comm_gpu{};
    CartesianDecomposition decomp_gpu{comm_gpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right, {}, device};
    auto & gpu_collection{decomp_gpu.get_collection()};
    auto & gpu_field{dynamic_cast<RealFieldDevice &>(
        gpu_collection.real_field("gpu_field", nb_components))};

    // Fill source CPU field with pattern
    auto * src_data = cpu_field_src.data();
    for (Index_t i = 0; i < cpu_field_src.get_buffer_size(); ++i) {
        src_data[i] = static_cast<Real>(i % 13 + 1);
    }

    // Round trip: CPU -> GPU -> CPU
    gpu_field.deep_copy_from(cpu_field_src);
    cpu_field_dst.deep_copy_from(gpu_field);

    // Verify all values match
    auto * dst_data = cpu_field_dst.data();
    bool all_match = true;
    for (Index_t i = 0; i < cpu_field_src.get_buffer_size(); ++i) {
        if (std::abs(src_data[i] - dst_data[i]) > 1e-14) {
            all_match = false;
            break;
        }
    }
    BOOST_CHECK(all_match);

    // Also check via norm_sq
    Real src_norm = linalg::norm_sq(cpu_field_src);
    Real dst_norm = linalg::norm_sq(cpu_field_dst);
    BOOST_CHECK_CLOSE(src_norm, dst_norm, 1e-10);
}

/**
 * Test deep_copy with non-constant values and verify via norm_sq.
 *
 * This test fills the CPU field with a pattern, copies to GPU (with automatic
 * AoS->SoA layout conversion), and verifies that norm_sq produces the same
 * result on both devices.
 */
BOOST_AUTO_TEST_CASE(deep_copy_cpu_to_gpu_varying_values) {
    constexpr Index_t len{8};
    constexpr Index_t nb_components{3};

    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

    // Create CPU decomposition (AoS layout)
    Communicator comm_cpu{};
    CartesianDecomposition decomp_cpu{comm_cpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right};
    auto & cpu_collection{decomp_cpu.get_collection()};
    auto & cpu_field{dynamic_cast<RealField &>(
        cpu_collection.real_field("cpu_field", nb_components))};

    // Create GPU decomposition (SoA layout)
#if defined(MUGRID_ENABLE_CUDA)
    auto device = Device::cuda();
#elif defined(MUGRID_ENABLE_HIP)
    auto device = Device::rocm();
#endif
    Communicator comm_gpu{};
    CartesianDecomposition decomp_gpu{comm_gpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right, {}, device};
    auto & gpu_collection{decomp_gpu.get_collection()};
    auto & gpu_field{dynamic_cast<RealFieldDevice &>(
        gpu_collection.real_field("gpu_field", nb_components))};

    // Fill CPU field with varying values
    auto * cpu_data = cpu_field.data();
    for (Index_t i = 0; i < cpu_field.get_buffer_size(); ++i) {
        cpu_data[i] = static_cast<Real>((i % 17) + 1) * 0.5;
    }

    // Compute expected norm_sq from CPU field
    Real cpu_norm = linalg::norm_sq(cpu_field);

    // Deep copy to GPU (with layout conversion)
    gpu_field.deep_copy_from(cpu_field);

    // Compute norm_sq on GPU - should match CPU if layout conversion worked
    Real gpu_norm = linalg::norm_sq(gpu_field);

    BOOST_CHECK_CLOSE(cpu_norm, gpu_norm, 1e-10);
}

/**
 * Test that CPU and GPU produce identical results for norm_sq with ghosts.
 * This is a critical regression test for the SoA memory layout fix.
 */
BOOST_AUTO_TEST_CASE(norm_sq_cpu_gpu_match_2d) {
    constexpr Index_t len{16};
    constexpr Index_t nb_components{2};
    const Real fill_value{3.5};

    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

    // Create CPU decomposition
    Communicator comm_cpu{};
    CartesianDecomposition decomp_cpu{comm_cpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right};
    auto & cpu_collection{decomp_cpu.get_collection()};
    auto & cpu_field{dynamic_cast<RealField &>(
        cpu_collection.real_field("cpu_field", nb_components))};

    // Create GPU decomposition
#if defined(MUGRID_ENABLE_CUDA)
    auto device = Device::cuda();
#elif defined(MUGRID_ENABLE_HIP)
    auto device = Device::rocm();
#endif
    Communicator comm_gpu{};
    CartesianDecomposition decomp_gpu{comm_gpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right, {}, device};
    auto & gpu_collection{decomp_gpu.get_collection()};
    auto & gpu_field{dynamic_cast<RealFieldDevice &>(
        gpu_collection.real_field("gpu_field", nb_components))};

    // Fill CPU field with constant value (works correctly for both layouts)
    auto * cpu_data = cpu_field.data();
    for (Index_t i = 0; i < cpu_field.get_buffer_size(); ++i) {
        cpu_data[i] = fill_value;
    }

    // Copy to GPU (deep_copy handles layout conversion)
    gpu_field.deep_copy_from(cpu_field);

    // Compute norm_sq on both
    Real cpu_result = linalg::norm_sq(cpu_field);
    Real gpu_result = linalg::norm_sq(gpu_field);

    // Results should match exactly
    BOOST_CHECK_CLOSE(cpu_result, gpu_result, 1e-10);

    // Also verify expected value
    const Real expected = static_cast<Real>(len * len * nb_components) *
                          fill_value * fill_value;
    BOOST_CHECK_CLOSE(cpu_result, expected, 1e-10);
}

/**
 * Test that CPU and GPU produce identical results for vecdot with ghosts.
 */
BOOST_AUTO_TEST_CASE(vecdot_cpu_gpu_match_2d) {
    constexpr Index_t len{16};
    constexpr Index_t nb_components{3};
    const Real fill_a{2.5};
    const Real fill_b{1.5};

    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

    // Create CPU decomposition
    Communicator comm_cpu{};
    CartesianDecomposition decomp_cpu{comm_cpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right};
    auto & cpu_collection{decomp_cpu.get_collection()};
    auto & cpu_field_a{dynamic_cast<RealField &>(
        cpu_collection.real_field("cpu_a", nb_components))};
    auto & cpu_field_b{dynamic_cast<RealField &>(
        cpu_collection.real_field("cpu_b", nb_components))};

    // Create GPU decomposition
#if defined(MUGRID_ENABLE_CUDA)
    auto device = Device::cuda();
#elif defined(MUGRID_ENABLE_HIP)
    auto device = Device::rocm();
#endif
    Communicator comm_gpu{};
    CartesianDecomposition decomp_gpu{comm_gpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right, {}, device};
    auto & gpu_collection{decomp_gpu.get_collection()};
    auto & gpu_field_a{dynamic_cast<RealFieldDevice &>(
        gpu_collection.real_field("gpu_a", nb_components))};
    auto & gpu_field_b{dynamic_cast<RealFieldDevice &>(
        gpu_collection.real_field("gpu_b", nb_components))};

    // Fill CPU fields with constant values
    auto * data_a = cpu_field_a.data();
    auto * data_b = cpu_field_b.data();
    for (Index_t i = 0; i < cpu_field_a.get_buffer_size(); ++i) {
        data_a[i] = fill_a;
        data_b[i] = fill_b;
    }

    // Copy to GPU
    gpu_field_a.deep_copy_from(cpu_field_a);
    gpu_field_b.deep_copy_from(cpu_field_b);

    // Compute vecdot on both
    Real cpu_result = linalg::vecdot(cpu_field_a, cpu_field_b);
    Real gpu_result = linalg::vecdot(gpu_field_a, gpu_field_b);

    BOOST_CHECK_CLOSE(cpu_result, gpu_result, 1e-10);

    // Also verify expected value
    const Real expected = static_cast<Real>(len * len * nb_components) *
                          fill_a * fill_b;
    BOOST_CHECK_CLOSE(cpu_result, expected, 1e-10);
}

/**
 * Test axpy_norm_sq on GPU - fused operation that was also affected.
 */
BOOST_AUTO_TEST_CASE(axpy_norm_sq_cpu_gpu_match_2d) {
    constexpr Index_t len{16};
    constexpr Index_t nb_components{2};
    const Real alpha{0.5};
    const Real fill_x{2.0};
    const Real fill_y{3.0};

    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 1};
    DynGridIndex nb_ghosts_right{1, 1};

    // Create CPU decomposition
    Communicator comm_cpu{};
    CartesianDecomposition decomp_cpu{comm_cpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right};
    auto & cpu_collection{decomp_cpu.get_collection()};
    auto & cpu_x{dynamic_cast<RealField &>(
        cpu_collection.real_field("cpu_x", nb_components))};
    auto & cpu_y{dynamic_cast<RealField &>(
        cpu_collection.real_field("cpu_y", nb_components))};

    // Create GPU decomposition
#if defined(MUGRID_ENABLE_CUDA)
    auto device = Device::cuda();
#elif defined(MUGRID_ENABLE_HIP)
    auto device = Device::rocm();
#endif
    Communicator comm_gpu{};
    CartesianDecomposition decomp_gpu{comm_gpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right, {}, device};
    auto & gpu_collection{decomp_gpu.get_collection()};
    auto & gpu_x{dynamic_cast<RealFieldDevice &>(
        gpu_collection.real_field("gpu_x", nb_components))};
    auto & gpu_y{dynamic_cast<RealFieldDevice &>(
        gpu_collection.real_field("gpu_y", nb_components))};

    // Fill CPU fields with constant values
    auto * x_data = cpu_x.data();
    auto * y_data = cpu_y.data();
    for (Index_t i = 0; i < cpu_x.get_buffer_size(); ++i) {
        x_data[i] = fill_x;
        y_data[i] = fill_y;
    }

    // Copy to GPU
    gpu_x.deep_copy_from(cpu_x);
    gpu_y.deep_copy_from(cpu_y);

    // Perform axpy_norm_sq on both
    Real cpu_result = linalg::axpy_norm_sq(alpha, cpu_x, cpu_y);
    Real gpu_result = linalg::axpy_norm_sq(alpha, gpu_x, gpu_y);

    BOOST_CHECK_CLOSE(cpu_result, gpu_result, 1e-10);

    // Also verify expected value: y_new = alpha * x + y = 0.5 * 2 + 3 = 4
    // norm_sq(y_new) = len * len * nb_components * 4^2 = 256 * 2 * 16 = 8192
    const Real y_new = alpha * fill_x + fill_y;
    const Real expected = static_cast<Real>(len * len * nb_components) *
                          y_new * y_new;
    BOOST_CHECK_CLOSE(cpu_result, expected, 1e-10);
}

/**
 * Test with asymmetric ghost sizes to verify edge cases.
 */
BOOST_AUTO_TEST_CASE(norm_sq_asymmetric_ghosts_2d) {
    constexpr Index_t len{10};
    constexpr Index_t nb_components{2};

    DynGridIndex nb_domain_grid_pts{len, len};
    DynGridIndex nb_subdivisions{1, 1};
    DynGridIndex nb_ghosts_left{1, 2};   // Asymmetric
    DynGridIndex nb_ghosts_right{2, 1};  // Asymmetric

    // Create CPU decomposition
    Communicator comm_cpu{};
    CartesianDecomposition decomp_cpu{comm_cpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right};
    auto & cpu_collection{decomp_cpu.get_collection()};
    auto & cpu_field{dynamic_cast<RealField &>(
        cpu_collection.real_field("cpu_field", nb_components))};

    // Create GPU decomposition
#if defined(MUGRID_ENABLE_CUDA)
    auto device = Device::cuda();
#elif defined(MUGRID_ENABLE_HIP)
    auto device = Device::rocm();
#endif
    Communicator comm_gpu{};
    CartesianDecomposition decomp_gpu{comm_gpu, nb_domain_grid_pts,
                                      nb_subdivisions, nb_ghosts_left,
                                      nb_ghosts_right, {}, device};
    auto & gpu_collection{decomp_gpu.get_collection()};
    auto & gpu_field{dynamic_cast<RealFieldDevice &>(
        gpu_collection.real_field("gpu_field", nb_components))};

    // Fill CPU field
    auto * cpu_data = cpu_field.data();
    for (Index_t i = 0; i < cpu_field.get_buffer_size(); ++i) {
        cpu_data[i] = 1.0;
    }

    // Copy to GPU
    gpu_field.deep_copy_from(cpu_field);

    // Both should give the same result
    Real cpu_result = linalg::norm_sq(cpu_field);
    Real gpu_result = linalg::norm_sq(gpu_field);

    BOOST_CHECK_CLOSE(cpu_result, gpu_result, 1e-10);

    // And should equal the expected value
    const Real expected = static_cast<Real>(len * len * nb_components);
    BOOST_CHECK_CLOSE(cpu_result, expected, 1e-10);
}

#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP

BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
