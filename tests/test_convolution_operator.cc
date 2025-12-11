/**
 * @file   test_convolution_operator.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   09 Dec 2024
 *
 * @brief  Tests for the ConvolutionOperator with GPU support
 *
 * Copyright © 2024 Lars Pastewka
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
#include "test_goodies.hh"

#include "libmugrid/convolution_operator.hh"
#include "libmugrid/field_collection_global.hh"
#include "libmugrid/field_typed.hh"
#include "libmugrid/field_map.hh"
#include "libmugrid/kokkos_types.hh"

#include <Kokkos_Core.hpp>
#include <boost/mpl/list.hpp>

#include <random>
#include <cmath>

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  /* Test fixtures for 2D and 3D grids                                      */
  /* ---------------------------------------------------------------------- */

  template <Index_t Dim_>
  struct ConvolutionFixtureBase {
    static constexpr Index_t Dim = Dim_;
    static constexpr Real del_x = 1.0;
    static constexpr Real del_y = 1.0;
    static constexpr Real del_z = 1.0;

    // Simple bilinear (2D) or trilinear (3D) operator
    // 1 nodal point per pixel, 1 quad point per pixel, 1 operator
    static constexpr Index_t NbNodalPts = 1;
    static constexpr Index_t NbQuadPts = 1;
    static constexpr Index_t NbOperators = Dim;  // gradient has Dim components

    static Shape_t get_pixel_offset() {
      Shape_t offset(Dim, 0);
      return offset;
    }

    static Shape_t get_conv_pts_shape() {
      Shape_t shape(Dim, 2);  // 2^Dim stencil points
      return shape;
    }

    static Eigen::MatrixXd get_pixel_operator() {
      // Simple finite difference gradient operator
      // rows = NbOperators * NbQuadPts = Dim
      // cols = NbNodalPts * 2^Dim
      const Index_t nb_stencil_pts = static_cast<Index_t>(std::pow(2, Dim));
      Eigen::MatrixXd op = Eigen::MatrixXd::Zero(Dim, nb_stencil_pts);

      if (Dim == 2) {
        // 2D gradient using bilinear interpolation
        // Stencil: (0,0), (1,0), (0,1), (1,1)
        // d/dx: [-1, 1, -1, 1] / 2
        // d/dy: [-1, -1, 1, 1] / 2
        op(0, 0) = -0.5; op(0, 1) = 0.5; op(0, 2) = -0.5; op(0, 3) = 0.5;
        op(1, 0) = -0.5; op(1, 1) = -0.5; op(1, 2) = 0.5; op(1, 3) = 0.5;
      } else if (Dim == 3) {
        // 3D gradient using trilinear interpolation
        // Stencil: (0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)
        // d/dx: [-1, 1, -1, 1, -1, 1, -1, 1] / 4
        // d/dy: [-1, -1, 1, 1, -1, -1, 1, 1] / 4
        // d/dz: [-1, -1, -1, -1, 1, 1, 1, 1] / 4
        for (Index_t i = 0; i < 8; ++i) {
          op(0, i) = ((i & 1) ? 0.25 : -0.25);       // x-derivative
          op(1, i) = ((i & 2) ? 0.25 : -0.25);       // y-derivative
          op(2, i) = ((i & 4) ? 0.25 : -0.25);       // z-derivative
        }
      }
      return op;
    }
  };

  template <Index_t Dim_>
  constexpr Index_t ConvolutionFixtureBase<Dim_>::Dim;

  /* ---------------------------------------------------------------------- */
  struct Fixture2D : public ConvolutionFixtureBase<twoD> {
    static constexpr Index_t grid_size = 8;
    IntCoord_t nb_grid_pts{grid_size, grid_size};
    IntCoord_t nb_subdomain_grid_pts{grid_size + 1, grid_size + 1};  // +1 for ghost
    IntCoord_t subdomain_locations{0, 0};
    IntCoord_t nb_ghosts_left{0, 0};
    IntCoord_t nb_ghosts_right{1, 1};

    ConvolutionOperator op{
        get_pixel_offset(),
        get_pixel_operator(),
        get_conv_pts_shape(),
        NbNodalPts,
        NbQuadPts,
        NbOperators};

    GlobalFieldCollection collection;

    Fixture2D()
        : collection(nb_grid_pts, nb_subdomain_grid_pts, subdomain_locations,
                     {{PixelTag, NbNodalPts}, {"quad", NbQuadPts}},
                     StorageOrder::ArrayOfStructures,
                     nb_ghosts_left, nb_ghosts_right) {
    }
  };

  /* ---------------------------------------------------------------------- */
  struct Fixture3D : public ConvolutionFixtureBase<threeD> {
    static constexpr Index_t grid_size = 8;
    IntCoord_t nb_grid_pts{grid_size, grid_size, grid_size};
    IntCoord_t nb_subdomain_grid_pts{grid_size + 1, grid_size + 1, grid_size + 1};
    IntCoord_t subdomain_locations{0, 0, 0};
    IntCoord_t nb_ghosts_left{0, 0, 0};
    IntCoord_t nb_ghosts_right{1, 1, 1};

    ConvolutionOperator op{
        get_pixel_offset(),
        get_pixel_operator(),
        get_conv_pts_shape(),
        NbNodalPts,
        NbQuadPts,
        NbOperators};

    GlobalFieldCollection collection;

    Fixture3D()
        : collection(nb_grid_pts, nb_subdomain_grid_pts, subdomain_locations,
                     {{PixelTag, NbNodalPts}, {"quad", NbQuadPts}},
                     StorageOrder::ArrayOfStructures,
                     nb_ghosts_left, nb_ghosts_right) {
    }
  };

  /* ---------------------------------------------------------------------- */
  /* Larger grids for performance testing                                   */
  /* ---------------------------------------------------------------------- */
  struct FixtureLarge2D : public ConvolutionFixtureBase<twoD> {
    static constexpr Index_t grid_size = 128;
    IntCoord_t nb_grid_pts{grid_size, grid_size};
    IntCoord_t nb_subdomain_grid_pts{grid_size + 1, grid_size + 1};
    IntCoord_t subdomain_locations{0, 0};
    IntCoord_t nb_ghosts_left{0, 0};
    IntCoord_t nb_ghosts_right{1, 1};

    ConvolutionOperator op{
        get_pixel_offset(),
        get_pixel_operator(),
        get_conv_pts_shape(),
        NbNodalPts,
        NbQuadPts,
        NbOperators};

    GlobalFieldCollection collection;

    FixtureLarge2D()
        : collection(nb_grid_pts, nb_subdomain_grid_pts, subdomain_locations,
                     {{PixelTag, NbNodalPts}, {"quad", NbQuadPts}},
                     StorageOrder::ArrayOfStructures,
                     nb_ghosts_left, nb_ghosts_right) {
    }
  };

  struct FixtureLarge3D : public ConvolutionFixtureBase<threeD> {
    static constexpr Index_t grid_size = 32;
    IntCoord_t nb_grid_pts{grid_size, grid_size, grid_size};
    IntCoord_t nb_subdomain_grid_pts{grid_size + 1, grid_size + 1, grid_size + 1};
    IntCoord_t subdomain_locations{0, 0, 0};
    IntCoord_t nb_ghosts_left{0, 0, 0};
    IntCoord_t nb_ghosts_right{1, 1, 1};

    ConvolutionOperator op{
        get_pixel_offset(),
        get_pixel_operator(),
        get_conv_pts_shape(),
        NbNodalPts,
        NbQuadPts,
        NbOperators};

    GlobalFieldCollection collection;

    FixtureLarge3D()
        : collection(nb_grid_pts, nb_subdomain_grid_pts, subdomain_locations,
                     {{PixelTag, NbNodalPts}, {"quad", NbQuadPts}},
                     StorageOrder::ArrayOfStructures,
                     nb_ghosts_left, nb_ghosts_right) {
    }
  };

  using ConvolutionFixtures = boost::mpl::list<Fixture2D, Fixture3D>;
  using LargeConvolutionFixtures = boost::mpl::list<FixtureLarge2D, FixtureLarge3D>;

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_SUITE(convolution_operator);

  /* ---------------------------------------------------------------------- */
  /* Basic construction and accessor tests                                   */
  /* ---------------------------------------------------------------------- */

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, Fix, ConvolutionFixtures,
                                   Fix) {
    BOOST_CHECK_EQUAL(Fix::op.get_spatial_dim(), Fix::Dim);
    BOOST_CHECK_EQUAL(Fix::op.get_nb_quad_pts(), Fix::NbQuadPts);
    BOOST_CHECK_EQUAL(Fix::op.get_nb_nodal_pts(), Fix::NbNodalPts);
    BOOST_CHECK_EQUAL(Fix::op.get_nb_operators(), Fix::NbOperators);

    auto & pixel_op = Fix::op.get_pixel_operator();
    BOOST_CHECK_EQUAL(pixel_op.rows(), Fix::NbOperators * Fix::NbQuadPts);
    const Index_t expected_cols = Fix::NbNodalPts *
                                  static_cast<Index_t>(std::pow(2, Fix::Dim));
    BOOST_CHECK_EQUAL(pixel_op.cols(), expected_cols);
  }

  /* ---------------------------------------------------------------------- */
  /* Test apply operation correctness                                        */
  /* ---------------------------------------------------------------------- */

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(apply_constant_field, Fix,
                                   ConvolutionFixtures, Fix) {
    // For a constant field, the gradient should be zero
    auto & nodal = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal", 1, PixelTag));
    auto & quad = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad", Fix::Dim, "quad"));

    // Set constant value everywhere (including ghosts)
    nodal.eigen_vec().setConstant(42.0);

    // Apply gradient
    Fix::op.apply(nodal, quad);

    // Check gradient is zero (within tolerance)
    Real max_error = quad.eigen_vec().cwiseAbs().maxCoeff();
    BOOST_CHECK_LE(max_error, tol);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(apply_linear_field, Fix, ConvolutionFixtures,
                                   Fix) {
    // For a linear field, the gradient should be constant
    auto & nodal = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_linear", 1, PixelTag));
    auto & quad = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_linear", Fix::Dim, "quad"));

    // Fill with linear function: f(x,y) = ax + by (+ cz for 3D)
    const Real a = 1.5, b = 2.3, c = 3.1;

    auto nodal_map = nodal.get_pixel_map();
    for (auto && [id, ccoord] : Fix::collection.get_pixels_with_ghosts().enumerate()) {
      Real val = a * ccoord[0] + b * ccoord[1];
      if (Fix::Dim == 3) {
        val += c * ccoord[2];
      }
      nodal_map[id](0, 0) = val;
    }

    // Apply gradient
    Fix::op.apply(nodal, quad);

    // Check gradient is [a, b] (or [a, b, c] for 3D) at interior points
    auto quad_map = quad.get_pixel_map();

    for (auto && [id, ccoord] : Fix::collection.get_pixels_without_ghosts().enumerate()) {
      auto grad = quad_map[id];
      Real err_x = std::abs(grad(0, 0) - a);
      Real err_y = std::abs(grad(1, 0) - b);
      BOOST_CHECK_LE(err_x, tol);
      BOOST_CHECK_LE(err_y, tol);
      if (Fix::Dim == 3) {
        Real err_z = std::abs(grad(2, 0) - c);
        BOOST_CHECK_LE(err_z, tol);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Test transpose operator (adjoint property)                              */
  /* ---------------------------------------------------------------------- */

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(transpose_adjoint_property, Fix,
                                   ConvolutionFixtures, Fix) {
    // Test that <Bu, v> = <u, B^T v> for random u, v
    auto & u = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("u", 1, PixelTag));
    auto & v = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("v", Fix::Dim, "quad"));
    auto & Bu = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("Bu", Fix::Dim, "quad"));
    auto & BTv = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("BTv", 1, PixelTag));

    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);

    for (Index_t i = 0; i < u.eigen_vec().size(); ++i) {
      u.eigen_vec()(i) = dist(gen);
    }
    for (Index_t i = 0; i < v.eigen_vec().size(); ++i) {
      v.eigen_vec()(i) = dist(gen);
    }

    // Compute Bu and B^T v
    Fix::op.apply(u, Bu);
    Fix::op.transpose(v, BTv);

    // Check <Bu, v> = <u, B^T v>
    Real Bu_dot_v = Bu.eigen_vec().dot(v.eigen_vec());
    Real u_dot_BTv = u.eigen_vec().dot(BTv.eigen_vec());

    Real error = testGoodies::rel_error(Bu_dot_v, u_dot_BTv);
    BOOST_CHECK_LE(error, tol);
  }

  /* ---------------------------------------------------------------------- */
  /* Test apply_increment and transpose_increment                            */
  /* ---------------------------------------------------------------------- */

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(apply_increment_test, Fix,
                                   ConvolutionFixtures, Fix) {
    auto & nodal = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_inc", 1, PixelTag));
    auto & quad = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_inc", Fix::Dim, "quad"));
    auto & quad_ref = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_ref", Fix::Dim, "quad"));

    // Set random values
    std::mt19937 gen(123);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    for (Index_t i = 0; i < nodal.eigen_vec().size(); ++i) {
      nodal.eigen_vec()(i) = dist(gen);
    }

    // Compute reference: quad_ref = 2.5 * B * nodal
    quad_ref.set_zero();
    Fix::op.apply(nodal, quad_ref);
    quad_ref.eigen_vec() *= 2.5;

    // Compute using apply_increment: quad = 0 + 2.5 * B * nodal
    quad.set_zero();
    Fix::op.apply_increment(nodal, 2.5, quad);

    // Check they match
    Real error = testGoodies::rel_error(quad.eigen_vec(), quad_ref.eigen_vec());
    BOOST_CHECK_LE(error, tol);

    // Test accumulation: quad += 1.0 * B * nodal
    Fix::op.apply_increment(nodal, 1.0, quad);

    // Should now be 3.5 * B * nodal
    quad_ref.eigen_vec() *= (3.5 / 2.5);

    error = testGoodies::rel_error(quad.eigen_vec(), quad_ref.eigen_vec());
    BOOST_CHECK_LE(error, tol);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(transpose_increment_test, Fix,
                                   ConvolutionFixtures, Fix) {
    auto & nodal = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_tinc", 1, PixelTag));
    auto & nodal_ref = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_ref", 1, PixelTag));
    auto & quad = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_tinc", Fix::Dim, "quad"));

    // Set random values
    std::mt19937 gen(456);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    for (Index_t i = 0; i < quad.eigen_vec().size(); ++i) {
      quad.eigen_vec()(i) = dist(gen);
    }

    // Compute reference: nodal_ref = 1.7 * B^T * quad
    nodal_ref.set_zero();
    Fix::op.transpose(quad, nodal_ref);
    nodal_ref.eigen_vec() *= 1.7;

    // Compute using transpose_increment
    nodal.set_zero();
    Fix::op.transpose_increment(quad, 1.7, nodal);

    // Check they match
    Real error = testGoodies::rel_error(nodal.eigen_vec(), nodal_ref.eigen_vec());
    BOOST_CHECK_LE(error, tol);
  }

  /* ---------------------------------------------------------------------- */
  /* Test sparse operator caching                                            */
  /* ---------------------------------------------------------------------- */

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(cache_consistency, Fix, ConvolutionFixtures,
                                   Fix) {
    auto & nodal = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_cache", 1, PixelTag));
    auto & quad1 = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_cache1", Fix::Dim, "quad"));
    auto & quad2 = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_cache2", Fix::Dim, "quad"));

    // Set random values
    std::mt19937 gen(789);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    for (Index_t i = 0; i < nodal.eigen_vec().size(); ++i) {
      nodal.eigen_vec()(i) = dist(gen);
    }

    // Apply twice (second should use cached operator)
    Fix::op.apply(nodal, quad1);
    Fix::op.apply(nodal, quad2);

    // Results should be identical
    Real error = testGoodies::rel_error(quad1.eigen_vec(), quad2.eigen_vec());
    BOOST_CHECK_EQUAL(error, 0.0);

    // Clear cache and apply again
    Fix::op.clear_cache();
    Fix::op.apply(nodal, quad2);

    // Results should still match
    error = testGoodies::rel_error(quad1.eigen_vec(), quad2.eigen_vec());
    BOOST_CHECK_EQUAL(error, 0.0);
  }

  /* ---------------------------------------------------------------------- */
  /* Test multi-component fields                                             */
  /* ---------------------------------------------------------------------- */

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(multi_component_field, Fix,
                                   ConvolutionFixtures, Fix) {
    const Index_t nb_components = 3;
    auto & nodal = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_multi", nb_components, PixelTag));
    auto & quad = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_multi",
                                            nb_components * Fix::Dim, "quad"));

    // Set random values
    std::mt19937 gen(999);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    for (Index_t i = 0; i < nodal.eigen_vec().size(); ++i) {
      nodal.eigen_vec()(i) = dist(gen);
    }

    // Apply gradient
    Fix::op.apply(nodal, quad);

    // Check adjoint property
    auto & nodal2 = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_multi2", nb_components, PixelTag));
    auto & quad2 = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_multi2",
                                            nb_components * Fix::Dim, "quad"));
    for (Index_t i = 0; i < quad2.eigen_vec().size(); ++i) {
      quad2.eigen_vec()(i) = dist(gen);
    }

    Fix::op.transpose(quad2, nodal2);

    Real Bu_dot_v = quad.eigen_vec().dot(quad2.eigen_vec());
    Real u_dot_BTv = nodal.eigen_vec().dot(nodal2.eigen_vec());

    Real error = testGoodies::rel_error(Bu_dot_v, u_dot_BTv);
    BOOST_CHECK_LE(error, tol);
  }

  /* ---------------------------------------------------------------------- */
  /* Tests for SparseOperatorSoA structure                                   */
  /* ---------------------------------------------------------------------- */

  BOOST_AUTO_TEST_CASE(sparse_operator_soa_construction) {
    const Index_t size = 100;
    SparseOperatorSoA<HostSpace> sparse_op(size);

    BOOST_CHECK_EQUAL(sparse_op.size, size);
    BOOST_CHECK_EQUAL(sparse_op.quad_indices.size(), static_cast<size_t>(size));
    BOOST_CHECK_EQUAL(sparse_op.nodal_indices.size(), static_cast<size_t>(size));
    BOOST_CHECK_EQUAL(sparse_op.values.size(), static_cast<size_t>(size));
    BOOST_CHECK(!sparse_op.empty());
  }

  BOOST_AUTO_TEST_CASE(sparse_operator_soa_empty) {
    SparseOperatorSoA<HostSpace> sparse_op;

    BOOST_CHECK_EQUAL(sparse_op.size, 0);
    BOOST_CHECK(sparse_op.empty());
  }

  /* ---------------------------------------------------------------------- */
  /* Tests for GridTraversalParams                                           */
  /* ---------------------------------------------------------------------- */

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(grid_traversal_params, Fix,
                                   ConvolutionFixtures, Fix) {
    auto & nodal = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_params", 1, PixelTag));
    auto & quad = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_params", Fix::Dim, "quad"));

    // Just verify apply works (internally tests compute_traversal_params)
    nodal.eigen_vec().setRandom();
    Fix::op.apply(nodal, quad);

    // No crash = success for this basic test
    BOOST_CHECK(true);
  }

  /* ---------------------------------------------------------------------- */
  /* Tests for deep_copy_sparse_operator                                     */
  /* ---------------------------------------------------------------------- */

  BOOST_AUTO_TEST_CASE(sparse_operator_deep_copy) {
    const Index_t size = 50;
    SparseOperatorSoA<HostSpace> src(size);

    // Fill with test data
    for (Index_t i = 0; i < size; ++i) {
      src.quad_indices(i) = i;
      src.nodal_indices(i) = size - i;
      src.values(i) = static_cast<Real>(i) * 0.1;
    }

    // Deep copy (same space - should still work)
    auto dst = deep_copy_sparse_operator<HostSpace, HostSpace>(src);

    BOOST_CHECK_EQUAL(dst.size, size);

    // Verify data was copied correctly
    for (Index_t i = 0; i < size; ++i) {
      BOOST_CHECK_EQUAL(dst.quad_indices(i), src.quad_indices(i));
      BOOST_CHECK_EQUAL(dst.nodal_indices(i), src.nodal_indices(i));
      BOOST_CHECK_EQUAL(dst.values(i), src.values(i));
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Tests for pad_shape_to_3d helper                                        */
  /* ---------------------------------------------------------------------- */

  BOOST_AUTO_TEST_CASE(pad_shape_to_3d_test) {
    // 1D shape
    Shape_t shape1d{5};
    auto padded1d = pad_shape_to_3d(shape1d);
    BOOST_CHECK_EQUAL(padded1d.size(), 3u);
    BOOST_CHECK_EQUAL(padded1d[0], 5);
    BOOST_CHECK_EQUAL(padded1d[1], 1);
    BOOST_CHECK_EQUAL(padded1d[2], 1);

    // 2D shape
    Shape_t shape2d{5, 7};
    auto padded2d = pad_shape_to_3d(shape2d);
    BOOST_CHECK_EQUAL(padded2d.size(), 3u);
    BOOST_CHECK_EQUAL(padded2d[0], 5);
    BOOST_CHECK_EQUAL(padded2d[1], 7);
    BOOST_CHECK_EQUAL(padded2d[2], 1);

    // 3D shape (no padding needed)
    Shape_t shape3d{5, 7, 9};
    auto padded3d = pad_shape_to_3d(shape3d);
    BOOST_CHECK_EQUAL(padded3d.size(), 3u);
    BOOST_CHECK_EQUAL(padded3d[0], 5);
    BOOST_CHECK_EQUAL(padded3d[1], 7);
    BOOST_CHECK_EQUAL(padded3d[2], 9);

    // Custom fill value
    auto padded_custom = pad_shape_to_3d(shape1d, 42);
    BOOST_CHECK_EQUAL(padded_custom[1], 42);
    BOOST_CHECK_EQUAL(padded_custom[2], 42);
  }

  /* ---------------------------------------------------------------------- */
  /* Performance/stress tests with larger grids                              */
  /* ---------------------------------------------------------------------- */

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(large_grid_apply, Fix, LargeConvolutionFixtures,
                                   Fix) {
    auto & nodal = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_large", 1, PixelTag));
    auto & quad = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_large", Fix::Dim, "quad"));

    // Set random values
    std::mt19937 gen(12345);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    for (Index_t i = 0; i < nodal.eigen_vec().size(); ++i) {
      nodal.eigen_vec()(i) = dist(gen);
    }

    // Multiple iterations to stress test
    for (int iter = 0; iter < 5; ++iter) {
      Fix::op.apply(nodal, quad);
    }

    // Basic sanity check - gradient shouldn't be all zeros for random input
    Real norm = quad.eigen_vec().norm();
    BOOST_CHECK_GT(norm, 0.0);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(large_grid_transpose, Fix,
                                   LargeConvolutionFixtures, Fix) {
    auto & nodal = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_large_t", 1, PixelTag));
    auto & quad = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_large_t", Fix::Dim, "quad"));

    // Set random values in quad field
    std::mt19937 gen(54321);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    for (Index_t i = 0; i < quad.eigen_vec().size(); ++i) {
      quad.eigen_vec()(i) = dist(gen);
    }

    // Multiple iterations
    for (int iter = 0; iter < 5; ++iter) {
      Fix::op.transpose(quad, nodal);
    }

    // Basic sanity check
    Real norm = nodal.eigen_vec().norm();
    BOOST_CHECK_GT(norm, 0.0);
  }

  /* ---------------------------------------------------------------------- */
  /* GPU-specific tests (only run when GPU backend is available)             */
  /* ---------------------------------------------------------------------- */

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)

  BOOST_AUTO_TEST_CASE(device_sparse_operator_copy) {
    // Create host sparse operator
    const Index_t size = 100;
    SparseOperatorSoA<HostSpace> host_op(size);

    // Fill with test data
    for (Index_t i = 0; i < size; ++i) {
      host_op.quad_indices(i) = i * 2;
      host_op.nodal_indices(i) = i * 3;
      host_op.values(i) = static_cast<Real>(i) * 0.5;
    }

    // Copy to device
    auto device_op = deep_copy_sparse_operator<DefaultDeviceSpace, HostSpace>(host_op);

    BOOST_CHECK_EQUAL(device_op.size, size);

    // Copy back to host for verification
    auto host_copy = deep_copy_sparse_operator<HostSpace, DefaultDeviceSpace>(device_op);

    // Verify data round-tripped correctly
    for (Index_t i = 0; i < size; ++i) {
      BOOST_CHECK_EQUAL(host_copy.quad_indices(i), host_op.quad_indices(i));
      BOOST_CHECK_EQUAL(host_copy.nodal_indices(i), host_op.nodal_indices(i));
      BOOST_CHECK_EQUAL(host_copy.values(i), host_op.values(i));
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(device_apply_consistency, Fix,
                                   ConvolutionFixtures, Fix) {
    // This test verifies that CPU apply produces consistent results
    // (GPU kernel tests would require access to private methods)
    auto & nodal = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_gpu", 1, PixelTag));
    auto & quad1 = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_gpu1", Fix::Dim, "quad"));
    auto & quad2 = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_gpu2", Fix::Dim, "quad"));

    // Set random values
    std::mt19937 gen(11111);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    for (Index_t i = 0; i < nodal.eigen_vec().size(); ++i) {
      nodal.eigen_vec()(i) = dist(gen);
    }

    // Compute twice - should give identical results
    Fix::op.apply(nodal, quad1);
    Fix::op.apply(nodal, quad2);

    // Results should be identical
    Real error = testGoodies::rel_error(quad1.eigen_vec(), quad2.eigen_vec());
    BOOST_CHECK_EQUAL(error, 0.0);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(device_transpose_consistency, Fix,
                                   ConvolutionFixtures, Fix) {
    // Verify transpose produces consistent results
    auto & nodal1 = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_gpu_t1", 1, PixelTag));
    auto & nodal2 = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_gpu_t2", 1, PixelTag));
    auto & quad = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_gpu_t", Fix::Dim, "quad"));

    // Set random values in quad field
    std::mt19937 gen(22222);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    for (Index_t i = 0; i < quad.eigen_vec().size(); ++i) {
      quad.eigen_vec()(i) = dist(gen);
    }

    // Compute twice
    Fix::op.transpose(quad, nodal1);
    Fix::op.transpose(quad, nodal2);

    // Results should be identical
    Real error = testGoodies::rel_error(nodal1.eigen_vec(), nodal2.eigen_vec());
    BOOST_CHECK_EQUAL(error, 0.0);
  }

#endif  // KOKKOS_ENABLE_CUDA || KOKKOS_ENABLE_HIP

  /* ---------------------------------------------------------------------- */
  /* Test that verifies Kokkos execution spaces are configured correctly     */
  /* ---------------------------------------------------------------------- */

  BOOST_AUTO_TEST_CASE(kokkos_configuration_test) {
    // Print Kokkos configuration for debugging
    std::cout << "Kokkos configuration:" << std::endl;
    std::cout << "  Default execution space: "
              << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
    std::cout << "  Default host execution space: "
              << typeid(Kokkos::DefaultHostExecutionSpace).name() << std::endl;
    std::cout << "  HostSpace: " << typeid(HostSpace).name() << std::endl;
    std::cout << "  DefaultDeviceSpace: "
              << typeid(DefaultDeviceSpace).name() << std::endl;

#if defined(KOKKOS_ENABLE_CUDA)
    std::cout << "  CUDA enabled" << std::endl;
    BOOST_CHECK(true);
#elif defined(KOKKOS_ENABLE_HIP)
    std::cout << "  HIP enabled" << std::endl;
    BOOST_CHECK(true);
#elif defined(KOKKOS_ENABLE_OPENMP)
    std::cout << "  OpenMP enabled" << std::endl;
    BOOST_CHECK(true);
#elif defined(KOKKOS_ENABLE_SERIAL)
    std::cout << "  Serial backend" << std::endl;
    BOOST_CHECK(true);
#else
    std::cout << "  Unknown backend" << std::endl;
#endif

    BOOST_CHECK(Kokkos::is_initialized());
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
