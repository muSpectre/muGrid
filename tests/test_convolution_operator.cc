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

#include "operators/convolution_operator.hh"
#include "collection/field_collection_global.hh"
#include "field/field_typed.hh"
#include "field/field_map.hh"

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

    static std::vector<Real> get_pixel_operator() {
      // Simple finite difference gradient operator
      // Layout: column-major (Fortran order) to match original Eigen matrix
      // rows = NbOperators * NbQuadPts = Dim
      // cols = NbNodalPts * 2^Dim
      const Index_t nb_stencil_pts = static_cast<Index_t>(std::pow(2, Dim));
      const Index_t nb_rows = Dim;
      std::vector<Real> op(nb_rows * nb_stencil_pts, 0.0);

      if (Dim == 2) {
        // 2D gradient using bilinear interpolation
        // Stencil: (0,0), (1,0), (0,1), (1,1)
        // d/dx: [-1, 1, -1, 1] / 2
        // d/dy: [-1, -1, 1, 1] / 2
        // Column-major: op[row + col * nb_rows]
        op[0 + 0*nb_rows] = -0.5; op[0 + 1*nb_rows] = 0.5; op[0 + 2*nb_rows] = -0.5; op[0 + 3*nb_rows] = 0.5;
        op[1 + 0*nb_rows] = -0.5; op[1 + 1*nb_rows] = -0.5; op[1 + 2*nb_rows] = 0.5; op[1 + 3*nb_rows] = 0.5;
      } else if (Dim == 3) {
        // 3D gradient using trilinear interpolation
        // Stencil: (0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)
        // d/dx: [-1, 1, -1, 1, -1, 1, -1, 1] / 4
        // d/dy: [-1, -1, 1, 1, -1, -1, 1, 1] / 4
        // d/dz: [-1, -1, -1, -1, 1, 1, 1, 1] / 4
        for (Index_t i = 0; i < 8; ++i) {
          op[0 + i*nb_rows] = ((i & 1) ? 0.25 : -0.25);       // x-derivative
          op[1 + i*nb_rows] = ((i & 2) ? 0.25 : -0.25);       // y-derivative
          op[2 + i*nb_rows] = ((i & 4) ? 0.25 : -0.25);       // z-derivative
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
    DynGridIndex nb_grid_pts{grid_size, grid_size};
    DynGridIndex nb_subdomain_grid_pts{grid_size + 2, grid_size + 2};  // +2 for symmetric ghosts
    DynGridIndex subdomain_locations{0, 0};
    DynGridIndex nb_ghosts_left{1, 1};   // needed for transpose (reads at negative offsets)
    DynGridIndex nb_ghosts_right{1, 1};  // needed for apply (reads at positive offsets)

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
    DynGridIndex nb_grid_pts{grid_size, grid_size, grid_size};
    DynGridIndex nb_subdomain_grid_pts{grid_size + 2, grid_size + 2, grid_size + 2};  // +2 for symmetric ghosts
    DynGridIndex subdomain_locations{0, 0, 0};
    DynGridIndex nb_ghosts_left{1, 1, 1};   // needed for transpose (reads at negative offsets)
    DynGridIndex nb_ghosts_right{1, 1, 1};  // needed for apply (reads at positive offsets)

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
    DynGridIndex nb_grid_pts{grid_size, grid_size};
    DynGridIndex nb_subdomain_grid_pts{grid_size + 2, grid_size + 2};  // +2 for symmetric ghosts
    DynGridIndex subdomain_locations{0, 0};
    DynGridIndex nb_ghosts_left{1, 1};   // needed for transpose (reads at negative offsets)
    DynGridIndex nb_ghosts_right{1, 1};  // needed for apply (reads at positive offsets)

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
    DynGridIndex nb_grid_pts{grid_size, grid_size, grid_size};
    DynGridIndex nb_subdomain_grid_pts{grid_size + 2, grid_size + 2, grid_size + 2};  // +2 for symmetric ghosts
    DynGridIndex subdomain_locations{0, 0, 0};
    DynGridIndex nb_ghosts_left{1, 1, 1};   // needed for transpose (reads at negative offsets)
    DynGridIndex nb_ghosts_right{1, 1, 1};  // needed for apply (reads at positive offsets)

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
    const Index_t expected_size = Fix::NbOperators * Fix::NbQuadPts *
                                  Fix::NbNodalPts *
                                  static_cast<Index_t>(std::pow(2, Fix::Dim));
    BOOST_CHECK_EQUAL(pixel_op.size(), expected_size);
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

    // Use raw data access to set values, matching how the sparse operator indexes
    Real* nodal_data = nodal.data();
    const auto& pixels = Fix::collection.get_pixels_with_ghosts();
    for (auto && [id, ccoord] : pixels.enumerate()) {
      Real val = a * ccoord[0] + b * ccoord[1];
      if (Fix::Dim == 3) {
        val += c * ccoord[2];
      }
      nodal_data[id] = val;
    }

    // Apply gradient
    Fix::op.apply(nodal, quad);

    // Debug: check if any non-zero values were written
    Real max_grad = quad.eigen_vec().cwiseAbs().maxCoeff();
    BOOST_TEST_MESSAGE("Max gradient magnitude: " << max_grad);
    BOOST_CHECK_GT(max_grad, 0.0);  // Should have non-zero gradient

    // Check gradient is [a, b] (or [a, b, c] for 3D) at interior points
    // Use raw data access with proper ghost offset calculation
    const Real* quad_data = quad.data();
    const Index_t nb_quad_components = Fix::Dim;
    const Index_t start_idx = Fix::collection.get_pixels_index_diff();

    // Compute stride (row width including ghosts) - must account for components per pixel
    const auto& subdomain_pts = Fix::collection.get_nb_subdomain_grid_pts_with_ghosts();
    const Index_t quad_stride_x = nb_quad_components;  // stride in x direction
    const Index_t quad_stride_y = subdomain_pts[0] * nb_quad_components;  // stride in y direction
    Index_t quad_stride_z = 1;
    if (Fix::Dim == 3) {
      quad_stride_z = subdomain_pts[0] * subdomain_pts[1] * nb_quad_components;
    }

    // Direct loop matching the kernel's iteration pattern
    const Index_t nx = Fix::nb_grid_pts[0];
    const Index_t ny = Fix::nb_grid_pts[1];
    const Index_t nz = (Fix::Dim == 3) ? Fix::nb_grid_pts[2] : 1;
    const Index_t quad_base = start_idx * nb_quad_components;

    for (Index_t z = 0; z < nz; ++z) {
      for (Index_t y = 0; y < ny; ++y) {
        for (Index_t x = 0; x < nx; ++x) {
          Index_t quad_offset = quad_base + z * quad_stride_z + y * quad_stride_y + x * quad_stride_x;
          const Real* grad = quad_data + quad_offset;

          Real err_x = std::abs(grad[0] - a);
          Real err_y = std::abs(grad[1] - b);
          BOOST_CHECK_LE(err_x, tol);
          BOOST_CHECK_LE(err_y, tol);
          if (Fix::Dim == 3) {
            Real err_z = std::abs(grad[2] - c);
            BOOST_CHECK_LE(err_z, tol);
          }
        }
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Test transpose operator (adjoint property)                              */
  /* ---------------------------------------------------------------------- */

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(transpose_adjoint_property, Fix,
                                   ConvolutionFixtures, Fix) {
    // Test that <Bu, v> = <u, B^T v> for random u, v
    // NOTE: Ghost pixels must be zero for the adjoint property to hold exactly,
    // because Apply reads from right ghosts while Transpose reads from left ghosts.
    auto & u = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("u", 1, PixelTag));
    auto & v = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("v", Fix::Dim, "quad"));
    auto & Bu = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("Bu", Fix::Dim, "quad"));
    auto & BTv = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("BTv", 1, PixelTag));

    // Initialize all to zero first (including ghosts)
    u.set_zero();
    v.set_zero();

    // Then set random values only for interior pixels
    std::mt19937 gen(42);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);

    // Set u (nodal field) interior values
    Real* u_data = u.data();
    const Index_t u_start = Fix::collection.get_pixels_index_diff();
    const Index_t u_elems_per_pixel = 1;  // 1 component, 1 nodal pt
    const auto& subdomain = Fix::collection.get_nb_subdomain_grid_pts_with_ghosts();
    const Index_t u_stride_x = u_elems_per_pixel;
    const Index_t u_stride_y = subdomain[0] * u_elems_per_pixel;
    const Index_t u_stride_z = (Fix::Dim == 3) ? subdomain[0] * subdomain[1] * u_elems_per_pixel : 1;

    const Index_t nx = Fix::nb_grid_pts[0];
    const Index_t ny = Fix::nb_grid_pts[1];
    const Index_t nz = (Fix::Dim == 3) ? Fix::nb_grid_pts[2] : 1;

    for (Index_t z = 0; z < nz; ++z) {
      for (Index_t y = 0; y < ny; ++y) {
        for (Index_t x = 0; x < nx; ++x) {
          Index_t idx = u_start * u_elems_per_pixel + z * u_stride_z + y * u_stride_y + x * u_stride_x;
          u_data[idx] = dist(gen);
        }
      }
    }

    // Set v (quad field) interior values
    Real* v_data = v.data();
    const Index_t v_elems_per_pixel = Fix::Dim;  // Dim components, 1 quad pt
    const Index_t v_stride_x = v_elems_per_pixel;
    const Index_t v_stride_y = subdomain[0] * v_elems_per_pixel;
    const Index_t v_stride_z = (Fix::Dim == 3) ? subdomain[0] * subdomain[1] * v_elems_per_pixel : 1;

    for (Index_t z = 0; z < nz; ++z) {
      for (Index_t y = 0; y < ny; ++y) {
        for (Index_t x = 0; x < nx; ++x) {
          Index_t idx = u_start * v_elems_per_pixel + z * v_stride_z + y * v_stride_y + x * v_stride_x;
          for (Index_t c = 0; c < Fix::Dim; ++c) {
            v_data[idx + c] = dist(gen);
          }
        }
      }
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
    // Test adjoint property with multi-component fields
    // NOTE: Ghost pixels must be zero for the adjoint property to hold exactly.
    const Index_t nb_components = 3;
    auto & nodal = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_multi", nb_components, PixelTag));
    auto & quad = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_multi",
                                            nb_components * Fix::Dim, "quad"));

    // Initialize all to zero first (including ghosts)
    nodal.set_zero();

    // Set random values only for interior pixels
    std::mt19937 gen(999);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);

    Real* nodal_data = nodal.data();
    const Index_t start_idx = Fix::collection.get_pixels_index_diff();
    const auto& subdomain = Fix::collection.get_nb_subdomain_grid_pts_with_ghosts();
    const Index_t nodal_elems = nb_components;  // components per pixel
    const Index_t nodal_stride_x = nodal_elems;
    const Index_t nodal_stride_y = subdomain[0] * nodal_elems;
    const Index_t nodal_stride_z = (Fix::Dim == 3) ? subdomain[0] * subdomain[1] * nodal_elems : 1;

    const Index_t nx = Fix::nb_grid_pts[0];
    const Index_t ny = Fix::nb_grid_pts[1];
    const Index_t nz = (Fix::Dim == 3) ? Fix::nb_grid_pts[2] : 1;

    for (Index_t z = 0; z < nz; ++z) {
      for (Index_t y = 0; y < ny; ++y) {
        for (Index_t x = 0; x < nx; ++x) {
          Index_t idx = start_idx * nodal_elems + z * nodal_stride_z + y * nodal_stride_y + x * nodal_stride_x;
          for (Index_t c = 0; c < nb_components; ++c) {
            nodal_data[idx + c] = dist(gen);
          }
        }
      }
    }

    // Apply gradient
    Fix::op.apply(nodal, quad);

    // Check adjoint property
    auto & nodal2 = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("nodal_multi2", nb_components, PixelTag));
    auto & quad2 = dynamic_cast<RealField &>(
        Fix::collection.register_real_field("quad_multi2",
                                            nb_components * Fix::Dim, "quad"));

    // Initialize quad2 with zeros, then set interior values
    quad2.set_zero();
    Real* quad2_data = quad2.data();
    const Index_t quad_elems = nb_components * Fix::Dim;  // components per pixel
    const Index_t quad_stride_x = quad_elems;
    const Index_t quad_stride_y = subdomain[0] * quad_elems;
    const Index_t quad_stride_z = (Fix::Dim == 3) ? subdomain[0] * subdomain[1] * quad_elems : 1;

    for (Index_t z = 0; z < nz; ++z) {
      for (Index_t y = 0; y < ny; ++y) {
        for (Index_t x = 0; x < nx; ++x) {
          Index_t idx = start_idx * quad_elems + z * quad_stride_z + y * quad_stride_y + x * quad_stride_x;
          for (Index_t c = 0; c < quad_elems; ++c) {
            quad2_data[idx + c] = dist(gen);
          }
        }
      }
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
      src.quad_indices[i] = i;
      src.nodal_indices[i] = size - i;
      src.values[i] = static_cast<Real>(i) * 0.1;
    }

    // Deep copy (same space - should still work)
    auto dst = deep_copy_sparse_operator<HostSpace, HostSpace>(src);

    BOOST_CHECK_EQUAL(dst.size, size);

    // Verify data was copied correctly
    for (Index_t i = 0; i < size; ++i) {
      BOOST_CHECK_EQUAL(dst.quad_indices[i], src.quad_indices[i]);
      BOOST_CHECK_EQUAL(dst.nodal_indices[i], src.nodal_indices[i]);
      BOOST_CHECK_EQUAL(dst.values[i], src.values[i]);
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

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)

  BOOST_AUTO_TEST_CASE(device_sparse_operator_copy) {
    // Create host sparse operator
    const Index_t size = 100;
    SparseOperatorSoA<HostSpace> host_op(size);

    // Fill with test data
    for (Index_t i = 0; i < size; ++i) {
      host_op.quad_indices[i] = i * 2;
      host_op.nodal_indices[i] = i * 3;
      host_op.values[i] = static_cast<Real>(i) * 0.5;
    }

    // Copy to device
    auto device_op = deep_copy_sparse_operator<DefaultDeviceSpace, HostSpace>(host_op);

    BOOST_CHECK_EQUAL(device_op.size, size);

    // Copy back to host for verification
    auto host_copy = deep_copy_sparse_operator<HostSpace, DefaultDeviceSpace>(device_op);

    // Verify data round-tripped correctly
    for (Index_t i = 0; i < size; ++i) {
      BOOST_CHECK_EQUAL(host_copy.quad_indices[i], host_op.quad_indices[i]);
      BOOST_CHECK_EQUAL(host_copy.nodal_indices[i], host_op.nodal_indices[i]);
      BOOST_CHECK_EQUAL(host_copy.values[i], host_op.values[i]);
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

#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP

  /* ---------------------------------------------------------------------- */
  /* Test that verifies memory spaces are configured correctly               */
  /* ---------------------------------------------------------------------- */

  BOOST_AUTO_TEST_CASE(memory_space_configuration_test) {
    // Print memory space configuration for debugging
    std::cout << "muGrid memory space configuration:" << std::endl;
    std::cout << "  HostSpace: " << typeid(HostSpace).name() << std::endl;
    std::cout << "  DefaultDeviceSpace: "
              << typeid(DefaultDeviceSpace).name() << std::endl;

#if defined(MUGRID_ENABLE_CUDA)
    std::cout << "  CUDA backend enabled" << std::endl;
    BOOST_CHECK(true);
#elif defined(MUGRID_ENABLE_HIP)
    std::cout << "  HIP backend enabled" << std::endl;
    BOOST_CHECK(true);
#else
    std::cout << "  CPU-only (no GPU backend)" << std::endl;
    BOOST_CHECK(true);
#endif

    // Verify HostSpace is correctly identified
    BOOST_CHECK(is_host_space_v<HostSpace>);
    BOOST_CHECK(!is_device_space_v<HostSpace>);
  }

  /* ---------------------------------------------------------------------- */
  /* Test fourier method                                                     */
  /* ---------------------------------------------------------------------- */

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(fourier_zero_phase, Fix, ConvolutionFixtures,
                                   Fix) {
    // At zero phase, fourier should return the sum of all operator coefficients
    Eigen::VectorXd phase = Eigen::VectorXd::Zero(Fix::Dim);
    Complex result = Fix::op.fourier(phase);

    // For gradient operators, the sum of coefficients should be zero
    // (finite differences sum to zero for translational invariance)
    BOOST_CHECK_SMALL(std::abs(result), tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(fourier_dimension_validation, Fix,
                                   ConvolutionFixtures, Fix) {
    // Test that fourier() validates phase dimension
    Eigen::VectorXd wrong_phase = Eigen::VectorXd::Zero(Fix::Dim + 1);

    BOOST_CHECK_THROW(Fix::op.fourier(wrong_phase), RuntimeError);
  }

  BOOST_AUTO_TEST_CASE(fourier_1d_central_difference) {
    // Create a 1D central difference operator: [-1/2, 0, 1/2]
    // Stencil at positions -1, 0, 1 (centered)
    // This should give Fourier representation: i*sin(2π*q)

    Shape_t pixel_offset{-1};  // offset [-1] for centered stencil
    Shape_t conv_pts_shape{3};  // 3 points in stencil
    std::vector<Real> pixel_operator{-0.5, 0.0, 0.5};  // central difference

    ConvolutionOperator op{pixel_offset, pixel_operator, conv_pts_shape,
                          1, 1, 1};  // 1 nodal pt, 1 quad pt, 1 operator

    // Test at several phase values
    const Real pi_val = 3.1415926535897932384626433;
    std::vector<Real> test_phases{0.0, 0.1, 0.25, 0.5};

    for (Real q : test_phases) {
      Eigen::VectorXd phase(1);
      phase(0) = q;
      Complex result = op.fourier(phase);

      // Expected: i*sin(2π*q)
      Complex expected(0.0, std::sin(2.0 * pi_val * q));

      BOOST_CHECK_SMALL(std::abs(result - expected), tol);
    }
  }

  BOOST_AUTO_TEST_CASE(fourier_1d_upwind_difference) {
    // Create a 1D backward (upwind) difference operator: [-1, 1, 0]
    // Stencil at positions -1, 0, 1
    // This should give Fourier representation: 1 - exp(-2πiq)

    Shape_t pixel_offset{-1};  // offset [-1] for centered stencil
    Shape_t conv_pts_shape{3};
    std::vector<Real> pixel_operator{-1.0, 1.0, 0.0};  // backward difference

    ConvolutionOperator op{pixel_offset, pixel_operator, conv_pts_shape,
                          1, 1, 1};

    const Real pi_val = 3.1415926535897932384626433;
    std::vector<Real> test_phases{0.0, 0.1, 0.25, 0.5};

    for (Real q : test_phases) {
      Eigen::VectorXd phase(1);
      phase(0) = q;
      Complex result = op.fourier(phase);

      // Expected: 1 - exp(-2πiq)
      Complex expected = Complex(1.0, 0.0) -
                        std::exp(Complex(0.0, -2.0 * pi_val * q));

      BOOST_CHECK_SMALL(std::abs(result - expected), tol);
    }
  }

  BOOST_AUTO_TEST_CASE(fourier_1d_second_derivative) {
    // Create a 1D second derivative operator: [1, -2, 1]
    // Stencil at positions -1, 0, 1 (centered)
    // This should give Fourier representation: -4*sin²(π*q)

    Shape_t pixel_offset{-1};  // offset [-1] for centered stencil
    Shape_t conv_pts_shape{3};
    std::vector<Real> pixel_operator{1.0, -2.0, 1.0};  // second derivative

    ConvolutionOperator op{pixel_offset, pixel_operator, conv_pts_shape,
                          1, 1, 1};

    const Real pi_val = 3.1415926535897932384626433;
    std::vector<Real> test_phases{0.0, 0.1, 0.25, 0.5};

    for (Real q : test_phases) {
      Eigen::VectorXd phase(1);
      phase(0) = q;
      Complex result = op.fourier(phase);

      // Expected: -4*sin²(π*q) = 2*(cos(2π*q) - 1)
      Real sin_term = std::sin(pi_val * q);
      Complex expected(-4.0 * sin_term * sin_term, 0.0);

      BOOST_CHECK_SMALL(std::abs(result - expected), tol);
    }
  }

  BOOST_AUTO_TEST_CASE(fourier_2d_x_derivative) {
    // Create a 2D x-derivative operator using central differences
    // Stencil at x positions -1, 0, 1; y position 0
    // This should give Fourier representation: i*sin(2π*qx)

    Shape_t pixel_offset{-1, 0};  // offset [-1, 0] for x-centered stencil
    Shape_t conv_pts_shape{3, 1};  // 3 points in x, 1 in y
    // Operator coefficients for x-derivative: [-1/2, 0, 1/2] at y=0
    std::vector<Real> pixel_operator{-0.5, 0.0, 0.5};

    ConvolutionOperator op{pixel_offset, pixel_operator, conv_pts_shape,
                          1, 1, 1};

    const Real pi_val = 3.1415926535897932384626433;
    std::vector<std::pair<Real, Real>> test_phases{
      {0.0, 0.0}, {0.1, 0.0}, {0.25, 0.15}, {0.5, 0.3}};

    for (auto [qx, qy] : test_phases) {
      Eigen::VectorXd phase(2);
      phase(0) = qx;
      phase(1) = qy;
      Complex result = op.fourier(phase);

      // Expected: i*sin(2π*qx) (independent of qy since no y variation)
      Complex expected(0.0, std::sin(2.0 * pi_val * qx));

      BOOST_CHECK_SMALL(std::abs(result - expected), tol);
    }
  }

  BOOST_AUTO_TEST_CASE(fourier_2d_y_derivative) {
    // Create a 2D y-derivative operator using central differences
    // Stencil at x position 0; y positions -1, 0, 1

    Shape_t pixel_offset{0, -1};  // offset [0, -1] for y-centered stencil
    Shape_t conv_pts_shape{1, 3};  // 1 point in x, 3 in y
    std::vector<Real> pixel_operator{-0.5, 0.0, 0.5};

    ConvolutionOperator op{pixel_offset, pixel_operator, conv_pts_shape,
                          1, 1, 1};

    const Real pi_val = 3.1415926535897932384626433;
    std::vector<std::pair<Real, Real>> test_phases{
      {0.0, 0.0}, {0.0, 0.1}, {0.15, 0.25}, {0.3, 0.5}};

    for (auto [qx, qy] : test_phases) {
      Eigen::VectorXd phase(2);
      phase(0) = qx;
      phase(1) = qy;
      Complex result = op.fourier(phase);

      // Expected: i*sin(2π*qy) (independent of qx since no x variation)
      Complex expected(0.0, std::sin(2.0 * pi_val * qy));

      BOOST_CHECK_SMALL(std::abs(result - expected), tol);
    }
  }

  BOOST_AUTO_TEST_CASE(fourier_2d_laplacian) {
    // Create a 2D 5-point Laplacian stencil
    // Pattern:     0  1  0
    //              1 -4  1
    //              0  1  0
    // This should give Fourier representation:
    // -4*(sin²(π*qx) + sin²(π*qy))

    Shape_t pixel_offset{-1, -1};  // center at (0,0), so offset is (-1,-1)
    Shape_t conv_pts_shape{3, 3};  // 3x3 stencil
    // Layout is column-major through stencil points
    std::vector<Real> pixel_operator{
      0.0, 1.0, 0.0,    // x=-1: (y=-1,y=0,y=1)
      1.0, -4.0, 1.0,   // x=0:  (y=-1,y=0,y=1)
      0.0, 1.0, 0.0     // x=1:  (y=-1,y=0,y=1)
    };

    ConvolutionOperator op{pixel_offset, pixel_operator, conv_pts_shape,
                          1, 1, 1};

    const Real pi_val = 3.1415926535897932384626433;
    std::vector<std::pair<Real, Real>> test_phases{
      {0.0, 0.0}, {0.1, 0.1}, {0.25, 0.15}, {0.5, 0.3}};

    for (auto [qx, qy] : test_phases) {
      Eigen::VectorXd phase(2);
      phase(0) = qx;
      phase(1) = qy;
      Complex result = op.fourier(phase);

      // Expected: -4*(sin²(π*qx) + sin²(π*qy))
      Real sin_x = std::sin(pi_val * qx);
      Real sin_y = std::sin(pi_val * qy);
      Complex expected(-4.0 * (sin_x * sin_x + sin_y * sin_y), 0.0);

      BOOST_CHECK_SMALL(std::abs(result - expected), 1e-10);
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
