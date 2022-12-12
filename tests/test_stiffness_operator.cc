/**
 * @file   test_stiffness_operator.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   20 Jul 2020
 *
 * @brief  tests for finite element stiffness operator
 *
 * Copyright © 2020 Till Junge, Martin Ladecký
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
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
#include "libmugrid/test_goodies.hh"
#include "libmugrid/test_discrete_gradient_operator.hh"
#include "test_cell_data.hh"

#include "projection/stiffness_operator.hh"
#include "projection/fem_stencil.hh"
#include "projection/fem_library.hh"
#include "projection/discretisation.hh"
#include "materials/material_linear_anisotropic.hh"

#include <libmugrid/gradient_operator_default.hh>
#include <libmugrid/field_collection_global.hh>
#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/mapped_field.hh>

#include <projection/discrete_greens_operator.hh>

#include "cell/cell_data.hh"

#include <boost/mpl/list.hpp>

namespace muSpectre {

  template <Index_t DomainRank, Index_t Dim>
  struct DiscretisationFixtureBase {
    static constexpr Index_t DisplacementRank() { return DomainRank - 1; }
    static constexpr Index_t SpatialDim() { return Dim; }

    DiscretisationFixtureBase(std::shared_ptr<FEMStencilBase> stencil,
                              Eigen::Ref<const Eigen::MatrixXd> material_props)
        : discretisation{stencil},
          impulse_response{discretisation.compute_impulse_response(
              DisplacementRank(), material_props)},
          greens_operator{stencil->get_cell_ptr()->get_FFT_engine(),
                          *this->impulse_response, DisplacementRank()},
          material_properties{material_props} {}
    CellData_ptr get_cell_data() { return this->discretisation.get_cell(); }
    Discretisation discretisation;
    std::unique_ptr<muGrid::RealField, muGrid::FieldDestructor<muGrid::Field>>
        impulse_response;
    DiscreteGreensOperator greens_operator;
    Eigen::MatrixXd material_properties;
  };

  template <Index_t DomainRank>
  struct StraightTrianglesDiscretisationFixture
      : public DiscretisationFixtureBase<DomainRank, twoD> {
    using Parent = DiscretisationFixtureBase<DomainRank, twoD>;
    StraightTrianglesDiscretisationFixture()
        : Parent{FEMLibrary::linear_triangle_straight(
                     CellDataFixture<twoD>{}.cell_data),
                 get_material_properties()} {};

   private:
    static Eigen::MatrixXd get_material_properties() {
      switch (DomainRank) {
      case 1: {
        Eigen::MatrixXd mat_prop{Eigen::MatrixXd::Identity(twoD, twoD) +
                                 Eigen::MatrixXd::Random(twoD, twoD) / 100};
        return (mat_prop + mat_prop.transpose()).eval();
        break;
      }
      case 2: {
        // random elastic constants:
        Eigen::VectorXd elastic_constants{Eigen::VectorXd::Ones(6) +
                                          Eigen::VectorXd::Random(6) / 100};
        std::vector<Real> C_input{};
        for (Index_t i{0}; i < 6; ++i) {
          C_input.push_back(elastic_constants(i));
        }
        // return MaterialLinearAnisotropic<twoD>::c_maker(C_input);
        return MaterialLinearAnisotropic<twoD>::Stiffness_t::Identity();
        break;
      }
      default:
        throw std::runtime_error("I can only handle rank 1 or 2 problems");
        break;
      }
    }
  };

  using DiscretisationFixtures =
      boost::mpl::list<StraightTrianglesDiscretisationFixture<1>,
                       StraightTrianglesDiscretisationFixture<2>>;

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_SUITE(test_stiffness_operator);

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(reconstruct_1d_stiffness_matrix) {
    const Index_t nb_grid_pts{5};
    const Index_t displacement_rank{0};
    const Real del_x{4};
    const Real EA{2.4};

    DynCcoord_t nb_pixels{nb_grid_pts};
    DynRcoord_t lengths{nb_grid_pts * del_x};

    auto cell_ptr{CellData::make(nb_pixels, lengths)};
    auto stencil_ptr{FEMLibrary::linear_1d(cell_ptr)};

    const Index_t spatial_dimension{
        stencil_ptr->get_gradient_operator()->get_spatial_dim()};

    Eigen::MatrixXd tangent{
        muGrid::ipow(spatial_dimension, displacement_rank + 1),
        muGrid::ipow(spatial_dimension, displacement_rank + 1)};
    tangent << EA;

    Discretisation discretisation(stencil_ptr);

    auto K_operator{discretisation.get_stiffness_operator(displacement_rank)};

    Eigen::MatrixXd reconstructed_stiffness{
        Eigen::MatrixXd::Zero(nb_grid_pts, nb_grid_pts)};

    muGrid::GlobalFieldCollection collection{
        spatial_dimension,
        {nb_grid_pts},
        {nb_grid_pts},
        {},
        {{QuadPtTag,
           stencil_ptr->get_nb_pixel_quad_pts()},
         {NodalPtTag,
          stencil_ptr->get_nb_pixel_nodal_pts()}}};

    auto & displacement{collection.register_real_field(
        "displacement", muGrid::ipow(spatial_dimension, displacement_rank))};
    auto & force{collection.register_real_field(
        "force", muGrid::ipow(spatial_dimension, displacement_rank))};

    for (Index_t i{0}; i < nb_grid_pts; ++i) {
      displacement.eigen_vec() =
          Eigen::MatrixXd::Identity(nb_grid_pts, nb_grid_pts).col(i);

      K_operator.apply(tangent, displacement, force);
      reconstructed_stiffness.col(i) = force.eigen_vec();
    }

    Eigen::MatrixXd reference_stiffness{
        Eigen::MatrixXd::Zero(nb_grid_pts, nb_grid_pts)};
    for (int i{0}; i < nb_grid_pts; ++i) {
      reference_stiffness(i, i) = 2 * EA / del_x;
      reference_stiffness(i, (i - 1 + nb_grid_pts) % nb_grid_pts) = -EA / del_x;
      reference_stiffness(i, (i + 1) % nb_grid_pts) = -EA / del_x;
    }

    auto error{muGrid::testGoodies::rel_error(reference_stiffness,
                                              reconstructed_stiffness)};

    BOOST_CHECK_LE(error, tol);
    if (not(error < tol)) {
      std::cout << "reconstructed:" << std::endl
                << reconstructed_stiffness << std::endl;
      std::cout << "reference:" << std::endl
                << reference_stiffness << std::endl;
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(reconstruct_2d_stiffness_matrix,
                          muGrid::FixtureTriangularStraight) {
    const Index_t nb_grid_pts{3};
    const Index_t displacement_rank{1};
    const Index_t spatial_dimension{twoD};
    const Index_t problem_size{
        nb_grid_pts * nb_grid_pts *
        muGrid::ipow(spatial_dimension, displacement_rank)};

    const Real E{2.4};
    const Real nu{.35};

    Eigen::MatrixXd tangent{
        muGrid::ipow(spatial_dimension, displacement_rank + 1),
        muGrid::ipow(spatial_dimension, displacement_rank + 1)};
    tangent
        // clang-format off
        <<
      1 - nu,       0,        0,     nu,  // εxx
           0, .5 - nu,  .5 - nu,      0,  // εxy
           0, .5 - nu,  .5 - nu,      0,  // εyx
          nu,       0,        0, 1 - nu;  // εyy
    // clang-format on

    tangent *= E / (1 + nu) / (1 - 2 * nu);

    std::cout << "elastic tangent tensor" << std::endl << tangent << std::endl;

    DynCcoord_t nb_pixels{nb_grid_pts, nb_grid_pts};
    DynRcoord_t lengths{nb_grid_pts * this->del_x, nb_grid_pts * this->del_y};

    auto cell_ptr{CellData::make(nb_pixels, lengths)};
    auto stencil_ptr{FEMLibrary::linear_triangle_straight(cell_ptr)};

    Discretisation discretisation(stencil_ptr);

    auto K_operator{discretisation.get_stiffness_operator(displacement_rank)};

    Eigen::MatrixXd reconstructed_stiffness{
        Eigen::MatrixXd::Zero(problem_size, problem_size)};

    muGrid::GlobalFieldCollection collection{
        spatial_dimension,
        {nb_grid_pts, nb_grid_pts},
        {nb_grid_pts, nb_grid_pts},
        {},
        {{QuadPtTag, stencil_ptr->get_nb_pixel_quad_pts()},
         {NodalPtTag,
          stencil_ptr->get_nb_pixel_nodal_pts()}}};
    auto & displacement{collection.register_real_field(
        "displacement", muGrid::ipow(spatial_dimension, displacement_rank))};
    auto & force{collection.register_real_field(
        "force", muGrid::ipow(spatial_dimension, displacement_rank))};

    for (Index_t i{0}; i < problem_size; ++i) {
      Eigen::VectorXd unit_disp{
          Eigen::MatrixXd::Identity(problem_size, problem_size).col(i)};
      displacement.eigen_vec() = unit_disp;

      K_operator.apply(tangent, displacement, force);
      reconstructed_stiffness.col(i) = force.eigen_vec();
    }

    Eigen::MatrixXd reference_stiffness{
        Eigen::MatrixXd::Zero(problem_size, problem_size)};
    reference_stiffness
        // random stiffness matrix. the numbers have no deeper meaning
        // clang-format off
        <<
        12.7407,   2.96296,  -5.77778,  -1.48148,  -5.77778,  -1.48148,
        -0.592593,  -1.48148,         0,         0,         0,   1.48148,
        -0.592593,  -1.48148,         0,   1.48148,         0,         0,
        2.96296,   7.80247,  -1.48148,  -1.33333,  -1.48148,  -1.33333,
        -1.48148,   -2.5679,         0,         0,   1.48148,         0,
        -1.48148,   -2.5679,   1.48148,         0,         0,         0,
        -5.77778,  -1.48148,   12.7407,   2.96296,  -5.77778,  -1.48148,
        0,   1.48148, -0.592593,  -1.48148,         0,         0,
        0,         0, -0.592593,  -1.48148,         0,   1.48148,
        -1.48148,  -1.33333,   2.96296,   7.80247,  -1.48148,  -1.33333,
        1.48148,         0,  -1.48148,   -2.5679,         0,         0,
        0,         0,  -1.48148,   -2.5679,   1.48148,         0,
        -5.77778,  -1.48148,  -5.77778,  -1.48148,   12.7407,   2.96296,
        0,         0,         0,   1.48148, -0.592593,  -1.48148,
        0,   1.48148,         0,         0, -0.592593,  -1.48148,
        -1.48148,  -1.33333,  -1.48148,  -1.33333,   2.96296,   7.80247,
        0,         0,   1.48148,         0,  -1.48148,   -2.5679,
        1.48148,         0,         0,         0,  -1.48148,   -2.5679,
        -0.592593,  -1.48148,         0,   1.48148,         0,         0,
        12.7407,   2.96296,  -5.77778,  -1.48148,  -5.77778,  -1.48148,
        -0.592593,  -1.48148,         0,         0,         0,   1.48148,
        -1.48148,   -2.5679,   1.48148,         0,         0,         0,
        2.96296,   7.80247,  -1.48148,  -1.33333,  -1.48148,  -1.33333,
        -1.48148,   -2.5679,         0,         0,   1.48148,         0,
        0,         0, -0.592593,  -1.48148,         0,   1.48148,
        -5.77778,  -1.48148,   12.7407,   2.96296,  -5.77778,  -1.48148,
        0,   1.48148, -0.592593,  -1.48148,         0,         0,
        0,         0,  -1.48148,   -2.5679,   1.48148,         0,
        -1.48148,  -1.33333,   2.96296,   7.80247,  -1.48148,  -1.33333,
        1.48148,         0,  -1.48148,   -2.5679,         0,         0,
        0,   1.48148,         0,         0, -0.592593,  -1.48148,
        -5.77778,  -1.48148,  -5.77778,  -1.48148,   12.7407,   2.96296,
        0,         0,         0,   1.48148, -0.592593,  -1.48148,
        1.48148,         0,         0,         0,  -1.48148,   -2.5679,
        -1.48148,  -1.33333,  -1.48148,  -1.33333,   2.96296,   7.80247,
        0,         0,   1.48148,         0,  -1.48148,   -2.5679,
        -0.592593,  -1.48148,         0,         0,         0,   1.48148,
        -0.592593,  -1.48148,         0,   1.48148,         0,         0,
        12.7407,   2.96296,  -5.77778,  -1.48148,  -5.77778,  -1.48148,
        -1.48148,   -2.5679,         0,         0,   1.48148,         0,
        -1.48148,   -2.5679,   1.48148,         0,         0,         0,
        2.96296,   7.80247,  -1.48148,  -1.33333,  -1.48148,  -1.33333,
        0,   1.48148, -0.592593,  -1.48148,         0,         0,
        0,         0, -0.592593,  -1.48148,         0,   1.48148,
        -5.77778,  -1.48148,   12.7407,   2.96296,  -5.77778,  -1.48148,
        1.48148,         0,  -1.48148,   -2.5679,         0,         0,
        0,         0,  -1.48148,   -2.5679,   1.48148,         0,
        -1.48148,  -1.33333,   2.96296,   7.80247,  -1.48148,  -1.33333,
        0,         0,         0,   1.48148, -0.592593,  -1.48148,
        0,   1.48148,         0,         0, -0.592593,  -1.48148,
        -5.77778,  -1.48148,  -5.77778,  -1.48148,   12.7407,   2.96296,
        0,         0,   1.48148,         0,  -1.48148,   -2.5679,
        1.48148,         0,         0,         0,  -1.48148,   -2.5679,
        -1.48148,  -1.33333,  -1.48148,  -1.33333,   2.96296,   7.80247;
    // clang-format on
    auto error{muGrid::testGoodies::rel_error(reference_stiffness,
                                              reconstructed_stiffness)};
    Real tol{1e-5};
    BOOST_CHECK_LE(error, tol);
    if (not(error < tol)) {
      std::cout << "reconstructed:" << std::endl
                << reconstructed_stiffness << std::endl;
      std::cout << "reference:" << std::endl
                << reference_stiffness << std::endl;
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(reconstruct_2d_scalar_stiffness_matrix,
                          muGrid::FixtureTriangularStraight) {
    const Index_t nb_grid_pts{3};
    const Index_t displacement_rank{0};
    const Index_t spatial_dimension{twoD};
    const Index_t problem_size{
        nb_grid_pts * nb_grid_pts *
        muGrid::ipow(spatial_dimension, displacement_rank)};

    const Real E{19.};
    const Real nu{0.35};

    Eigen::MatrixXd tangent{
        muGrid::ipow(spatial_dimension, displacement_rank + 1),
        muGrid::ipow(spatial_dimension, displacement_rank + 1)};
    tangent <<  // clang-format off
                    E,        nu,         // εxx
                    nu,      E/5,   // εyy
                // clang-format on

        std::cout << "elastic tangent tensor" << std::endl
                  << tangent << std::endl;

    DynCcoord_t nb_pixels{nb_grid_pts, nb_grid_pts};
    DynRcoord_t lengths{nb_grid_pts * this->del_x, nb_grid_pts * this->del_y};

    auto cell_ptr{CellData::make(nb_pixels, lengths)};
    auto stencil_ptr{FEMLibrary::linear_triangle_straight(cell_ptr)};

    Discretisation discretisation(stencil_ptr);

    auto K_operator{discretisation.get_stiffness_operator(displacement_rank)};

    Eigen::MatrixXd reconstructed_stiffness{
        Eigen::MatrixXd::Zero(problem_size, problem_size)};

    muGrid::GlobalFieldCollection collection{
        spatial_dimension,
        {nb_grid_pts, nb_grid_pts},
        {nb_grid_pts, nb_grid_pts},
        {},
        {{QuadPtTag, stencil_ptr->get_nb_pixel_quad_pts()},
         {NodalPtTag, stencil_ptr->get_nb_pixel_nodal_pts()}}};

    auto & displacement{collection.register_real_field(
        "displacement", muGrid::ipow(spatial_dimension, displacement_rank))};
    auto & force{collection.register_real_field(
        "force", muGrid::ipow(spatial_dimension, displacement_rank))};

    for (Index_t i{0}; i < problem_size; ++i) {
      Eigen::VectorXd unit_disp{
          Eigen::MatrixXd::Identity(problem_size, problem_size).col(i)};
      displacement.eigen_vec() = unit_disp;

      K_operator.apply(tangent, displacement, force);
      reconstructed_stiffness.col(i) = force.eigen_vec();
    }

    Eigen::MatrixXd reference_stiffness{
        Eigen::MatrixXd::Zero(problem_size, problem_size)};
    reference_stiffness <<
        // clang-format off
          62.7667,     -28.85,    -28.85,
            -2.88333,         0,      0.35,
              -2.88333,     0.35,          0,
          -28.85,     62.7667,    -28.85,
                0.35,   -2.88333,         0,
                     0, -2.88333,       0.35,
          -28.85,      -28.85,   62.7667,
                   0,      0.35,  -2.88333,
                  0.35,        0,    -2.88333,
        -2.88333,        0.35,         0,
             62.7667,    -28.85,    -28.85,
              -2.88333,        0,        0.35,
               0,    -2.88333,      0.35,
              -28.85,   62.7667,    -28.85,
                  0.35, -2.88333,           0,
            0.35,           0,   -2.88333,
             -28.85,   -28.85,     62.7667,
                     0,    0.35,     -2.88333,
        -2.88333,          0,       0.35,
           -2.88333,     0.35,           0,
               62.7667,  -28.85,       -28.85,
            0.35,   -2.88333,          0,
                  0, -2.88333,        0.35,
                -28.85, 62.7667,       -28.85,
               0,       0.35,   -2.88333,
               0.35,        0,    -2.88333,
                -28.85,  -28.85,      62.7667;
    // clang-format on
    auto error{muGrid::testGoodies::rel_error(reference_stiffness,
                                              reconstructed_stiffness)};

    Real tol{1e-5};
    BOOST_CHECK_LE(error, tol);
    if (not(error < tol)) {
      std::cout << "reconstructed:" << std::endl
                << reconstructed_stiffness << std::endl;
      std::cout << "reference:" << std::endl
                << reference_stiffness << std::endl;
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(reconstruct_2d_scalar_stiffness_matrix_bilinear,
                          muGrid::FixtureBilinearQuadrilat) {
    const Index_t nb_grid_pts{3};
    const Index_t displacement_rank{0};
    const Index_t spatial_dimension{twoD};
    const Index_t problem_size{
        nb_grid_pts * nb_grid_pts *
        muGrid::ipow(spatial_dimension, displacement_rank)};

    const Real E{19.};
    const Real nu{0.35};

    Eigen::MatrixXd tangent{
        muGrid::ipow(spatial_dimension, displacement_rank + 1),
        muGrid::ipow(spatial_dimension, displacement_rank + 1)};
    tangent <<  // clang-format off
              E,        nu,  // εxx
             nu,      E/5,   // εyy
        // clang-format on

        std::cout << "elastic tangent tensor" << std::endl
                  << tangent << std::endl;

    Eigen::MatrixXd ref_tangent{
        muGrid::ipow(spatial_dimension, displacement_rank + 1),
        muGrid::ipow(spatial_dimension, displacement_rank + 1)};
    ref_tangent <<  // clang-format off
           E,       nu,   // εxx
          nu,      E/5,   // εyy
        // clang-format on

        std::cout << "elastic ref_tangent tensor" << std::endl
                  << ref_tangent << std::endl;

    DynCcoord_t nb_pixels{nb_grid_pts, nb_grid_pts};
    DynRcoord_t lengths{nb_grid_pts * this->del_x, nb_grid_pts * this->del_y};

    auto cell_ptr{CellData::make(nb_pixels, lengths)};
    auto stencil_ptr{FEMLibrary::bilinear_quadrangle(cell_ptr)};

    Discretisation discretisation(stencil_ptr);

    auto K_operator{discretisation.get_stiffness_operator(displacement_rank)};

    Eigen::MatrixXd reconstructed_stiffness{
        Eigen::MatrixXd::Zero(problem_size, problem_size)};

    muGrid::GlobalFieldCollection collection{
        spatial_dimension,
        {nb_grid_pts, nb_grid_pts},
        {nb_grid_pts, nb_grid_pts},
        {},
        {{QuadPtTag, stencil_ptr->get_nb_pixel_quad_pts()},
         {NodalPtTag, stencil_ptr->get_nb_pixel_nodal_pts()}}};

    auto & displacement{collection.register_real_field(
        "displacement", muGrid::ipow(spatial_dimension, displacement_rank))};
    auto & force{collection.register_real_field(
        "force", muGrid::ipow(spatial_dimension, displacement_rank))};

    for (Index_t i{0}; i < problem_size; ++i) {
      Eigen::VectorXd unit_disp{
          Eigen::MatrixXd::Identity(problem_size, problem_size).col(i)};
      displacement.eigen_vec() = unit_disp;

      K_operator.apply(tangent, displacement, force);
      reconstructed_stiffness.col(i) = force.eigen_vec();
    }

    Eigen::MatrixXd reference_stiffness{
        Eigen::MatrixXd::Zero(problem_size, problem_size)};
    reference_stiffness << 10.3444, -4.53889, -4.53889, 1.95278, -1.33681,
        -1.24931, 1.95278, -1.24931, -1.33681, -4.53889, 10.3444, -4.53889,
        -1.24931, 1.95278, -1.33681, -1.33681, 1.95278, -1.24931, -4.53889,
        -4.53889, 10.3444, -1.33681, -1.24931, 1.95278, -1.24931, -1.33681,
        1.95278, 1.95278, -1.24931, -1.33681, 10.3444, -4.53889, -4.53889,
        1.95278, -1.33681, -1.24931, -1.33681, 1.95278, -1.24931, -4.53889,
        10.3444, -4.53889, -1.24931, 1.95278, -1.33681, -1.24931, -1.33681,
        1.95278, -4.53889, -4.53889, 10.3444, -1.33681, -1.24931, 1.95278,
        1.95278, -1.33681, -1.24931, 1.95278, -1.24931, -1.33681, 10.3444,
        -4.53889, -4.53889, -1.24931, 1.95278, -1.33681, -1.33681, 1.95278,
        -1.24931, -4.53889, 10.3444, -4.53889, -1.33681, -1.24931, 1.95278,
        -1.24931, -1.33681, 1.95278, -4.53889, -4.53889, 10.3444;

    auto error{muGrid::testGoodies::rel_error(reference_stiffness,
                                              reconstructed_stiffness)};

    Real tol{1e-5};
    BOOST_CHECK_LE(error, tol);
    if (not(error < tol)) {
      std::cout << "reconstructed:" << std::endl
                << reconstructed_stiffness << std::endl;
      std::cout << "reference:" << std::endl
                << reference_stiffness << std::endl;
      std::cout << "ratio:" << std::endl
                << reference_stiffness.array() / reconstructed_stiffness.array()
                << std::endl;
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(recover_fluctuation_test, Fix,
                                   DiscretisationFixtures, Fix) {
    auto && collection{this->get_cell_data()->get_fields()};
    muGrid::MappedTensorField<Real, Mapping::Mut, Fix::DisplacementRank(),
                              Fix::SpatialDim(), IterUnit::SubPt>
        random_input{"Random_input", collection, NodalPtTag};

    muGrid::MappedTensorField<Real, Mapping::Mut, Fix::DisplacementRank(),
                              Fix::SpatialDim(), IterUnit::SubPt>
        response{"response", collection, NodalPtTag};
    muGrid::MappedTensorField<Real, Mapping::Mut, Fix::DisplacementRank(),
                              Fix::SpatialDim(), IterUnit::SubPt>
        recovered_input{"Recovered_input", collection, NodalPtTag};

    random_input.get_field().eigen_vec().setRandom();
    auto && mean{random_input.get_map().mean()};
    for (auto && entry : random_input.get_map()) {
      entry -= mean;
    }

    auto && stiffness_operator{
        this->discretisation.get_stiffness_operator(Fix::DisplacementRank())};
    stiffness_operator.apply(this->material_properties,
                             random_input.get_field(), response.get_field());

    this->greens_operator.apply(response.get_field(),
                                recovered_input.get_field());
    auto && error{muGrid::testGoodies::rel_error(
        random_input.get_field().eigen_vec(),
        recovered_input.get_field().eigen_vec())};

    BOOST_CHECK_LE(error, tol);

    if (not(error <= tol)) {
      std::cout << "response:" << std::endl
                << response.get_field().eigen_pixel() << std::endl;
      std::cout << "<response> = " << response.get_map().mean().transpose()
                << std::endl;
      std::cout << "input:" << std::endl
                << random_input.get_field().eigen_pixel() << std::endl;
      std::cout << "<input> = " << random_input.get_map().mean().transpose()
                << std::endl;
      std::cout << "recovered:" << std::endl
                << recovered_input.get_field().eigen_pixel() << std::endl;
      std::cout << "<recovered> = "
                << recovered_input.get_map().mean().transpose() << std::endl;
      std::cout << "difference:" << std::endl
                << random_input.get_field().eigen_pixel() -
                       recovered_input.get_field().eigen_pixel()
                << std::endl;
    }
  }
  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muSpectre
