/**
 * @file   test_material_linear_diffusion.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   24 Jun 2020
 *
 * @brief  Tests for MaterialLinearDiffusion
 *
 * Copyright © 2020 Till Junge
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

#include "materials/material_linear_diffusion.hh"

#include <libmugrid/mapped_field.hh>
#include <libmugrid/iterators.hh>

#include <boost/mpl/list.hpp>
#include <Eigen/Dense>

namespace muSpectre {
  namespace testGoodies = muGrid::testGoodies;

  template <Index_t DimM>
  struct Fixture : public MaterialLinearDiffusion<DimM> {
    using Parent = MaterialLinearDiffusion<DimM>;
    using Parent::Parent;
    constexpr static Index_t NbQuadPts{1};
    constexpr static Index_t SpatialDim{DimM};
  };

  template <Index_t DimM>
  constexpr Index_t Fixture<DimM>::NbQuadPts;
  template <Index_t DimM>
  constexpr Index_t Fixture<DimM>::SpatialDim;

  template <Index_t DimM>
  struct IsotropicFixture : public Fixture<DimM> {
    using Parent = Fixture<DimM>;
    IsotropicFixture()
        : Parent{"unimportant name", Parent::SpatialDim, Parent::NbQuadPts,
                 DiffVal} {}
    constexpr static Real DiffVal{3.};
    static Eigen::Matrix<Real, DimM, DimM> DiffCoeff() {
      return DiffVal * Eigen::Matrix<Real, DimM, DimM>::Identity();
    }
  };

  template <Index_t DimM>
  constexpr Real IsotropicFixture<DimM>::DiffVal;

  template <Index_t DimM>
  struct AnisotropicFixture : public Fixture<DimM> {
    using Parent = Fixture<DimM>;
    AnisotropicFixture()
        : Parent{"unimportant name", Parent::SpatialDim, Parent::NbQuadPts,
                 DiffCoeff()} {}
    static Eigen::MatrixXd DiffCoeff() {
      using M_t = Eigen::Matrix<Real, DimM, DimM>;
      auto symm{[](M_t mat) -> M_t { return mat + mat.transpose(); }};
      static M_t retval{
          .5 * (symm(M_t::Random() + M_t::Ones()) + 4 * M_t::Identity())};
      return retval;
    }
  };
  using MatList =
      boost::mpl::list<IsotropicFixture<twoD>, IsotropicFixture<threeD>,
                       AnisotropicFixture<twoD>, AnisotropicFixture<threeD>>;

  BOOST_AUTO_TEST_SUITE(linear_diffusion_material_tests);

  BOOST_AUTO_TEST_CASE(negative_coeff) {
    constexpr auto Dim{twoD};
    constexpr auto NbQuad{OneQuadPt};
    constexpr Real NegVal{-1.2};
    using Material_t = MaterialLinearDiffusion<twoD>;
    BOOST_CHECK_THROW(Material_t("unimportant name", Dim, NbQuad, NegVal),
                      MaterialError);
  }

  BOOST_AUTO_TEST_CASE(not_pos_dev_matrix) {
    constexpr auto Dim{twoD};
    constexpr auto NbQuad{OneQuadPt};

    // construct a matrix with a negative eigenvalue:
    using Vec_t = Eigen::Matrix<Real, Dim, 1>;
    using Mat_t = Eigen::Matrix<Real, Dim, Dim>;
    Vec_t eig_vals{(Vec_t{} << 1.2, -1.3).finished()};
    Vec_t vec{Vec_t::Random()};
    vec /= vec.norm();
    Mat_t eig_vecs{};
    eig_vecs.col(0) = vec;
    eig_vecs.col(1) << -vec(1), vec(0);
    Mat_t diffusion{eig_vecs * Eigen::DiagonalMatrix<Real, Dim>{eig_vals} *
                    eig_vecs.transpose()};
    using Material_t = MaterialLinearDiffusion<twoD>;
    BOOST_CHECK_THROW(Material_t("unimportant name", Dim, NbQuad, diffusion),
                      MaterialError);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(construction_test, Fix, MatList, Fix) {}

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(eval_single_quad_test, Fix, MatList, Fix) {
    using Strain_t = Eigen::Matrix<Real, Fix::SpatialDim, 1>;
    using Tangent_t = Eigen::Matrix<Real, Fix::SpatialDim, Fix::SpatialDim>;
    Strain_t random_grad{Strain_t::Random()};

    constexpr Index_t DummyQuadPtId{};
    Strain_t flux_answer{Fix::evaluate_stress(random_grad, DummyQuadPtId)};
    Strain_t flux_reference{Fix::DiffCoeff() * random_grad};

    Real error{testGoodies::rel_error(flux_answer, flux_reference)};
    BOOST_CHECK_LE(error, tol);

    auto && flux_diffusivity{
        Fix::evaluate_stress_tangent(random_grad, DummyQuadPtId)};
    flux_answer = std::get<0>(flux_diffusivity);
    Tangent_t diffusivity_answer{std::get<1>(flux_diffusivity)};
    Tangent_t diffusivity_reference{Fix::DiffCoeff()};

    error = testGoodies::rel_error(flux_answer, flux_reference);
    BOOST_CHECK_LE(error, tol);
    error = testGoodies::rel_error(diffusivity_answer, diffusivity_reference);
    BOOST_CHECK_LE(error, tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(compute_stresses_test, Fix, MatList, Fix) {
    using Strain_t = Eigen::Matrix<Real, Fix::SpatialDim, 1>;
    using Tangent_t = Eigen::Matrix<Real, Fix::SpatialDim, Fix::SpatialDim>;
    Strain_t random_grad{Strain_t::Random()};

    // make a 2×2[×2] grid
    muGrid::GlobalFieldCollection collection{
        Fix::SpatialDim,
        muGrid::CcoordOps::get_cube(Fix::SpatialDim, 2),
        muGrid::CcoordOps::get_cube(Fix::SpatialDim, 2),
        {},
        {{QuadPtTag, Fix::NbQuadPts}}};

    for (auto && pix_id : collection.get_pixel_indices()) {
      Fix::add_pixel(pix_id);
    }
    Fix::initialise();

    muGrid::MappedT1Field<Real, Mapping::Mut, Fix::SpatialDim, IterUnit::SubPt>
        gradients{"gradients", collection, QuadPtTag};
    muGrid::MappedT1Field<Real, Mapping::Mut, Fix::SpatialDim, IterUnit::SubPt>
        fluxes{"fluxes", collection, QuadPtTag};
    muGrid::MappedT2Field<Real, Mapping::Mut, Fix::SpatialDim, IterUnit::SubPt>
        tangents{"tangents", collection, QuadPtTag};

    gradients.get_map() = random_grad;

    Fix::compute_stresses(gradients.get_field(), fluxes.get_field());

    Strain_t flux_reference{Fix::DiffCoeff() * random_grad};
    for (auto && quad_id_flux_answer : fluxes.get_map().enumerate_indices()) {
      muGrid::TypedField<Real> & field{fluxes.get_field()};
      auto && quad_id{std::get<0>(quad_id_flux_answer)};
      auto && flux_answer{std::get<1>(quad_id_flux_answer)};
      auto && error{testGoodies::rel_error(flux_answer, flux_reference)};
      if (not(error < tol)) {
        std::cout << field.get_name();
        std::cout << "At quadrature point " << quad_id << std::endl
                  << "flux_answer: " << flux_answer.transpose() << std::endl;
        std::cout << "flux_reference: " << flux_reference.transpose()
                  << std::endl;
      }
      BOOST_CHECK_LE(error, tol);
    }

    Fix::compute_stresses_tangent(gradients.get_field(), fluxes.get_field(),
                                  tangents.get_field());
    Tangent_t diffusivity_reference{Fix::DiffCoeff()};
    for (auto && tup : akantu::zip(fluxes, tangents)) {
      auto && flux_answer{std::get<0>(tup)};
      auto && diffusivity_answer{std::get<1>(tup)};
      Real error{testGoodies::rel_error(flux_answer, flux_reference)};
      BOOST_CHECK_LE(error, tol);
      error = testGoodies::rel_error(diffusivity_answer, diffusivity_reference);
      BOOST_CHECK_LE(error, tol);
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
