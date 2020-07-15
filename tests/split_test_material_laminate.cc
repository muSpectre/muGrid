/**
 * @file   split_test_material_laminate.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   18 May 2020
 *
 * @brief  tests for both MaterialLaminateSS and MaterialLaminateFS
 *
 * Copyright © 2020 Ali Falsafi
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

#include "materials/material_laminate.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/material_linear_orthotropic.hh"
#include "cell/cell.hh"
#include "projection/projection_finite_strain_fast.hh"

#include <libmufft/fftw_engine.hh>

#include <libmugrid/field_collection.hh>
#include <libmugrid/iterators.hh>

#include <type_traits>
#include <boost/mpl/list.hpp>
#include <boost/range/combine.hpp>

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(material_laminate);

  template <Index_t DimS, Index_t DimM, int c1, int c2, Formulation Form,
            bool RndVec = true, bool RndRatio = true>
  struct MaterialFixture {
    constexpr static Index_t sdim{DimS};
    constexpr static Index_t mdim{DimM};
    constexpr static Formulation form{Form};

    using Vec_t = Eigen::Matrix<Real, DimM, 1>;

    using MaterialLam_t =
        std::conditional_t<Form == Formulation::small_strain,
                           MaterialLaminate<DimM, Formulation::small_strain>,
                           MaterialLaminate<DimM, Formulation::finite_strain>>;
    using Material_t = MaterialLinearElastic1<DimM>;
    using Mat_ptr = std::shared_ptr<Material_t>;

    constexpr static Real lambda{2}, mu{1.5};

    MaterialFixture()
        : young{mu * (3 * lambda + 2 * mu) / (lambda + mu)},
          poisson{lambda / (2 * (lambda + mu))},
          mat_precipitate_ptr{std::make_shared<Material_t>(
              "mat1", DimM, 1, c1 * young, poisson)},
          mat_matrix_ptr{std::make_shared<Material_t>("mat2", DimM, 1,
                                                      c2 * young, poisson)},
          mat("Name", DimM, 1), normal_vector_holder{std::make_unique<Vec_t>(
                                    this->normal_vector_maker())},
          normal_vec{*normal_vector_holder}, ratio{ratio_maker()} {
      this->normal_vec = this->normal_vec / this->normal_vec.norm();
    }

    Vec_t normal_vector_maker() {
      if (RndVec) {
        return Vec_t::Random();
      } else {
        return Vec_t::UnitX();
      }
    }

    Real ratio_maker() {
      if (RndRatio) {
        muGrid::testGoodies::RandRange<Real> rng;
        return 0.5 + rng.randval(-0.5, 0.5);
      } else {
        return 0.5;
      }
    }

    constexpr static Index_t NbQuadPts() { return 1; }
    constexpr static Formulation get_form() { return Form; }

   protected:
    Real young;
    Real poisson;
    Mat_ptr mat_precipitate_ptr;
    Mat_ptr mat_matrix_ptr;
    MaterialLam_t mat;
    std::unique_ptr<Vec_t> normal_vector_holder;
    Vec_t & normal_vec{};
    Real ratio;
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t DimM, int c1, int c2, Formulation Form,
            bool RndVec, bool RndRatio>
  constexpr Index_t
      MaterialFixture<DimS, DimM, c1, c2, Form, RndVec, RndRatio>::mdim;

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t DimM, int c1, int c2, Formulation Form,
            bool RndVec, bool RndRatio>
  constexpr Index_t
      MaterialFixture<DimS, DimM, c1, c2, Form, RndVec, RndRatio>::sdim;

  /* ---------------------------------------------------------------------- */

  template <Dim_t DimS, Dim_t DimM, int c1, int c2, Formulation Form,
            bool RndVec = false, bool RndRatio = false>
  struct MaterialFixture_with_ortho_ref
      : public MaterialFixture<DimS, DimM, c1, c2, Form, RndVec, RndRatio> {
    using Parent = MaterialFixture<DimS, DimM, c1, c2, Form, RndVec, RndRatio>;
    using MaterialRef_t = MaterialLinearOrthotropic<DimM>;

    // Constructor
    MaterialFixture_with_ortho_ref()
        : Parent(), material_ref{"Name Ref", DimS, 1,
                                 mat_aniso_c_inp_maker(
                                     c1 * Parent::lambda, c1 * Parent::mu,
                                     c2 * Parent::lambda, c2 * Parent::mu)} {}

    std::vector<Real> mat_aniso_c_inp_maker(Real lambda1, Real mu1,
                                            Real lambda2, Real mu2) {
      Real G1{mu1};
      Real G2{mu2};
      Real young1{lambda1 + 2 * mu1};
      Real young2{lambda2 + 2 * mu2};
      Real poisson1{lambda1 / ((lambda1 + 2 * mu1))};
      Real poisson2{lambda2 / ((lambda2 + 2 * mu2))};

      auto && get_average{[](Real A_1, Real A_2, Real ratio) {
        return ratio * A_1 + (1 - ratio) * A_2;
      }};

      std::vector<Real> ret_val{};
      Real young_avg{1.0 /
                     get_average(1.0 / young1, 1.0 / young2, this->ratio)};
      Real G_inv_avg{1.0 / get_average(1.0 / G1, 1.0 / G2, this->ratio)};
      Real G_avg{get_average(G1, G2, this->ratio)};
      Real G_times_poisson_avg{
          get_average(G1 * poisson1, G2 * poisson2, this->ratio)};
      Real poisson_avg{get_average(poisson1, poisson2, this->ratio)};

      Real young_avg_y{get_average(young1 * (1.0 - poisson1 * poisson1),
                                   young2 * (1.0 - poisson2 * poisson2),
                                   this->ratio) +
                       poisson_avg * poisson1 * young_avg};
      switch (DimS) {
      case twoD:
        ret_val.push_back(young_avg);
        ret_val.push_back(poisson_avg * young_avg);
        ret_val.push_back(young_avg_y);
        ret_val.push_back(G_inv_avg);
        break;
      case threeD:
        ret_val.push_back(young_avg);
        ret_val.push_back(poisson_avg * young_avg);
        ret_val.push_back(poisson_avg * young_avg);
        ret_val.push_back(young_avg_y);
        ret_val.push_back(2 * G_times_poisson_avg +
                          young_avg * poisson_avg * poisson_avg);
        ret_val.push_back(young_avg_y);
        ret_val.push_back(G_avg);
        ret_val.push_back(G_inv_avg);
        ret_val.push_back(G_inv_avg);
        break;
      default:
        break;
      }

      return ret_val;
    }

    MaterialRef_t material_ref;
  };

  /* ---------------------------------------------------------------------- */
  using mat_list = boost::mpl::list<
      MaterialFixture<twoD, twoD, 1, 3, Formulation::finite_strain, true, true>,
      MaterialFixture<threeD, threeD, 7, 3, Formulation::finite_strain, true,
                      true>,
      MaterialFixture<twoD, twoD, 1, 3, Formulation::small_strain, true, true>,
      MaterialFixture<threeD, threeD, 7, 3, Formulation::small_strain, true,
                      true>>;

  using mat_list_identical = boost::mpl::list<
      MaterialFixture<twoD, twoD, 1, 1, Formulation::finite_strain, true, true>,
      MaterialFixture<threeD, threeD, 1, 1, Formulation::finite_strain, true,
                      true>,
      MaterialFixture<twoD, twoD, 1, 1, Formulation::small_strain, true, true>,
      MaterialFixture<threeD, threeD, 1, 1, Formulation::small_strain, true,
                      true>>;

  using mat_list_oriented = boost::mpl::list<
      MaterialFixture_with_ortho_ref<twoD, twoD, 1, 3,
                                     Formulation::finite_strain, false, false>,
      MaterialFixture_with_ortho_ref<threeD, threeD, 1, 3,
                                     Formulation::finite_strain, false, false>,
      MaterialFixture_with_ortho_ref<twoD, twoD, 1, 3,
                                     Formulation::small_strain, false, false>,
      MaterialFixture_with_ortho_ref<threeD, threeD, 1, 3,
                                     Formulation::small_strain, false, false>>;

  using mat_list_twoD = boost::mpl::list<
      MaterialFixture_with_ortho_ref<twoD, twoD, 1, 3,
                                     Formulation::finite_strain, false, false>,
      MaterialFixture_with_ortho_ref<twoD, twoD, 1, 3,
                                     Formulation::small_strain, false, false>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mat_list, Fix) {
    BOOST_CHECK_EQUAL("Name", Fix::mat.get_name());
    auto & mat{Fix::mat};
    // auto sdim{Fix::sdim};
    auto mdim{Fix::mdim};
    // BOOST_CHECK_EQUAL(sdim, mat.sdim());
    BOOST_CHECK_EQUAL(mdim, mat.get_material_dimension());
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_add_pixel, Fix, mat_list, Fix) {
    auto & mat{Fix::mat};
    auto & mat2{Fix::mat_precipitate_ptr};
    auto & mat1{Fix::mat_matrix_ptr};
    auto & ratio{Fix::ratio};
    auto & normal_vec{Fix::normal_vec};
    muGrid::testGoodies::RandRange<size_t> rng;

    const Index_t nb_pixel{7};
    for (Index_t i = 0; i < nb_pixel; ++i) {
      auto && j{rng.randval(0, nb_pixel)};
      BOOST_CHECK_NO_THROW(mat.add_pixel(j, mat1, mat2, ratio, normal_vec));
    }

    BOOST_CHECK_NO_THROW(mat.initialise());
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_identical, Fix, mat_list_identical,
                                   Fix) {
    auto & mat{Fix::mat};
    auto & mat_precipitate{Fix::mat_precipitate_ptr};
    auto & mat_matrix{Fix::mat_matrix_ptr};
    auto & ratio{Fix::ratio};
    auto & normal_vec{Fix::normal_vec};

    const Index_t nb_pixel{1};
    constexpr auto cube{muGrid::CcoordOps::get_cube<Fix::sdim>(nb_pixel)};
    constexpr auto loc{muGrid::CcoordOps::get_cube<Fix::sdim>(Index_t{0})};

    using Mat_t = Eigen::Matrix<Real, Fix::mdim, Fix::mdim>;
    using FC_t = muGrid::GlobalFieldCollection;
    // using Ccoord = Ccoord_t<Fix::sdim>;
    FC_t globalfields{Fix::mdim};
    globalfields.set_nb_sub_pts(QuadPtTag, Fix::NbQuadPts());
    globalfields.initialise(cube, loc);
    muGrid::MappedT2Field<Real, Mapping::Mut, Fix::mdim, IterUnit::SubPt>
        F1_f{"Transformation Gradient 1", globalfields, QuadPtTag};
    muGrid::MappedT2Field<Real, Mapping::Mut, Fix::mdim, IterUnit::SubPt>
        P1_f{"Nominal Stress 1", globalfields, QuadPtTag};
    muGrid::MappedT4Field<Real, Mapping::Mut, Fix::mdim, IterUnit::SubPt>
        K1_f{"Tangent Moduli 1", globalfields, QuadPtTag};

    muGrid::MappedT2Field<Real, Mapping::Mut, Fix::mdim, IterUnit::SubPt>
        F2_f{"Transformation Gradient 2", globalfields, QuadPtTag};
    muGrid::MappedT2Field<Real, Mapping::Mut, Fix::mdim, IterUnit::SubPt>
        P2_f{"Nominal Stress 2", globalfields, QuadPtTag};
    muGrid::MappedT4Field<Real, Mapping::Mut, Fix::mdim, IterUnit::SubPt>
        K2_f{"Tangent Moduli 2", globalfields, QuadPtTag};

    Mat_t zero{Mat_t::Zero()};
    Mat_t F{1e-6 * Mat_t::Random() + Mat_t::Identity()};
    Mat_t strain{F};

    if (Fix::get_form() == Formulation::small_strain) {
      strain = {0.5 * ((F * F.transpose()) - Mat_t::Identity())};
    }

    Index_t pix0{0};
    Real error{0.0};
    Real tol{1e-12};

    F1_f.get_map()[pix0] = strain;
    F2_f.get_map()[pix0] = strain;

    mat.add_pixel(pix0, mat_precipitate, mat_matrix, ratio, normal_vec);
    mat_precipitate->add_pixel(pix0);
    mat.initialise();
    mat_precipitate->initialise();

    mat.compute_stresses_tangent(
        globalfields.get_field("Transformation Gradient 1"),
        globalfields.get_field("Nominal Stress 1"),
        globalfields.get_field("Tangent Moduli 1"), Fix::get_form());
    mat_precipitate->compute_stresses_tangent(
        globalfields.get_field("Transformation Gradient 2"),
        globalfields.get_field("Nominal Stress 2"),
        globalfields.get_field("Tangent Moduli 2"), Fix::get_form());

    error = (P1_f.get_map()[pix0] - P2_f.get_map()[pix0]).norm();
    BOOST_CHECK_LT(error, tol);

    error = (K1_f.get_map()[pix0] - K2_f.get_map()[pix0]).norm();
    BOOST_CHECK_LT(error, tol);

    F1_f.get_map()[pix0] = strain;
    F2_f.get_map()[pix0] = strain;
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_patch_material_laminate, Fix,
                                   mat_list_oriented, Fix) {
    auto && mat{Fix::mat};
    auto && mat_ref{Fix::material_ref};
    auto & mat_precipitate{Fix::mat_precipitate_ptr};
    auto & mat_matrix{Fix::mat_matrix_ptr};
    auto && ratio{Fix::ratio};
    auto && normal_vec{Fix::normal_vec};

    using Mat_t = Eigen::Matrix<Real, Fix::mdim, Fix::mdim>;
    using FC_t = muGrid::GlobalFieldCollection;

    const Index_t nb_pixel{1};
    constexpr auto cube{muGrid::CcoordOps::get_cube<Fix::sdim>(nb_pixel)};
    constexpr auto loc{muGrid::CcoordOps::get_cube<Fix::sdim>(Index_t{0})};

    FC_t globalfields{Fix::mdim};
    globalfields.set_nb_sub_pts(QuadPtTag, Fix::NbQuadPts());
    globalfields.initialise(cube, loc);
    muGrid::MappedT2Field<Real, Mapping::Mut, Fix::mdim, IterUnit::SubPt>
        F1_f{"Transformation Gradient 1", globalfields, QuadPtTag};
    muGrid::MappedT2Field<Real, Mapping::Mut, Fix::mdim, IterUnit::SubPt>
        P1_f{"Nominal Stress 1", globalfields, QuadPtTag};
    muGrid::MappedT4Field<Real, Mapping::Mut, Fix::mdim, IterUnit::SubPt>
        K1_f{"Tangent Moduli 1", globalfields, QuadPtTag};

    muGrid::MappedT2Field<Real, Mapping::Mut, Fix::mdim, IterUnit::SubPt>
        F2_f{"Transformation Gradient 2", globalfields, QuadPtTag};
    muGrid::MappedT2Field<Real, Mapping::Mut, Fix::mdim, IterUnit::SubPt>
        P2_f{"Nominal Stress 2", globalfields, QuadPtTag};
    muGrid::MappedT4Field<Real, Mapping::Mut, Fix::mdim, IterUnit::SubPt>
        K2_f{"Tangent Moduli 2", globalfields, QuadPtTag};

    Mat_t zero{Mat_t::Zero()};
    Mat_t F{1e-6 * Mat_t::Random() + Mat_t::Identity()};
    Mat_t strain{F};

    if (Fix::get_form() == Formulation::small_strain) {
      strain = {0.5 * ((F * F.transpose()) - Mat_t::Identity())};
    }
    // using Ccoord = Ccoord_t<Fix::sdim>;
    Index_t pix0{0};
    Real error{0.0};
    Real tol{1e-12};

    F1_f.get_map()[pix0] = strain;
    F2_f.get_map()[pix0] = strain;

    mat.add_pixel(pix0, mat_precipitate, mat_matrix, ratio, normal_vec);
    mat_ref.add_pixel(pix0);

    mat.initialise();
    mat_ref.initialise();

    F = Mat_t::Identity();
    F1_f.get_map()[pix0] = F;
    F2_f.get_map()[pix0] = F;

    mat.compute_stresses_tangent(
        globalfields.get_field("Transformation Gradient 1"),
        globalfields.get_field("Nominal Stress 1"),
        globalfields.get_field("Tangent Moduli 1"), Fix::get_form());
    mat_ref.compute_stresses_tangent(
        globalfields.get_field("Transformation Gradient 2"),
        globalfields.get_field("Nominal Stress 2"),
        globalfields.get_field("Tangent Moduli 2"), Fix::get_form());

    error = rel_error(P1_f.get_map()[pix0], P2_f.get_map()[pix0]);
    BOOST_CHECK_LT(error, tol);

    error = rel_error(K1_f.get_map()[pix0], K2_f.get_map()[pix0]);
    BOOST_CHECK_LT(error, tol);

    mat.compute_stresses(globalfields.get_field("Transformation Gradient 1"),
                         globalfields.get_field("Nominal Stress 1"),
                         Fix::get_form());
    mat_ref.compute_stresses(
        globalfields.get_field("Transformation Gradient 2"),
        globalfields.get_field("Nominal Stress 2"), Fix::get_form());

    auto a{P1_f.get_map()[pix0]};
    auto b{P2_f.get_map()[pix0]};

    error = rel_error(a, b);
    BOOST_CHECK_LT(error, tol);

    if (Fix::get_form() == Formulation::finite_strain) {
      BOOST_CHECK_THROW(mat.compute_stresses_tangent(
                            globalfields.get_field("Transformation Gradient 1"),
                            globalfields.get_field("Nominal Stress 1"),
                            globalfields.get_field("Tangent Moduli 1"),
                            Formulation::small_strain),
                        std::runtime_error);
      BOOST_CHECK_THROW(mat.compute_stresses(
                            globalfields.get_field("Transformation Gradient 1"),
                            globalfields.get_field("Nominal Stress 1"),
                            Formulation::small_strain),
                        std::runtime_error);
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_patch_material_laminate_precipitate,
                                   Fix, mat_list_twoD, Fix) {
    constexpr size_t sdim{Fix::sdim};

    auto & mat_precipitate{Fix::mat_precipitate_ptr};
    auto & mat_matrix{Fix::mat_matrix_ptr};
    constexpr Index_t grid{15};
    constexpr Real length_pixel{1.0};
    constexpr Real length_cell{grid * length_pixel};

    DynCcoord_t nb_grid_pts{grid, grid};
    DynRcoord_t lengths{length_cell, length_cell};

    auto fft_ptr{std::make_unique<muFFT::FFTWEngine>(nb_grid_pts)};
    auto proj_ptr{std::make_unique<ProjectionFiniteStrainFast<sdim>>(
        std::move(fft_ptr), lengths)};
    Cell sys(std::move(proj_ptr));

    auto & mat_lam = MaterialLaminate<sdim, Formulation::small_strain>::make(
        sys, "laminate");
    auto & mat_precipitate_cell =
        MaterialLinearElastic1<sdim>::make(sys, "preciptiate", 2.5, 0.4);
    auto & mat_matrix_cell =
        MaterialLinearElastic1<sdim>::make(sys, "matrix", 2.5, 0.4);

    Real half_length{4.00};
    Real half_width{4.00};
    Real x_center{length_cell * 0.5};
    Real y_center{length_cell * 0.5};
    Real x_length{length_pixel * half_length};
    Real y_length{length_pixel * half_width};

    auto left_x = static_cast<uint>(x_center - x_length);
    auto right_x = static_cast<uint>(x_center + x_length + 1.00);

    auto bottom_y = static_cast<uint>(y_center - x_length);
    auto top_y = static_cast<uint>(y_center + x_length + 1.00);

    std::vector<DynRcoord_t> precipitate_vertices;
    precipitate_vertices.push_back({x_center - x_length, y_center - y_length});
    precipitate_vertices.push_back({x_center + x_length, y_center - y_length});
    precipitate_vertices.push_back({x_center - x_length, y_center + y_length});
    precipitate_vertices.push_back({x_center + x_length, y_center + y_length});

    sys.make_pixels_precipitate_for_laminate_material(
        precipitate_vertices, mat_lam, mat_precipitate_cell, mat_precipitate,
        mat_matrix);
    sys.complete_material_assignment_simple(mat_matrix_cell);
    sys.initialise();

    BOOST_CHECK_EQUAL(mat_lam.size(),
                      2 * ((top_y - bottom_y) + (right_x - left_x) - 2));

    BOOST_CHECK_EQUAL(mat_precipitate_cell.size(),
                      ((top_y - bottom_y) * (right_x - left_x)) -
                          mat_lam.size());

    BOOST_CHECK_EQUAL(mat_precipitate_cell.size() + mat_matrix_cell.size() +
                          mat_lam.size(),
                      grid * grid);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
