/**
 * @file   split_test_material_laminate.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   21 Jun 2019
 *
 * @brief Tests for the objective Hooke's law with eigenstrains,
 *        (tests that do not require add_pixel are integrated into
 *        `split_test_material_laminate.cc`
 *
 * @section LICENSE
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */
#include "tests.hh"
#include "materials/material_laminate.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/material_linear_orthotropic.hh"
#include "projection/projection_finite_strain_fast.hh"

#include "libmugrid/test_goodies.hh"
#include "libmufft/fftw_engine.hh"
#include "libmugrid/field_collection.hh"
#include "libmugrid/iterators.hh"


#include <type_traits>
#include <boost/mpl/list.hpp>
#include <boost/range/combine.hpp>

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(material_laminate);

  template <Dim_t DimS, Dim_t DimM, int c1, int c2, bool RndVec = true,
            bool RndRatio = true>
  struct MaterialFixture {
    constexpr static Dim_t sdim{DimS};
    constexpr static Dim_t mdim{DimM};

    using Vec_t = Eigen::Matrix<Real, DimM, 1>;
    using MaterialLam_t = MaterialLaminate<DimS, DimM>;
    using Material_t = MaterialLinearElastic1<DimS, DimM>;
    using Mat_ptr = std::shared_ptr<Material_t>;

    constexpr static Real lambda{2}, mu{1.5};

    MaterialFixture()
        : mat("Name"), young{mu * (3 * lambda + 2 * mu) / (lambda + mu)},
          poisson{lambda / (2 * (lambda + mu))},
          mat_precipitate_ptr{
              std::make_shared<Material_t>("mat1", c1 * young, poisson)},
          mat_matrix_ptr{
              std::make_shared<Material_t>("mat2", c2 * young, poisson)},
          normal_vec{normal_vector_maker()}, ratio{ratio_maker()} {
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

    MaterialLam_t mat;
    Real young;
    Real poisson;
    Mat_ptr mat_precipitate_ptr;
    Mat_ptr mat_matrix_ptr;
    Vec_t normal_vec{};
    Real ratio;
  };

  template <Dim_t DimS, Dim_t DimM, int c1, int c2, bool RndVec = false,
            bool RndRatio = false>
  struct MaterialFixture_with_ortho_ref
      : public MaterialFixture<DimS, DimM, c1, c2, RndVec, RndRatio> {
    using Parent = MaterialFixture<DimS, DimM, c1, c2, RndVec, RndRatio>;
    using MaterialRef_t = MaterialLinearOrthotropic<DimS, DimM>;
    // Constructor
    MaterialFixture_with_ortho_ref()
        : Parent(), material_ref{"Name Ref",
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

  using mat_list = boost::mpl::list<MaterialFixture<twoD, twoD, 1, 3>,
                                    MaterialFixture<threeD, threeD, 7, 3>>;

  using mat_list_identical =
      boost::mpl::list<MaterialFixture<twoD, twoD, 1, 1>,
                       MaterialFixture<threeD, threeD, 1, 1>>;

  using mat_list_oriented = boost::mpl::list<
      MaterialFixture_with_ortho_ref<twoD, twoD, 1, 3, false, false>,
      MaterialFixture_with_ortho_ref<threeD, threeD, 1, 3, false, false>>;

  using mat_list_twoD = boost::mpl::list<
      MaterialFixture_with_ortho_ref<twoD, twoD, 1, 3, false, false>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mat_list, Fix) {
    BOOST_CHECK_EQUAL("Name", Fix::mat.get_name());
    auto & mat{Fix::mat};
    auto sdim{Fix::sdim};
    auto mdim{Fix::mdim};
    BOOST_CHECK_EQUAL(sdim, mat.sdim());
    BOOST_CHECK_EQUAL(mdim, mat.mdim());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_add_pixel, Fix, mat_list, Fix) {
    auto & mat{Fix::mat};
    auto & mat2{Fix::mat_precipitate_ptr};
    auto & mat1{Fix::mat_matrix_ptr};
    auto & ratio{Fix::ratio};
    auto & normal_vec{Fix::normal_vec};
    constexpr Dim_t sdim{Fix::sdim};
    muGrid::testGoodies::RandRange<size_t> rng;

    const Dim_t nb_pixel{7}, box_size{17};
    using Ccoord = Ccoord_t<sdim>;
    for (Dim_t i = 0; i < nb_pixel; ++i) {
      Ccoord c{};
      for (Dim_t j = 0; j < sdim; ++j) {
        c[j] = rng.randval(0, box_size);
      }
      BOOST_CHECK_NO_THROW(mat.add_pixel(c, mat1, mat2, ratio, normal_vec));
    }

    BOOST_CHECK_NO_THROW(mat.initialise());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_identical, Fix, mat_list_identical,
                                   Fix) {
    auto & mat{Fix::mat};
    auto & mat_precipitate{Fix::mat_precipitate_ptr};
    auto & mat_matrix{Fix::mat_matrix_ptr};
    auto & ratio{Fix::ratio};
    auto & normal_vec{Fix::normal_vec};

    const Dim_t nb_pixel{1};
    constexpr auto cube{muGrid::CcoordOps::get_cube<Fix::sdim>(nb_pixel)};
    constexpr auto loc{muGrid::CcoordOps::get_cube<Fix::sdim>(0)};

    using Mat_t = Eigen::Matrix<Real, Fix::mdim, Fix::mdim>;
    using FC_t = muGrid::GlobalFieldCollection<Fix::sdim>;
    using Ccoord = Ccoord_t<Fix::sdim>;
    FC_t globalfields{};
    auto & F1_f{muGrid::make_field<typename Fix::MaterialLam_t::StrainField_t>(
        "Transformation Gradient1", globalfields)};
    auto & P1_f{muGrid::make_field<typename Fix::MaterialLam_t::StressField_t>(
        "Nominal Stress1", globalfields)};  // to be computed alone
    auto & K1_f{muGrid::make_field<typename Fix::MaterialLam_t::TangentField_t>(
        "Tangent Moduli1", globalfields)};  // to be computed with tangent

    auto & F2_f{muGrid::make_field<typename Fix::Material_t::StrainField_t>(
        "Transformation Gradient2", globalfields)};
    auto & P2_f{muGrid::make_field<typename Fix::Material_t::StressField_t>(
        "Nominal Stress2", globalfields)};  // to be computed alone
    auto & K2_f{muGrid::make_field<typename Fix::Material_t::TangentField_t>(
        "Tangent Moduli2", globalfields)};  // to be computed with tangent

    globalfields.initialise(cube, loc);

    Mat_t zero{Mat_t::Zero()};
    Mat_t F{Mat_t::Random() / 10000 + Mat_t::Identity()};
    Mat_t strain{0.5 * ((F * F.transpose()) - Mat_t::Identity())};

    Ccoord pix0{0};
    Real error{0.0};
    Real tol{1e-12};

    F1_f.get_map()[pix0] = F;
    F2_f.get_map()[pix0] = F;

    mat.add_pixel(pix0, mat_precipitate, mat_matrix, ratio, normal_vec);
    mat_precipitate->add_pixel(pix0);

    mat.compute_stresses_tangent(F1_f, P1_f, K1_f, Formulation::finite_strain);
    mat_precipitate->compute_stresses_tangent(F2_f, P2_f, K2_f,
                                              Formulation::finite_strain);

    error = (P1_f.get_map()[pix0] - P2_f.get_map()[pix0]).norm();
    BOOST_CHECK_LT(error, tol);

    error = (K1_f.get_map()[pix0] - K2_f.get_map()[pix0]).norm();
    BOOST_CHECK_LT(error, tol);

    F1_f.get_map()[pix0] = strain;
    F2_f.get_map()[pix0] = strain;

    mat.compute_stresses_tangent(F1_f, P1_f, K1_f, Formulation::small_strain);
    mat_precipitate->compute_stresses_tangent(F2_f, P2_f, K2_f,
                                              Formulation::small_strain);

    error = (P1_f.get_map()[pix0] - P2_f.get_map()[pix0]).norm();
    BOOST_CHECK_LT(error, tol);

    error = (K1_f.get_map()[pix0] - K2_f.get_map()[pix0]).norm();
    BOOST_CHECK_LT(error, tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_patch_material_laminate, Fix,
                                   mat_list_oriented, Fix) {
    auto && mat{Fix::mat};
    auto && mat_ref{Fix::material_ref};
    auto & mat_precipitate{Fix::mat_precipitate_ptr};
    auto & mat_matrix{Fix::mat_matrix_ptr};
    auto && ratio{Fix::ratio};
    auto && normal_vec{Fix::normal_vec};

    using Mat_t = Eigen::Matrix<Real, Fix::mdim, Fix::mdim>;
    using FC_t = muGrid::GlobalFieldCollection<Fix::sdim>;

    const Dim_t nb_pixel{1};
    constexpr auto cube{muGrid::CcoordOps::get_cube<Fix::sdim>(nb_pixel)};
    constexpr auto loc{muGrid::CcoordOps::get_cube<Fix::sdim>(0)};

    FC_t globalfields{};
    auto & F1_f{muGrid::make_field<typename Fix::MaterialLam_t::StrainField_t>(
        "Transformation Gradient1", globalfields)};
    auto & P1_f{muGrid::make_field<typename Fix::MaterialLam_t::StressField_t>(
        "Nominal Stress1", globalfields)};  // to be computed alone
    auto & K1_f{muGrid::make_field<typename Fix::MaterialLam_t::TangentField_t>(
        "Tangent Moduli1", globalfields)};  // to be computed with tangent

    auto & F2_f{muGrid::make_field<typename Fix::Material_t::StrainField_t>(
        "Transformation Gradient2", globalfields)};
    auto & P2_f{muGrid::make_field<typename Fix::Material_t::StressField_t>(
        "Nominal Stress2", globalfields)};  // to be computed alone
    auto & K2_f{muGrid::make_field<typename Fix::Material_t::TangentField_t>(
        "Tangent Moduli2", globalfields)};  // to be computed with tangent

    globalfields.initialise(cube, loc);

    Mat_t zero{Mat_t::Zero()};
    Mat_t F{1e-6 * Mat_t::Random() + Mat_t::Identity()};
    Mat_t strain{0.5 * ((F * F.transpose()) - Mat_t::Identity())};

    using Ccoord = Ccoord_t<Fix::sdim>;
    Ccoord pix0{0};
    Real error{0.0};
    Real tol{1e-12};

    F1_f.get_map()[pix0] = strain;
    F2_f.get_map()[pix0] = strain;

    mat.compute_stresses_tangent(F1_f, P1_f, K1_f, Formulation::small_strain);
    mat_ref.compute_stresses_tangent(F2_f, P2_f, K2_f,
                                     Formulation::small_strain);

    error = rel_error(P1_f.get_map()[pix0], P2_f.get_map()[pix0]);
    BOOST_CHECK_LT(error, tol);

    error = rel_error(K1_f.get_map()[pix0], K2_f.get_map()[pix0]);
    BOOST_CHECK_LT(error, tol);

    mat.compute_stresses(F1_f, P1_f, Formulation::small_strain);
    mat_ref.compute_stresses(F2_f, P2_f, Formulation::small_strain);

    error = rel_error(P1_f.get_map()[pix0], P2_f.get_map()[pix0]);
    BOOST_CHECK_LT(error, tol);

    F = Mat_t::Identity();
    F1_f.get_map()[pix0] = F;
    F2_f.get_map()[pix0] = F;

    mat.add_pixel(pix0, mat_precipitate, mat_matrix, ratio, normal_vec);
    mat_ref.add_pixel(pix0);

    mat.compute_stresses_tangent(F1_f, P1_f, K1_f, Formulation::finite_strain);
    mat_ref.compute_stresses_tangent(F2_f, P2_f, K2_f,
                                     Formulation::finite_strain);

    error = rel_error(P1_f.get_map()[pix0], P2_f.get_map()[pix0]);
    BOOST_CHECK_LT(error, tol);

    error = rel_error(K1_f.get_map()[pix0], K2_f.get_map()[pix0]);
    BOOST_CHECK_LT(error, tol);

    mat.compute_stresses(F1_f, P1_f, Formulation::finite_strain);
    mat_ref.compute_stresses(F2_f, P2_f, Formulation::finite_strain);

    error = rel_error(P1_f.get_map()[pix0], P2_f.get_map()[pix0]);
    BOOST_CHECK_LT(error, tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_patch_material_laminate_precipitate,
                                   Fix, mat_list_twoD, Fix) {
    constexpr size_t sdim{Fix::sdim};
    using Rcoord = Rcoord_t<sdim>;
    using Ccoord = Ccoord_t<sdim>;

    auto & mat_precipitate{Fix::mat_precipitate_ptr};
    auto & mat_matrix{Fix::mat_matrix_ptr};
    constexpr Dim_t grid{15};
    constexpr Real length_pixel{1.0};
    constexpr Real length_cell{grid * length_pixel};

    constexpr Ccoord nb_grid_pts{grid, grid};
    constexpr Rcoord lengths{length_cell, length_cell};

    auto fft_ptr{std::make_unique<muFFT::FFTWEngine<sdim>>(
        nb_grid_pts, muGrid::ipow(sdim, 2))};
    auto proj_ptr{std::make_unique<ProjectionFiniteStrainFast<sdim, sdim>>(
        std::move(fft_ptr), lengths)};
    CellBase<sdim, sdim> sys(std::move(proj_ptr));

    auto & mat_lam = MaterialLaminate<sdim, sdim>::make(sys, "lamiante");
    auto & mat_precipitate_cell =
        MaterialLinearElastic1<sdim, sdim>::make(sys, "matrix", 2.5, 0.4);
    auto & mat_matrix_cell =
        MaterialLinearElastic1<sdim, sdim>::make(sys, "matrix", 2.5, 0.4);

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

    std::vector<Rcoord> precipitate_vertices;
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
