/**
 * @file   test_cell_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   14 Dec 2017
 *
 * @brief  Tests for the basic cell class
 *
 * Copyright © 2017 Till Junge
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include <boost/mpl/list.hpp>
#include <Eigen/Dense>

#include "tests.hh"
#include "libmugrid/test_goodies.hh"

#include <libmugrid/iterators.hh>
#include <libmugrid/field_typed.hh>
#include <libmugrid/state_field.hh>
#include <libmugrid/field_map.hh>
#include <cell/cell_factory.hh>
#include <materials/material_linear_elastic1.hh>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(cell_base);
  template <Dim_t DimS>
  struct Sizes {};
  template <>
  struct Sizes<twoD> {
    constexpr static Dim_t sdim{twoD};
    static DynCcoord_t get_nb_grid_pts() { return DynCcoord_t{3, 5}; }
    static DynRcoord_t get_lengths() { return DynRcoord_t{3.4, 5.8}; }
  };
  template <>
  struct Sizes<threeD> {
    constexpr static Dim_t sdim{threeD};
    static DynCcoord_t get_nb_grid_pts() { return DynCcoord_t{3, 5, 7}; }
    static DynRcoord_t get_lengths() { return DynRcoord_t{3.4, 5.8, 6.7}; }
  };

  template <Dim_t DimS, Dim_t DimM, Formulation form>
  struct CellFixture : Cell {
    constexpr static Dim_t sdim{DimS};
    constexpr static Dim_t mdim{DimM};
    constexpr static Formulation formulation{form};
    CellFixture()
        : Cell{std::move(cell_input(Sizes<DimS>::get_nb_grid_pts(),
                                    Sizes<DimS>::get_lengths(), form))} {}
  };

  using fixlist =
      boost::mpl::list<CellFixture<twoD, twoD, Formulation::finite_strain>,
                       CellFixture<threeD, threeD, Formulation::finite_strain>,
                       CellFixture<twoD, twoD, Formulation::small_strain>,
                       CellFixture<threeD, threeD, Formulation::small_strain>>;

  BOOST_AUTO_TEST_CASE(manual_construction) {
    constexpr Dim_t dim{twoD};

    DynCcoord_t nb_grid_pts{3, 3};
    DynRcoord_t lengths{2.3, 2.7};
    Formulation form{Formulation::finite_strain};
    auto fft_ptr{std::make_unique<muFFT::FFTWEngine>(nb_grid_pts)};
    auto proj_ptr{std::make_unique<ProjectionFiniteStrainFast<dim>>(
        std::move(fft_ptr), lengths)};
    Cell sys{std::move(proj_ptr)};

    auto sys2{make_cell(nb_grid_pts, lengths, form)};
    auto sys2b{std::move(sys2)};
    BOOST_CHECK_EQUAL(sys2b.get_nb_pixels(), sys.get_nb_pixels());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    BOOST_CHECK_THROW(fix::check_material_coverage(), std::runtime_error);
    BOOST_CHECK_THROW(fix::initialise(), std::runtime_error);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(add_material_test, fix, fixlist, fix) {
    constexpr Dim_t dim{fix::sdim};
    using Material_t = MaterialLinearElastic1<dim>;
    auto Material_hard =
        std::make_unique<Material_t>("hard", dim, OneQuadPt, 210e9, .33);
    BOOST_CHECK_NO_THROW(fix::add_material(std::move(Material_hard)));
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(double_initialisation_test, fix, fixlist,
                                   fix) {
    constexpr Dim_t dim{fix::sdim};
    using Material_t = MaterialLinearElastic1<dim>;
    auto material_hard =
        std::make_unique<Material_t>("hard", dim, OneQuadPt, 210e9, 0.33);
    for (const auto & pixel_id : this->get_pixel_indices()) {
      material_hard->add_pixel(pixel_id);
    }
    fix::add_material(std::move(material_hard));
    fix::initialise();
    BOOST_CHECK_THROW(fix::initialise(), std::runtime_error);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(double_material_initialisation_testt, fix,
                                   fixlist, fix) {
    constexpr Dim_t Dim{fix::sdim};
    using Mat_t = MaterialLinearElastic1<Dim>;
    auto & material_hard{Mat_t::make(*this, "hard", 210e9, .3)};
    for (const auto & pixel_id : this->get_pixel_indices()) {
      material_hard.add_pixel(pixel_id);
    }
    material_hard.initialise();
    BOOST_CHECK_THROW(material_hard.initialise(), std::runtime_error);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(simple_evaluation_test, fix, fixlist, fix) {
    constexpr Dim_t dim{fix::sdim};
    constexpr Formulation form{fix::formulation};
    using Mat_t = MaterialLinearElastic1<dim>;
    const Real Young{210e9}, Poisson{.33};
    const Real lambda{Young * Poisson / ((1 + Poisson) * (1 - 2 * Poisson))};
    const Real mu{Young / (2 * (1 + Poisson))};
    auto Material_hard =
        std::make_unique<Mat_t>("hard", dim, OneQuadPt, Young, Poisson);
    for (const auto & pixel_id : this->get_pixel_indices()) {
      Material_hard->add_pixel(pixel_id);
    }

    fix::add_material(std::move(Material_hard));
    switch (form) {
    case Formulation::finite_strain: {
      this->set_uniform_strain(Eigen::Matrix<Real, dim, dim>::Identity());
      break;
    }
    case Formulation::small_strain: {
      this->set_uniform_strain(Eigen::Matrix<Real, dim, dim>::Zero());
      break;
    }
    default:
      BOOST_CHECK(false);
      break;
    }

    auto res_tup{fix::evaluate_stress_tangent()};
    muGrid::T2FieldMap<Real, Mapping::Const, dim, IterUnit::SubPt> stress{
        std::get<0>(res_tup)};
    muGrid::T4FieldMap<Real, Mapping::Const, dim, IterUnit::SubPt> tangent{
        std::get<1>(res_tup)};

    auto tup = muGrid::testGoodies::objective_hooke_explicit(
        lambda, mu, Matrices::I2<dim>());
    auto P_ref = std::get<0>(tup);
    for (auto mat : stress) {
      Real norm = (mat - P_ref).norm();
      BOOST_CHECK_EQUAL(norm, 0.);
    }

    auto tan_ref = std::get<1>(tup);
    for (const auto tan : tangent) {
      Real norm = (tan - tan_ref).norm();
      BOOST_CHECK_EQUAL(norm, 0.);
    }
  }

  // the following test wants to assure that the order of pixels in different
  // materials is respected
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(two_materials_evaluation_test, fix, fixlist,
                                   fix) {
    constexpr Dim_t dim{fix::sdim};
    constexpr Formulation form{fix::formulation};
    using Mat_t = MaterialLinearElastic1<dim>;
    const Real Young{210e9}, Poisson{.33};
    const Real lambda{Young * Poisson / ((1 + Poisson) * (1 - 2 * Poisson))};
    const Real mu{Young / (2 * (1 + Poisson))};
    const Real contrast{.5};

    auto & material_hard{Mat_t::make(*this, "hard", Young, Poisson)};
    auto & material_soft{Mat_t::make(*this, "soft", contrast * Young, Poisson)};

    for (const auto & pixel_id : this->get_pixel_indices()) {
      static_cast<bool>(pixel_id % 2) ? material_hard.add_pixel(pixel_id)
                                      : material_soft.add_pixel(pixel_id);
    }

    switch (form) {
    case Formulation::finite_strain: {
      this->set_uniform_strain(Eigen::Matrix<Real, dim, dim>::Identity());
      break;
    }
    case Formulation::small_strain: {
      this->set_uniform_strain(Eigen::Matrix<Real, dim, dim>::Zero());
      break;
    }
    default:
      BOOST_CHECK(false);
      break;
    }

    auto res_tup{fix::evaluate_stress_tangent()};
    muGrid::T2FieldMap<Real, Mapping::Const, dim, IterUnit::SubPt> stress{
        std::get<0>(res_tup)};
    muGrid::T4FieldMap<Real, Mapping::Const, dim, IterUnit::SubPt> tangent{
        std::get<1>(res_tup)};

    auto tup_hard{muGrid::testGoodies::objective_hooke_explicit(
        lambda, mu, Matrices::I2<dim>())};
    auto tup_soft{muGrid::testGoodies::objective_hooke_explicit(
        contrast * lambda, contrast * mu, Matrices::I2<dim>())};

    auto P_ref_hard{std::get<0>(tup_hard)};
    auto P_ref_soft{std::get<0>(tup_soft)};
    for (auto id_mat : stress.enumerate_indices()) {
      const auto & id{std::get<0>(id_mat)};
      const auto & mat{std::get<1>(id_mat)};
      Real norm =
          (mat - (static_cast<bool>(id % 2) ? P_ref_hard : P_ref_soft)).norm();
      BOOST_CHECK_EQUAL(norm, 0.);
    }

    auto tan_ref_hard{std::get<1>(tup_hard)};
    auto tan_ref_soft{std::get<1>(tup_soft)};
    for (const auto id_tan : tangent.enumerate_indices()) {
      const auto & id{std::get<0>(id_tan)};
      const auto & tan{std::get<1>(id_tan)};
      Real norm =
          (tan - (static_cast<bool>(id % 2) ? tan_ref_hard : tan_ref_soft))
              .norm();
      BOOST_CHECK_EQUAL(norm, 0.);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(evaluation_test, fix, fixlist, fix) {
    constexpr Dim_t dim{fix::sdim};
    using Mat_t = MaterialLinearElastic1<dim>;
    auto Material_hard =
        std::make_unique<Mat_t>("hard", dim, OneQuadPt, 210e9, .33);
    auto Material_soft =
        std::make_unique<Mat_t>("soft", dim, OneQuadPt, 70e9, .3);

    for (const auto & counter : this->get_pixel_indices()) {
      if (counter < 5) {
        Material_hard->add_pixel(counter);
      } else {
        Material_soft->add_pixel(counter);
      }
    }

    fix::add_material(std::move(Material_hard));
    fix::add_material(std::move(Material_soft));

    fix::evaluate_stress_tangent();

    fix::evaluate_stress_tangent();
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(evaluation_test_new_interface, fix, fixlist,
                                   fix) {
    constexpr Dim_t dim{fix::sdim};
    using Mat_t = MaterialLinearElastic1<dim>;
    auto Material_hard =
        std::make_unique<Mat_t>("hard", dim, OneQuadPt, 210e9, .33);
    auto Material_soft =
        std::make_unique<Mat_t>("soft", dim, OneQuadPt, 70e9, .3);

    for (const auto & counter : this->get_pixel_indices()) {
      if (counter < 5) {
        Material_hard->add_pixel(counter);
      } else {
        Material_soft->add_pixel(counter);
      }
    }

    fix::add_material(std::move(Material_hard));
    fix::add_material(std::move(Material_soft));

    auto F_vec{fix::get_strain().eigen_vec()};

    F_vec.setZero();

    fix::evaluate_stress_tangent();
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_globalised_fields, Fix, fixlist, Fix) {
    constexpr Dim_t Dim{Fix::sdim};
    using Mat_t = MaterialLinearElastic1<Dim>;
    auto & material_soft{Mat_t::make(*this, "soft", 70e9, .3)};
    auto & material_hard{Mat_t::make(*this, "hard", 210e9, .3)};

    for (const auto & pixel_id : this->get_pixel_indices()) {
      if (pixel_id % 2) {
        material_soft.add_pixel(pixel_id);
      } else {
        material_hard.add_pixel(pixel_id);
      }
    }
    material_soft.initialise();
    material_hard.initialise();

    auto & col_soft{material_soft.get_collection()};
    auto & col_hard{material_hard.get_collection()};

    // compatible fields:
    const std::string compatible_name{"compatible"};
    auto & compatible_soft{
        col_soft.register_real_field(compatible_name, 1, QuadPtTag)};
    auto & compatible_hard{
        col_hard.register_real_field(compatible_name, 1, QuadPtTag)};

    auto pixler = [](auto & field) {
      auto map{field.get_sub_pt_map()};
      for (auto && tup : map.enumerate_indices()) {
        const auto & quad_pt_id{std::get<0>(tup)};
        auto & val{std::get<1>(tup)};
        val(0) = quad_pt_id;
      }
    };
    pixler(compatible_soft);
    pixler(compatible_hard);

    auto & global_compatible_field{
        this->globalise_real_internal_field(compatible_name)};

    // make sure we get the same field again
    auto & global_compatible_field_again{
        this->globalise_real_internal_field(compatible_name)};
    BOOST_CHECK_EQUAL(&global_compatible_field, &global_compatible_field_again);

    auto glo_map{global_compatible_field.get_sub_pt_map()};
    for (auto && tup : glo_map.enumerate_indices()) {
      const auto & quad_pt_id{std::get<0>(tup)};
      const auto & val(std::get<1>(tup));

      Real err{(val(0) - quad_pt_id)};
      BOOST_CHECK_LT(err, tol);
    }

    // incompatible fields:
    const std::string incompatible_name{"incompatible"};
    col_soft.register_real_field(incompatible_name, Dim, QuadPtTag);

    col_hard.register_real_field(incompatible_name, Dim + 1,
                                 QuadPtTag);
    BOOST_CHECK_THROW(this->globalise_real_internal_field(incompatible_name),
                      std::runtime_error);

    // wrong name/ inexistant field
    const std::string wrong_name{"wrong_name"};
    BOOST_CHECK_THROW(this->globalise_real_internal_field(wrong_name),
                      std::runtime_error);

    // wrong scalar type:
    const std::string wrong_scalar_name{"wrong_scalar"};
    col_soft.register_real_field(wrong_scalar_name, Dim, QuadPtTag);

    col_hard.register_int_field(wrong_scalar_name, Dim, QuadPtTag);
    BOOST_CHECK_THROW(this->globalise_real_internal_field(wrong_scalar_name),
                      std::runtime_error);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_globalised_current_state_fields, Fix,
                                   fixlist, Fix) {
    constexpr Dim_t Dim{Fix::sdim};
    const size_t nb_steps_to_save{1};
    const size_t nb_dof{1};
    using Mat_t = MaterialLinearElastic1<Dim>;
    auto & material_soft{Mat_t::make(*this, "soft", 70e9, .3)};
    auto & material_hard{Mat_t::make(*this, "hard", 210e9, .3)};

    for (const auto & pixel_id : this->get_pixel_indices()) {
      if (pixel_id % 2) {
        material_soft.add_pixel(pixel_id);
      } else {
        material_hard.add_pixel(pixel_id);
      }
    }
    material_soft.initialise();
    material_hard.initialise();

    auto & col_soft{material_soft.get_collection()};
    auto & col_hard{material_hard.get_collection()};

    // compatible fields:
    const std::string compatible_name{"compatible"};
    auto & compatible_soft{col_soft.register_real_state_field(
        compatible_name, nb_steps_to_save, nb_dof, QuadPtTag)};
    auto & compatible_hard{col_hard.register_real_state_field(
        compatible_name, nb_steps_to_save, nb_dof, QuadPtTag)};

    auto pixler = [](auto & state_field) {
      auto & field_current(
          muGrid::TypedField<Real>::safe_cast(state_field.current()));
      auto map{field_current.get_sub_pt_map()};
      for (auto && tup : map.enumerate_indices()) {
        const auto & quad_pt_id{std::get<0>(tup)};
        auto & val{std::get<1>(tup)};
        val(0) = quad_pt_id;
      }
    };

    pixler(compatible_soft);
    pixler(compatible_hard);

    auto & global_compatible_field_current{
        this->globalise_real_current_field(compatible_name)};

    // make sure we get the same field again
    auto & global_compatible_field_again{
        this->globalise_real_current_field(compatible_name)};
    BOOST_CHECK_EQUAL(&global_compatible_field_current,
                      &global_compatible_field_again);

    auto glo_map_current{global_compatible_field_current.get_sub_pt_map()};

    for (auto && tup : glo_map_current.enumerate_indices()) {
      const auto & quad_pt_id{std::get<0>(tup)};
      const auto & val(std::get<1>(tup));

      Real err{(val(0) - quad_pt_id)};
      BOOST_CHECK_LT(err, tol);
    }

    // incompatible fields:
    const std::string incompatible_name{"incompatible"};
    col_soft.register_real_state_field(incompatible_name, nb_steps_to_save, Dim,
                                       QuadPtTag);

    col_hard.register_real_state_field(incompatible_name, nb_steps_to_save,
                                       Dim + 1, QuadPtTag);
    BOOST_CHECK_THROW(this->globalise_real_current_field(incompatible_name),
                      std::runtime_error);

    // wrong name/ inexistant field
    const std::string wrong_name{"wrong_name"};
    BOOST_CHECK_THROW(this->globalise_real_current_field(wrong_name),
                      std::runtime_error);

    // wrong scalar type:
    const std::string wrong_scalar_name{"wrong_scalar"};
    col_soft.register_real_state_field(wrong_scalar_name, nb_steps_to_save, Dim,
                                       QuadPtTag);

    col_hard.register_int_state_field(wrong_scalar_name, nb_steps_to_save, Dim,
                                      QuadPtTag);
    BOOST_CHECK_THROW(this->globalise_real_current_field(wrong_scalar_name),
                      std::runtime_error);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_globalised_old_state_fields, Fix,
                                   fixlist, Fix) {
    constexpr Dim_t Dim{Fix::sdim};
    const size_t nb_steps_to_save{1};
    const size_t nb_dof{1};
    using Mat_t = MaterialLinearElastic1<Dim>;
    auto & material_soft{Mat_t::make(*this, "soft", 70e9, .3)};
    auto & material_hard{Mat_t::make(*this, "hard", 210e9, .3)};

    for (const auto & pixel_id : this->get_pixel_indices()) {
      if (pixel_id % 2) {
        material_soft.add_pixel(pixel_id);
      } else {
        material_hard.add_pixel(pixel_id);
      }
    }
    material_soft.initialise();
    material_hard.initialise();

    auto & col_soft{material_soft.get_collection()};
    auto & col_hard{material_hard.get_collection()};

    // compatible fields:
    const std::string compatible_name{"compatible"};
    auto & compatible_soft{col_soft.register_real_state_field(
        compatible_name, nb_steps_to_save, nb_dof, QuadPtTag)};
    auto & compatible_hard{col_hard.register_real_state_field(
        compatible_name, nb_steps_to_save, nb_dof, QuadPtTag)};

    auto pixler = [](auto & state_field) {
      auto & field_old(
          muGrid::TypedField<Real>::safe_cast(state_field.current()));
      auto map{field_old.get_sub_pt_map()};
      for (auto && tup : map.enumerate_indices()) {
        const auto & quad_pt_id{std::get<0>(tup)};
        auto & val{std::get<1>(tup)};
        val(0) = quad_pt_id;
      }
    };

    pixler(compatible_soft);
    pixler(compatible_hard);

    compatible_hard.cycle();
    compatible_soft.cycle();

    auto & global_compatible_field_old{
        this->globalise_real_old_field(compatible_name, nb_steps_to_save)};

    // make sure we get the same field again
    auto & global_compatible_field_again{
        this->globalise_real_old_field(compatible_name, nb_steps_to_save)};
    BOOST_CHECK_EQUAL(&global_compatible_field_old,
                      &global_compatible_field_again);

    auto glo_map_old{global_compatible_field_old.get_sub_pt_map()};

    for (auto && tup : glo_map_old.enumerate_indices()) {
      const auto & quad_pt_id{std::get<0>(tup)};
      const auto & val(std::get<1>(tup));

      Real err{(val(0) - quad_pt_id)};
      BOOST_CHECK_LT(err, tol);
    }

    // incompatible fields:
    const std::string incompatible_name{"incompatible"};
    col_soft.register_real_state_field(incompatible_name, nb_steps_to_save, Dim,
                                       QuadPtTag);
    col_hard.register_real_state_field(incompatible_name, nb_steps_to_save,
                                       Dim + 1, QuadPtTag);

    BOOST_CHECK_THROW(this->globalise_real_old_field(incompatible_name, 1),
                      std::runtime_error);

    // wrong name/ inexistant field
    const std::string wrong_name{"wrong_name"};
    BOOST_CHECK_THROW(this->globalise_real_old_field(wrong_name, 1),
                      std::runtime_error);

    // wrong scalar type:
    const std::string wrong_scalar_name{"wrong_scalar"};
    col_soft.register_real_state_field(wrong_scalar_name, nb_steps_to_save, Dim,
                                       QuadPtTag);
    col_hard.register_int_state_field(wrong_scalar_name, nb_steps_to_save, Dim,
                                      QuadPtTag);

    BOOST_CHECK_THROW(this->globalise_real_old_field(wrong_scalar_name, 1),
                      std::runtime_error);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
