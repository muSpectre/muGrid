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


#include <boost/mpl/list.hpp>
#include <Eigen/Dense>

#include "tests.hh"
#include "common/common.hh"
#include "common/iterators.hh"
#include "common/field_map.hh"
#include "tests/test_goodies.hh"
#include "cell/cell_factory.hh"
#include "materials/material_linear_elastic1.hh"


namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(cell_base);
  template <Dim_t DimS>
  struct Sizes {
  };
  template<>
  struct Sizes<twoD> {
    constexpr static Dim_t sdim{twoD};
    constexpr static Ccoord_t<sdim> get_resolution() {
      return Ccoord_t<sdim>{3, 5};}
    constexpr static Rcoord_t<sdim> get_lengths() {
      return Rcoord_t<sdim>{3.4, 5.8};}
  };
  template<>
  struct Sizes<threeD> {
    constexpr static Dim_t sdim{threeD};
    constexpr static Ccoord_t<sdim> get_resolution() {
      return Ccoord_t<sdim>{3, 5, 7};}
    constexpr static Rcoord_t<sdim> get_lengths() {
      return Rcoord_t<sdim>{3.4, 5.8, 6.7};}
  };

  template <Dim_t DimS, Dim_t DimM, Formulation form>
  struct CellBaseFixture: CellBase<DimS, DimM> {
    constexpr static Dim_t sdim{DimS};
    constexpr static Dim_t mdim{DimM};
    constexpr static Formulation formulation{form};
    CellBaseFixture()
      :CellBase<DimS, DimM>{
      std::move(cell_input<DimS, DimM>(Sizes<DimS>::get_resolution(),
                                         Sizes<DimS>::get_lengths(),
                                         form))} {}
  };

  using fixlist = boost::mpl::list<CellBaseFixture<twoD, twoD,
                                                     Formulation::finite_strain>,
                                   CellBaseFixture<threeD, threeD,
                                                     Formulation::finite_strain>,
                                   CellBaseFixture<twoD, twoD,
                                                     Formulation::small_strain>,
                                   CellBaseFixture<threeD, threeD,
                                                     Formulation::small_strain>>;

  BOOST_AUTO_TEST_CASE(manual_construction) {
    constexpr Dim_t dim{twoD};

    Ccoord_t<dim> resolutions{3, 3};
    Rcoord_t<dim> lengths{2.3, 2.7};
    Formulation form{Formulation::finite_strain};
    auto fft_ptr{std::make_unique<FFTWEngine<dim>>(resolutions, dim*dim)};
    auto proj_ptr{std::make_unique<ProjectionFiniteStrainFast<dim, dim>>(std::move(fft_ptr), lengths)};
    CellBase<dim, dim> sys{std::move(proj_ptr)};

    auto sys2{make_cell<dim, dim>(resolutions, lengths, form)};
    auto sys2b{std::move(sys2)};
    BOOST_CHECK_EQUAL(sys2b.size(), sys.size());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    BOOST_CHECK_THROW(fix::check_material_coverage(), std::runtime_error);
    BOOST_CHECK_THROW(fix::initialise(), std::runtime_error);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(add_material_test, fix, fixlist, fix) {
    constexpr Dim_t dim{fix::sdim};
    using Material_t = MaterialLinearElastic1<dim, dim>;
    auto Material_hard = std::make_unique<Material_t>("hard", 210e9, .33);
    BOOST_CHECK_NO_THROW(fix::add_material(std::move(Material_hard)));
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(simple_evaluation_test, fix, fixlist, fix) {
    constexpr Dim_t dim{fix::sdim};
    constexpr Formulation form{fix::formulation};
    using Mat_t = MaterialLinearElastic1<dim, dim>;
    const Real Young{210e9}, Poisson{.33};
    const Real lambda{Young*Poisson/((1+Poisson)*(1-2*Poisson))};
    const Real mu{Young/(2*(1+Poisson))};
    auto Material_hard = std::make_unique<Mat_t>("hard", Young, Poisson);

    for (auto && pixel: *this) {
      Material_hard->add_pixel(pixel);
    }

    fix::add_material(std::move(Material_hard));
    auto & F = fix::get_strain();
    auto F_map = F.get_map();
    // finite strain formulation expects the deformation gradient F,
    // while small strain expects infinitesimal strain ε
    for (auto grad: F_map) {
      switch (form) {
      case Formulation::finite_strain: {
        grad = grad.Identity();
        break;
      }
      case Formulation::small_strain: {
        grad = grad.Zero();
        break;
      }
      default:
        BOOST_CHECK(false);
        break;
      }
    }

    auto res_tup{fix::evaluate_stress_tangent(F)};
    auto stress{std::get<0>(res_tup).get_map()};
    auto tangent{std::get<1>(res_tup).get_map()};

    auto tup = testGoodies::objective_hooke_explicit
      (lambda, mu, Matrices::I2<dim>());
    auto P_ref = std::get<0>(tup);
    for (auto mat: stress) {
      Real norm = (mat - P_ref).norm();
      BOOST_CHECK_EQUAL(norm, 0.);
    }

    auto tan_ref = std::get<1>(tup);
    for (const auto tan: tangent) {
      Real norm = (tan - tan_ref).norm();
      BOOST_CHECK_EQUAL(norm, 0.);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(evaluation_test, fix, fixlist, fix) {
    constexpr Dim_t dim{fix::sdim};
    using Mat_t = MaterialLinearElastic1<dim, dim>;
    auto Material_hard = std::make_unique<Mat_t>("hard", 210e9, .33);
    auto Material_soft = std::make_unique<Mat_t>("soft",  70e9, .3);

    for (auto && cnt_pixel: akantu::enumerate(*this)) {
      auto counter = std::get<0>(cnt_pixel);
      auto && pixel = std::get<1>(cnt_pixel);
      if (counter < 5) {
        Material_hard->add_pixel(pixel);
      } else {
        Material_soft->add_pixel(pixel);
      }
    }

    fix::add_material(std::move(Material_hard));
    fix::add_material(std::move(Material_soft));

    auto & F = fix::get_strain();
    fix::evaluate_stress_tangent(F);

    fix::evaluate_stress_tangent(F);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(evaluation_test_new_interface, fix, fixlist, fix) {
    constexpr Dim_t dim{fix::sdim};
    using Mat_t = MaterialLinearElastic1<dim, dim>;
    auto Material_hard = std::make_unique<Mat_t>("hard", 210e9, .33);
    auto Material_soft = std::make_unique<Mat_t>("soft",  70e9, .3);

    for (auto && cnt_pixel: akantu::enumerate(*this)) {
      auto counter = std::get<0>(cnt_pixel);
      auto && pixel = std::get<1>(cnt_pixel);
      if (counter < 5) {
        Material_hard->add_pixel(pixel);
      } else {
        Material_soft->add_pixel(pixel);
      }
    }

    fix::add_material(std::move(Material_hard));
    fix::add_material(std::move(Material_soft));

    auto F_vec = fix::get_strain_vector();

    F_vec.setZero();

    fix::evaluate_stress_tangent();

  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_managed_fields, Fix, fixlist, Fix) {
    Cell & dyn_handle{*this};
    CellBase<Fix::sdim, Fix::mdim> & base_handle{*this};

    const std::string name1{"aaa"};
    constexpr size_t nb_comp{5};

    auto new_dyn_array{dyn_handle.get_managed_real_array(name1, nb_comp)};
    BOOST_CHECK_EQUAL(new_dyn_array.rows(), nb_comp);
    BOOST_CHECK_EQUAL(new_dyn_array.cols(), dyn_handle.size());

    BOOST_CHECK_THROW(dyn_handle.get_managed_real_array(name1, nb_comp+1),
                      std::runtime_error);

    auto & new_field{base_handle.get_managed_real_field(name1, nb_comp)};
    BOOST_CHECK_EQUAL(new_field.get_nb_components(), nb_comp);
    BOOST_CHECK_EQUAL(new_field.size(), dyn_handle.size());

    BOOST_CHECK_THROW(base_handle.get_managed_real_field(name1, nb_comp+1),
                      std::runtime_error);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_globalised_fields, Fix, fixlist, Fix) {
    constexpr Dim_t Dim{Fix::sdim};
    using Mat_t = MaterialLinearElastic1<Dim, Dim>;
    using LColl_t = typename Mat_t::MFieldCollection_t;
    auto & material_soft{Mat_t::make(*this, "soft",  70e9, .3)};
    auto & material_hard{Mat_t::make(*this, "hard", 210e9, .3)};

    for (auto && tup: akantu::enumerate(*this)) {
      const auto & i{std::get<0>(tup)};
      const auto & pixel{std::get<1>(tup)};
      if (i%2) {
        material_soft.add_pixel(pixel);
      } else {
        material_hard.add_pixel(pixel);
      }
    }
    material_soft.initialise();
    material_hard.initialise();

    auto & col_soft{material_soft.get_collection()};
    auto & col_hard{material_hard.get_collection()};

    // compatible fields:
    const std::string compatible_name{"compatible"};
    auto & compatible_soft{
      make_field<TypedField<LColl_t, Real>>(compatible_name,
                                            col_soft,
                                            Dim)};
    auto & compatible_hard{
      make_field<TypedField<LColl_t, Real>>(compatible_name,
                                            col_hard,
                                            Dim)};
    auto pixler = [](auto& field) {
      for (auto && tup: field.get_map().enumerate()) {
        const auto & pixel{std::get<0>(tup)};
        auto & val{std::get<1>(tup)};
        for (Dim_t i{0}; i < Dim; ++i) {
          val(i) = pixel[i];
        }
      }
    };
    pixler(compatible_soft);
    pixler(compatible_hard);

    auto & global_compatible_field{
      this->get_globalised_internal_real_field(compatible_name)};

    auto glo_map{global_compatible_field.get_map()};
    for (auto && tup: glo_map.enumerate()) {
      const auto & pixel{std::get<0>(tup)};
      const auto & val(std::get<1>(tup));

      using Map_t = Eigen::Map<const Eigen::Array<Dim_t, Dim, 1>>;
      Real err {(val -
                 Map_t(pixel.data()).template cast<Real>()).matrix().norm()};
      BOOST_CHECK_LT(err, tol);
    }

    // incompatible fields:
    const std::string incompatible_name{"incompatible"};
    make_field<TypedField<LColl_t, Real>>(incompatible_name,
                                          col_soft,
                                          Dim);

    make_field<TypedField<LColl_t, Real>>(incompatible_name,
                                          col_hard,
                                          Dim+1);
    BOOST_CHECK_THROW(this->get_globalised_internal_real_field(incompatible_name),
                      std::runtime_error);

    // wrong name/ inexistant field
    const std::string wrong_name{"wrong_name"};
    BOOST_CHECK_THROW(this->get_globalised_internal_real_field(wrong_name),
                      std::runtime_error);

    // wrong scalar type:
    const std::string wrong_scalar_name{"wrong_scalar"};
    make_field<TypedField<LColl_t, Real>>(wrong_scalar_name,
                                          col_soft,
                                          Dim);

    make_field<TypedField<LColl_t, Dim_t>>(wrong_scalar_name,
                                           col_hard,
                                           Dim);
    BOOST_CHECK_THROW(this->get_globalised_internal_real_field(wrong_scalar_name),
                      std::runtime_error);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
