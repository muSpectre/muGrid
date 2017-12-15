/**
 * file   test_system_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   14 Dec 2017
 *
 * @brief  Tests for the basic system class
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */


#include <boost/mpl/list.hpp>
#include <Eigen/Dense>

#include "tests.hh"
#include "common/common.hh"
#include "common/iterators.hh"
#include "common/field_map.hh"
#include "common/test_goodies.hh"
#include "system/system_factory.hh"
#include "materials/material_hyper_elastic1.hh"


namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(system_base);
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

  template <Dim_t DimS, Dim_t DimM>
  struct SystemBaseFixture: SystemBase<DimS, DimM> {
    constexpr static Dim_t sdim{DimS};
    constexpr static Dim_t mdim{DimM};
    SystemBaseFixture()
      :SystemBase<DimS, DimM>{
      make_system<DimS, DimM>(Sizes<DimS>::get_resolution(),
                              Sizes<DimS>::get_lengths())} {}
  };

  using fixlist = boost::mpl::list<SystemBaseFixture<twoD, twoD>,
                                   SystemBaseFixture<threeD, threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    BOOST_CHECK_THROW(fix::check_material_coverage(), std::runtime_error);
    BOOST_CHECK_THROW(fix::initialise(), std::runtime_error);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(add_material_test, fix, fixlist, fix) {
    constexpr Dim_t dim{fix::sdim};
    using Material_t = MaterialHyperElastic1<dim, dim>;
    auto Material_hard = std::make_unique<Material_t>("hard", 210e9, .33);
    BOOST_CHECK_NO_THROW(fix::add_material(std::move(Material_hard)));
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(simple_evaluation_test, fix, fixlist, fix) {
    using fc = typename fix::FieldCollection_t;
    constexpr Dim_t dim{fix::sdim};
    auto & F = fix::get_strain();
    MatrixFieldMap<fc, Real, dim, dim> F_map(F);
    for (auto grad: F_map) {
      grad = grad.Identity();
    }
    BOOST_CHECK_THROW(fix::evaluate_stress_tangent(F), std::runtime_error);
    using Mat_t = MaterialHyperElastic1<dim, dim>;
    const Real Young{210e9}, Poisson{.33};
    const Real lambda{Young*Poisson/((1+Poisson)*(1-2*Poisson))};
    const Real mu{Young/(2*(1+Poisson))};
    auto Material_hard = std::make_unique<Mat_t>("hard", Young, Poisson);

    for (auto && pixel: *this) {
      Material_hard->add_pixel(pixel);
    }

    fix::add_material(std::move(Material_hard));

    auto res_tup{fix::evaluate_stress_tangent(F)};
    MatrixFieldMap<fc, Real, dim, dim, true> stress(std::get<0>(res_tup));
    T4MatrixFieldMap<fc, Real, dim, true> tangent{std::get<1>(res_tup)};
    auto tup = testGoodies::objective_hooke_explicit
      (lambda, mu, Matrices::Tens2_t<dim>::Identity());
    for (auto mat: stress) {
      Real norm = (mat - std::get<0>(tup)).norm();
      BOOST_CHECK_EQUAL(norm, 0.);
    }

    for (auto tan: tangent) {
      Real norm = (tan - std::get<1>(tup)).norm();
      BOOST_CHECK_EQUAL(norm, 0.);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(evaluation_test, fix, fixlist, fix) {
    auto & F = fix::get_strain();
    BOOST_CHECK_THROW(fix::evaluate_stress_tangent(F), std::runtime_error);
    constexpr Dim_t dim{fix::sdim};
    using Mat_t = MaterialHyperElastic1<dim, dim>;
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

    fix::evaluate_stress_tangent(F);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
