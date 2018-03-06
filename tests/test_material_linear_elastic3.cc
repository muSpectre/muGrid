/**
 * @file   test_material_linear_elastic3.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   21 Feb 2018
 *
 * @brief  description
 *
 * Copyright © 2018 Richard Leute
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
#include "tests.hh"
#include "materials/material_linear_elastic3.hh"
#include "materials/materials_toolbox.hh"
#include "common/T4_map_proxy.hh"
#include "cmath"


namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(material_linear_elastic_3);


  template <class Mat_t>
  struct MaterialFixture {
    using Material_t = Mat_t;
    Material_t mat;
    MaterialFixture():mat("name"){
      mat.add_pixel({0}, Young, Poisson);
    }
    Real Young{10};
    Real Poisson{0.3};
  };

  using mat_list = boost::mpl::list<
    MaterialFixture<MaterialLinearElastic3<twoD, twoD>>,
    MaterialFixture<MaterialLinearElastic3<twoD, threeD>>,
    MaterialFixture<MaterialLinearElastic3<threeD, threeD>>
    >;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mat_list, Fix) {

  };

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_response, Fix, mat_list, Fix) {
    constexpr Dim_t Dim{Fix::Material_t::Parent::Parent::sdim()};
    Eigen::Matrix<Real, Dim, Dim>E;
    E.setZero();
    E(0,0) = 0.001;
    E(1,0) = E(0,1) = 0.005;
    using Hooke = MatTB::Hooke<Dim, Eigen::Matrix<Real, Dim, Dim>, T4Mat<Real, Dim> >;
    Real lambda = Hooke::compute_lambda(Fix::Young, Fix::Poisson);
    Real mu     = Hooke::compute_mu(Fix::Young, Fix::Poisson);
    auto C = Hooke::compute_C(lambda, mu);
    T4MatMap<Real, Dim> Cmap{C.data()};
    Eigen::Matrix<Real, Dim, Dim> stress = Fix::mat.evaluate_stress(E, Cmap);
    Real sigma00 = lambda*E(0,0) + 2*mu*E(0,0);
    Real sigma01 = 2*mu*E(0,1);
    Real sigma11 = lambda*E(0,0);
    BOOST_CHECK_LT( std::abs(stress(0,0)- sigma00), tol);
    BOOST_CHECK_LT( std::abs(stress(0,1)- sigma01), tol);
    BOOST_CHECK_LT( std::abs(stress(1,0)- sigma01), tol);
    BOOST_CHECK_LT( std::abs(stress(1,1)- sigma11), tol);
    if (Dim == threeD){
      for (int i=0; i<Dim-1 ; ++i){
	BOOST_CHECK_LT( std::abs(stress(2,i)), tol );
	BOOST_CHECK_LT( std::abs(stress(i,2)), tol );
      }
      BOOST_CHECK_LT( std::abs(stress(2,2) - sigma11), tol );
    }
  };

  BOOST_AUTO_TEST_SUITE_END();
}  // muSpectre