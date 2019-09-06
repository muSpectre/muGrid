/**
 * @file   test_material_linear_elastic4.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   27 Mar 2018
 *
 * @brief  description
 *
 * Copyright © 2018 Richard Leute
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
#include "tests.hh"
#include "materials/material_linear_elastic4.hh"
#include "materials/materials_toolbox.hh"
#include <libmugrid/T4_map_proxy.hh>
#include <cmath>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(material_linear_elastic_4);

  template <class Mat_t>
  struct MaterialFixture {
    using Material_t = Mat_t;
    Material_t mat;
    constexpr static Dim_t mdim() { return Material_t::MaterialDimension(); }
    constexpr static Dim_t NbQuadPts() { return 2; }
    MaterialFixture() : mat("name", mdim(), NbQuadPts()) {
      mat.add_pixel({0}, Youngs_modulus, Poisson_ratio);
    }
    Real Youngs_modulus{10};
    Real Poisson_ratio{0.3};
  };

  using mat_list =
      boost::mpl::list<MaterialFixture<MaterialLinearElastic4<twoD>>,
                       MaterialFixture<MaterialLinearElastic4<threeD>>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mat_list, Fix){};

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_response, Fix, mat_list, Fix) {
    constexpr Dim_t Dim{Fix::mdim()};
    Eigen::Matrix<Real, Dim, Dim> E;
    E.setZero();
    E(0, 0) = 0.001;
    E(1, 0) = E(0, 1) = 0.005;

    using Hooke = MatTB::Hooke<Dim, Eigen::Matrix<Real, Dim, Dim>,
                               muGrid::T4Mat<Real, Dim>>;
    Real lambda =
        Hooke::compute_lambda(Fix::Youngs_modulus, Fix::Poisson_ratio);
    Real mu = Hooke::compute_mu(Fix::Youngs_modulus, Fix::Poisson_ratio);

    Eigen::Matrix<Real, Dim, Dim> stress =
        Fix::mat.evaluate_stress(E, lambda, mu);

    Real sigma00 = lambda * E(0, 0) + 2 * mu * E(0, 0);
    Real sigma01 = 2 * mu * E(0, 1);
    Real sigma11 = lambda * E(0, 0);
    BOOST_CHECK_LT(std::abs(stress(0, 0) - sigma00), tol);
    BOOST_CHECK_LT(std::abs(stress(0, 1) - sigma01), tol);
    BOOST_CHECK_LT(std::abs(stress(1, 0) - sigma01), tol);
    BOOST_CHECK_LT(std::abs(stress(1, 1) - sigma11), tol);
    if (Dim == threeD) {
      for (int i = 0; i < Dim - 1; ++i) {
        BOOST_CHECK_LT(std::abs(stress(2, i)), tol);
        BOOST_CHECK_LT(std::abs(stress(i, 2)), tol);
      }
      BOOST_CHECK_LT(std::abs(stress(2, 2) - sigma11), tol);
    }
  };

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muSpectre
