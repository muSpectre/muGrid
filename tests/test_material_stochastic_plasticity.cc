/**
 * @file   test_material_stochastic_plasticity.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   27 Nov 2020
 *
 * @brief  description
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

#include "tests.hh"
#include <boost/mpl/list.hpp>
#include "materials/material_stochastic_plasticity.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(material_stochastic_plasticity);

  template <class Mat_t>
  struct MaterialFixture {
    using Material_t = Mat_t;
    Material_t mat;
    constexpr static Dim_t spatial_dim() {
      return Material_t::MaterialDimension();
    }
    constexpr static Dim_t NbQuadPts() { return 2; }
    MaterialFixture() : mat("name", spatial_dim(), NbQuadPts()) {
      mat.add_pixel({0}, Youngs_modulus, Poisson_ratio, plastic_increment,
                    stress_threshold, eigen_strain);
    }
    constexpr static Dim_t Dim{spatial_dim()};
    Real Youngs_modulus{10};
    Real Poisson_ratio{0.3};
    Real plastic_increment{1e-5};
    Real stress_threshold{0.1};
    const Eigen::Matrix<Real, Dim, Dim> eigen_strain =
        Eigen::MatrixXd::Zero(Dim, Dim);
  };

  using mat_list =
      boost::mpl::list<MaterialFixture<MaterialStochasticPlasticity<twoD>>,
                       MaterialFixture<MaterialStochasticPlasticity<threeD>>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mat_list, Fix){};

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_set_threshold, Fix, mat_list, Fix) {
    size_t quad_pt_id {0};
    Real new_threshold{0.2};
    Fix::mat.initialise();
    Fix::mat.set_stress_threshold(quad_pt_id, new_threshold);
    Real read_threshold = Fix::mat.get_stress_threshold(quad_pt_id);

    BOOST_CHECK_EQUAL(new_threshold, read_threshold);
  };

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muSpectre
