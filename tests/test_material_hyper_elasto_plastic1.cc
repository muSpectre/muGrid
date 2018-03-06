/**
 * @file   test_material_hyper_elasto_plastic1.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   25 Feb 2018
 *
 * @brief  Tests for the large-strain Simo-type plastic law implemented
 *         using MaterialMuSpectre
 *
 * Copyright © 2018 Till Junge
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

#include "boost/mpl/list.hpp"

#include "materials/material_hyper_elasto_plastic1.hh"
#include "materials/materials_toolbox.hh"
#include "tests.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(material_hyper_elasto_plastic_1);

  template <class Mat_t>
  struct MaterialFixture
  {
    using Mat = Mat_t;
    constexpr static Real K{.833};      // bulk modulus
    constexpr static Real mu{.386};     // shear modulus
    constexpr static Real H{.004};      // hardening modulus
    constexpr static Real tau_y0{.003}; // initial yield stress
    constexpr static Real young{
      MatTB::convert_elastic_modulus<ElasticModulus::Young,
                                     ElasticModulus::Bulk,
                                     ElasticModulus::Shear>(K, mu)};
    constexpr static Real poisson{
      MatTB::convert_elastic_modulus<ElasticModulus::Poisson,
                                     ElasticModulus::Bulk,
                                     ElasticModulus::Shear>(K, mu)};
    MaterialFixture():mat("Name", young, poisson, H, tau_y0){};
    constexpr static Dim_t sdim{Mat_t::sdim()};
    constexpr static Dim_t mdim{Mat_t::mdim()};

    Mat_t mat;
  };

  using mats = boost::mpl::list<MaterialFixture<MaterialHyperElastoPlastic1<  twoD,   twoD>>,
                                MaterialFixture<MaterialHyperElastoPlastic1<  twoD, threeD>>,
                                MaterialFixture<MaterialHyperElastoPlastic1<threeD, threeD>>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mats, Fix) {
    BOOST_CHECK_EQUAL("Name", Fix::mat.get_name());
    auto & mat{Fix::mat};
    auto sdim{Fix::sdim};
    auto mdim{Fix::mdim};
    BOOST_CHECK_EQUAL(sdim, mat.sdim());
    BOOST_CHECK_EQUAL(mdim, mat.mdim());
  }


  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_stress, Fix, mats, Fix) {
    constexpr Dim_t dim{Fix::mdim};
    constexpr bool verbose{false};
    using Strain_t = Eigen::Matrix<Real, dim, dim>;
    using StrainRef_t = Eigen::Map<Strain_t>;

    // elastic deformation
    Strain_t F{Strain_t::Identity()};
    F(0, 1) = 1e-5;
    Strain_t F_prev{Strain_t::Identity()};
    Strain_t be_prev{Strain_t::Identity()};
    Real eps_prev{0};

    Strain_t stress{Fix::mat.evaluate_stress(F,
                                             StrainRef_t(F_prev.data()),
                                             StrainRef_t(be_prev.data()),
                                             eps_prev)};

    if (verbose) {
      std::cout << "τ  =" << std::endl << stress << std::endl
                << "F  =" << std::endl << F << std::endl
                << "Fₜ =" << std::endl << F_prev << std::endl
                << "bₑ =" << std::endl << be_prev << std::endl
                << "εₚ =" << std::endl << eps_prev << std::endl;
    }

    // plastic deformation
    F(0, 1) = .2;
    stress = Fix::mat.evaluate_stress(F,
                                      StrainRef_t(F_prev.data()),
                                      StrainRef_t(be_prev.data()),
                                      eps_prev);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
