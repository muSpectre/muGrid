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
    constexpr Dim_t mdim{Fix::mdim}, sdim{Fix::sdim};
    constexpr bool verbose{false};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;
    using traits = MaterialMuSpectre_traits<MaterialHyperElastoPlastic1<sdim, mdim>>;
    using LColl_t = typename traits::LFieldColl_t;
    using StrainStField_t = StateField<
      TensorField<LColl_t, Real, secondOrder, mdim>>;
    using FlowStField_t = StateField<
      ScalarField<LColl_t, Real>>;

    // using StrainStRef_t = typename traits::LStrainMap_t::reference;
    // using ScalarStRef_t = typename traits::LScalarMap_t::reference;

    // create statefields
    LColl_t coll{};
    coll.add_pixel({0});
    coll.initialise();

    StrainStField_t F_("previous gradient", coll);
    StrainStField_t be_("previous elastic strain", coll);
    FlowStField_t eps_("plastic flow", coll);

    auto F_prev{F_.get_map()};
    F_prev[0].current() = Strain_t::Identity();
    auto be_prev{be_.get_map()};
    be_prev[0].current() = Strain_t::Identity();
    auto eps_prev{eps_.get_map()};
    eps_prev[0].current() = 0;
    // elastic deformation
    Strain_t F{Strain_t::Identity()};
    F(0, 1) = 1e-5;


    Strain_t stress{Fix::mat.evaluate_stress(F,
                                             F_prev[0],
                                             be_prev[0],
                                             eps_prev[0])};

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
