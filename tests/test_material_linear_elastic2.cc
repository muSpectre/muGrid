/**
 * @file   test_material_linear_elastic2.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   04 Feb 2018
 *
 * @brief Tests for the objective Hooke's law with eigenstrains,
 *        (tests that do not require add_pixel are integrated into
 *        `test_material_linear_elastic1.cc`
 *
 * @section LICENSE
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
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include <type_traits>

#include <boost/mpl/list.hpp>
#include <boost/range/combine.hpp>

#include "materials/material_linear_elastic2.hh"
#include "tests.hh"
#include "tests/test_goodies.hh"
#include "common/field_collection.hh"
#include "common/iterators.hh"
namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(material_linear_elastic_2);

  template <class Mat_t>
  struct MaterialFixture
  {
    using Mat = Mat_t;
    constexpr static Real lambda{2}, mu{1.5};
    constexpr static Real young{mu*(3*lambda + 2*mu)/(lambda + mu)};
    constexpr static Real poisson{lambda/(2*(lambda + mu))};
    MaterialFixture():mat("Name", young, poisson){};
    constexpr static Dim_t sdim{Mat_t::sdim()};
    constexpr static Dim_t mdim{Mat_t::mdim()};

    Mat_t mat;
  };

  using mat_list = boost::mpl::list<
     MaterialFixture<MaterialLinearElastic2<twoD, twoD>>,
     MaterialFixture<MaterialLinearElastic2<twoD, threeD>>,
     MaterialFixture<MaterialLinearElastic2<threeD, threeD>>>;

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
    constexpr Dim_t sdim{Fix::sdim};
    testGoodies::RandRange<size_t> rng;;
    const Dim_t nb_pixel{7}, box_size{17};
    using Ccoord = Ccoord_t<sdim>;
    for (Dim_t i = 0; i < nb_pixel; ++i) {
      Ccoord c;
      for (Dim_t j = 0; j < sdim; ++j) {
        c[j] = rng.randval(0, box_size);
      }
      Eigen::Matrix<Real, Fix::mdim, Fix::mdim> Zero = Eigen::Matrix<Real, Fix::mdim, Fix::mdim>::Zero();
      BOOST_CHECK_NO_THROW
        (mat.add_pixel(c, Zero));
    }

    BOOST_CHECK_NO_THROW(mat.initialise());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_eigenstrain_equivalence, Fix,
                                   mat_list, Fix) {
    auto & mat{Fix::mat};

    const Dim_t nb_pixel{2};
    constexpr auto cube{CcoordOps::get_cube<Fix::sdim>(nb_pixel)};
    constexpr auto loc{CcoordOps::get_cube<Fix::sdim>(0)};

    using Mat_t = Eigen::Matrix<Real, Fix::mdim, Fix::mdim>;
    using FC_t = GlobalFieldCollection<Fix::sdim>;
    FC_t globalfields;
    auto & F_f{make_field<typename Fix::Mat::StrainField_t>
        ("Transformation Gradient", globalfields)};
    auto & P1_f = make_field<typename Fix::Mat::StressField_t>
      ("Nominal Stress1", globalfields); // to be computed alone
    auto & K_f = make_field<typename Fix::Mat::TangentField_t>
      ("Tangent Moduli", globalfields); // to be computed with tangent
    globalfields.initialise(cube, loc);

    Mat_t zero{Mat_t::Zero()};
    Mat_t F{Mat_t::Random()/100 + Mat_t::Identity()};
    Mat_t strain{-.5*(F+F.transpose())-Mat_t::Identity()};

    using Ccoord = Ccoord_t<Fix::sdim>;
    Ccoord pix0{0};
    Ccoord pix1{1};

    mat.add_pixel(pix0, zero);
    mat.add_pixel(pix1, strain);
    mat.initialise();

    F_f.get_map()[pix0] = -strain;
    F_f.get_map()[pix1] = zero;

    mat.compute_stresses_tangent(F_f, P1_f, K_f, Formulation::small_strain);

    Real error{(P1_f.get_map()[pix0]-P1_f.get_map()[pix1]).norm()};

    Real tol{1e-12};
    if (error >= tol) {
      std::cout << "error = " << error << " >= " << tol << " = tol" << std::endl;
      std::cout << "P(0) =" << std::endl << P1_f.get_map()[pix0]  << std::endl;
      std::cout << "P(1) =" << std::endl << P1_f.get_map()[pix1]  << std::endl;
    }
    BOOST_CHECK_LT(error, tol);
  }

  template <class Mat_t>
  struct MaterialFixtureFilled:
    public MaterialFixture<Mat_t>
  {
    using Par = MaterialFixture<Mat_t>;
    using Mat = Mat_t;
    constexpr static Dim_t box_size{3};
    MaterialFixtureFilled():MaterialFixture<Mat_t>(){
      using Ccoord = Ccoord_t<Mat_t::sdim()>;
      Ccoord cube{CcoordOps::get_cube<Mat_t::sdim()>(box_size)};
      CcoordOps::Pixels<Mat_t::sdim()> pixels(cube);
      for (auto pixel: pixels) {
        Eigen::Matrix<Real, Par::mdim, Par::mdim> Zero = Eigen::Matrix<Real, Par::mdim, Par::mdim>::Zero();
        this->mat.add_pixel(pixel,
                            Zero);
      }
      this->mat.initialise();
    };
  };

  using mat_fill = boost::mpl::list
    <MaterialFixtureFilled<MaterialLinearElastic2<twoD, twoD>>,
     MaterialFixtureFilled<MaterialLinearElastic2<twoD, threeD>>,
     MaterialFixtureFilled<MaterialLinearElastic2<threeD, threeD>>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_law, Fix, mat_fill, Fix) {
    constexpr auto cube{CcoordOps::get_cube<Fix::sdim>(Fix::box_size)};
    constexpr auto loc{CcoordOps::get_cube<Fix::sdim>(0)};
    auto & mat{Fix::mat};

    using FC_t = GlobalFieldCollection<Fix::sdim>;
    FC_t globalfields;
    auto & F{make_field<typename Fix::Mat::StrainField_t>
        ("Transformation Gradient", globalfields)};
    auto & P1 = make_field<typename Fix::Mat::StressField_t>
      ("Nominal Stress1", globalfields); // to be computed alone
    auto & P2 = make_field<typename Fix::Mat::StressField_t>
      ("Nominal Stress2", globalfields); // to be computed with tangent
    auto & K = make_field<typename Fix::Mat::TangentField_t>
      ("Tangent Moduli", globalfields); // to be computed with tangent
    auto & Pr = make_field<typename Fix::Mat::StressField_t>
      ("Nominal Stress reference", globalfields);
    auto & Kr = make_field<typename Fix::Mat::TangentField_t>
      ("Tangent Moduli reference", globalfields); // to be computed with tangent

    globalfields.initialise(cube, loc);

    static_assert(std::is_same<decltype(P1),
                  typename Fix::Mat::StressField_t&>::value,
                  "oh oh");
    static_assert(std::is_same<decltype(F),
                  typename Fix::Mat::StrainField_t&>::value,
                  "oh oh");
    static_assert(std::is_same<decltype(P1), decltype(P2)&>::value,
                  "oh oh");
    static_assert(std::is_same<decltype(K),
                  typename Fix::Mat::TangentField_t&>::value,
                  "oh oh");
    static_assert(std::is_same<decltype(Pr), decltype(P1)&>::value,
                  "oh oh");
    static_assert(std::is_same<decltype(Kr), decltype(K)&>::value,
                  "oh oh");
    using traits = MaterialMuSpectre_traits<typename Fix::Mat>;
    { // block to contain not-constant gradient map
      typename traits::StressMap_t grad_map
        (globalfields["Transformation Gradient"]);
      for (auto F_: grad_map) {
        F_.setRandom();
      }
      grad_map[0] = grad_map[0].Identity(); // identifiable gradients for debug
      grad_map[1] = 1.2*grad_map[1].Identity(); // ditto
    }

    //compute stresses using material
    mat.compute_stresses(globalfields["Transformation Gradient"],
                         globalfields["Nominal Stress1"],
                         Formulation::finite_strain);

    //compute stresses and tangent moduli using material
    BOOST_CHECK_THROW
      (mat.compute_stresses_tangent(globalfields["Transformation Gradient"],
                                    globalfields["Nominal Stress2"],
                                    globalfields["Nominal Stress2"],
                                    Formulation::finite_strain),
       std::runtime_error);

    mat.compute_stresses_tangent(globalfields["Transformation Gradient"],
                                 globalfields["Nominal Stress2"],
                                 globalfields["Tangent Moduli"],
                                 Formulation::finite_strain);

    typename traits::StrainMap_t Fmap(globalfields["Transformation Gradient"]);
    typename traits::StressMap_t Pmap_ref(globalfields["Nominal Stress reference"]);
    typename traits::TangentMap_t Kmap_ref(globalfields["Tangent Moduli reference"]);

    for (auto tup: akantu::zip(Fmap, Pmap_ref, Kmap_ref)) {
      auto F_ = std::get<0>(tup);
      auto P_ = std::get<1>(tup);
      auto K_ = std::get<2>(tup);
      std::tie(P_,K_) = testGoodies::objective_hooke_explicit<Fix::mdim>
        (Fix::lambda, Fix::mu, F_);
    }

    typename traits::StressMap_t Pmap_1(globalfields["Nominal Stress1"]);
    for (auto tup: akantu::zip(Pmap_ref, Pmap_1)) {
      auto P_r = std::get<0>(tup);
      auto P_1 = std::get<1>(tup);
      Real error = (P_r - P_1).norm();
      BOOST_CHECK_LT(error, tol);
    }

    typename traits::StressMap_t Pmap_2(globalfields["Nominal Stress2"]);
    typename traits::TangentMap_t Kmap(globalfields["Tangent Moduli"]);
    for (auto tup: akantu::zip(Pmap_ref, Pmap_2, Kmap_ref, Kmap)) {
      auto P_r = std::get<0>(tup);
      auto P = std::get<1>(tup);
      Real error = (P_r - P).norm();
      if (error >= tol) {
        std::cout << "error = " << error << " >= " << tol << " = tol" << std::endl;
        std::cout << "P(0) =" << std::endl << P_r << std::endl;
        std::cout << "P(1) =" << std::endl << P   << std::endl;
      }
      BOOST_CHECK_LT(error, tol);

      auto K_r = std::get<2>(tup);
      auto K = std::get<3>(tup);
      error = (K_r - K).norm();
      BOOST_CHECK_LT(error, tol);
    }

  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
