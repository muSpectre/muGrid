/**
 * @file   test_material_linear_elastic1.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   28 Nov 2017
 *
 * @brief  Tests for the large-strain, objective Hooke's law, implemented in
 *         the convenient strategy (i.e., using MaterialMuSpectre), also used
 *         to test parts of MaterialLinearElastic2
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

#include "tests.hh"
#include "libmugrid/test_goodies.hh"

#include "materials/material_linear_elastic1.hh"
#include "materials/material_linear_elastic2.hh"

#include <libmugrid/iterators.hh>
#include <libmugrid/nfield_typed.hh>

#include <type_traits>
#include <boost/mpl/list.hpp>
#include <boost/range/combine.hpp>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(material_linear_elastic_1);

  template <class Mat_t>
  struct MaterialFixture : public Mat_t {
    using Mat = Mat_t;
    const Dim_t NbQuadPts{1};
    const Real lambda{2}, mu{1.5};
    const Real young{mu * (3 * lambda + 2 * mu) / (lambda + mu)};
    const Real poisson{lambda / (2 * (lambda + mu))};
    MaterialFixture()
        : mat("Name", Mat_t::MaterialDimension(), NbQuadPts, young, poisson) {}
    Mat_t mat;
  };

  template <class Mat_t>
  struct has_internals {
    constexpr static bool value{false};
  };

  template <Dim_t DimM>
  struct has_internals<MaterialLinearElastic2<DimM>> {
    constexpr static bool value{true};
  };

  using mat_list =
      boost::mpl::list<MaterialFixture<MaterialLinearElastic1<twoD>>,
                       MaterialFixture<MaterialLinearElastic1<threeD>>,
                       MaterialFixture<MaterialLinearElastic2<twoD>>,
                       MaterialFixture<MaterialLinearElastic2<threeD>>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mat_list, Fix) {
    BOOST_CHECK_EQUAL("Name", Fix::mat.get_name());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_add_pixel, Fix, mat_list, Fix) {
    auto & mat{Fix::mat};
    muGrid::testGoodies::RandRange<size_t> rng;

    const Dim_t nb_pixel{7}, box_size{17};
    for (Dim_t i = 0; i < nb_pixel; ++i) {
      auto && c =
          rng.randval(0, muGrid::ipow(box_size, mat.MaterialDimension()));
      if (!has_internals<typename Fix::Mat>::value) {
        BOOST_CHECK_NO_THROW(Fix::add_pixel(c));
      }
    }

    BOOST_CHECK_NO_THROW(Fix::initialise());
  }

  template <class Mat_t>
  struct MaterialFixtureFilled : public MaterialFixture<Mat_t> {
    using Mat = Mat_t;
    constexpr static Dim_t box_size{3};
    MaterialFixtureFilled()
        : MaterialFixture<Mat_t>(),
          mat("Mat Name", this->young, this->poisson) {
      using Ccoord = Ccoord_t<Mat_t::sdim()>;
      Ccoord cube{muGrid::CcoordOps::get_cube<Mat_t::sdim()>(box_size)};
      muGrid::CcoordOps::Pixels<Mat_t::sdim()> pixels(cube);
      for (auto && id_pixel : akantu::enumerate(pixels)) {
        auto && id{std::get<0>(id_pixel)};
        this->mat.add_pixel(id);
      }
      this->mat.initialise();
    }
    Mat_t mat;
  };

  using mat_fill =
      boost::mpl::list<MaterialFixtureFilled<MaterialLinearElastic1<twoD>>,
                       MaterialFixtureFilled<MaterialLinearElastic1<threeD>>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_single_pixel, Fix, mat_fill,
                                   Fix) {
    Eigen::Matrix<Real, Fix::mdim, Fix::mdim> I{
        Eigen::Matrix<Real, Fix::mdim, Fix::mdim>::Identity() +
        0.1 * Eigen::Matrix<Real, Fix::mdim, Fix::mdim>::Random()};
    auto origin_eval_func_result = Fix::evaluate_stress(I, 0);
    auto base_eval_func_result =
        Fix::evaluate_stress_base(I, 0, Formulation::small_strain);
    Real error = (origin_eval_func_result - base_eval_func_result).norm();
    BOOST_CHECK_EQUAL(error, 0.0);

    auto origin_eval_func_result_2 = Fix::evaluate_stress_tangent(
        Eigen::Matrix<Real, Fix::mdim, Fix::mdim>::Identity(), 0);
    auto base_eval_func_result_2 =
        Fix::evaluate_stress_tangent_base(I, 0, Formulation::small_strain);
    error = (std::get<1>(origin_eval_func_result_2) -
             std::get<1>(base_eval_func_result_2))
                .norm();
    BOOST_CHECK_EQUAL(error, 0.0);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_iterable_proxy_constructors, Fix,
                                   mat_fill, Fix) {
    constexpr auto cube{muGrid::CcoordOps::get_cube<Fix::sdim>(Fix::box_size)};
    constexpr auto loc{muGrid::CcoordOps::get_cube<Fix::sdim>(0)};

    using FC_t = muGrid::GlobalFieldCollection<Fix::sdim>;
    FC_t globalfields;
    auto & F{muGrid::make_field<typename Fix::Mat::StrainField_t>(
        "Transformation Gradient", globalfields)};
    auto & F_rate{muGrid::make_field<typename Fix::Mat::StrainField_t>(
        "Transformation Gradient Rate", globalfields)};
    auto & P = muGrid::make_field<typename Fix::Mat::StressField_t>(
        "Nominal Stress1", globalfields);  // to be computed alone
    auto & K = muGrid::make_field<typename Fix::Mat::TangentField_t>(
        "Tangent Moduli", globalfields);  // to be computed with tangent
    globalfields.initialise(cube, loc);

    using traits =
        MaterialMuSpectre_traits<MaterialLinearElastic1<Fix::sdim, Fix::mdim>>;

    using iterable_proxy_t_without_rate_without_tangent =
        typename Fix::template iterable_proxy<
            std::tuple<typename traits::StrainMap_t>,
            std::tuple<typename traits::StressMap_t>>;
    iterable_proxy_t_without_rate_without_tangent
        field_without_rate_without_tangent(Fix::mat, F, P);

    using iterable_proxy_t_without_rate_with_tangent =
        typename Fix::template iterable_proxy<
            std::tuple<typename traits::StrainMap_t>,
            std::tuple<typename traits::StressMap_t,
                       typename traits::TangentMap_t>>;
    iterable_proxy_t_without_rate_with_tangent fields_without_rate_with_tangent(
        Fix::mat, F, P, K);

    using iterable_proxy_t_with_rate_without_tangent =
        typename Fix::template iterable_proxy<
            std::tuple<typename traits::StrainMap_t,
                       typename traits::StrainMap_t>,
            std::tuple<typename traits::StressMap_t>>;
    iterable_proxy_t_with_rate_without_tangent fields_with_rate_without_tangent(
        Fix::mat, F, F_rate, P);

    using iterable_proxy_t_with_rate_with_tangent =
        typename Fix::template iterable_proxy<
            std::tuple<typename traits::StrainMap_t,
                       typename traits::StrainMap_t>,
            std::tuple<typename traits::StressMap_t,
                       typename traits::TangentMap_t>>;

    iterable_proxy_t_with_rate_with_tangent fields_with_rate_with_tangent(
        Fix::mat, F, F_rate, P, K);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_law, Fix, mat_fill, Fix) {
    constexpr auto cube{muGrid::CcoordOps::get_cube<Fix::sdim>(Fix::box_size)};
    constexpr auto loc{muGrid::CcoordOps::get_cube<Fix::sdim>(0)};

    using FC_t = muGrid::GlobalNFieldCollection;
    FC_t globalfields{SDim, muGrid::Unknown};
    globalfields.set_nb_quad(Fix::NbQuadPts);
    globalfields.initialise(cube, loc);
    globalfields.register_real_field("Transformation Gradient", mdim * mdim);
    auto & P1 = globalfields.register_real_field(
        "Nominal Stress1", mdim * mdim);  // to be computed alone
    globalfields.register_real_field(
        "Nominal Stress2", mdim * mdim);  // to be computed with tangent
    globalfields.register_real_field(
        "Tangent Moduli",
        muGrid::ipow(mdim, 4));  // to be computed with tangent
    globalfields.register_real_field("Nominal Stress reference", mdim * mdim);
    globalfields.register_real_field(
        "Tangent Moduli reference",
        muGrid::ipow(mdim, 4));  // to be computed with tangent

    static_assert(std::is_same<decltype(P1), muGrid::RealNField &>::value,
                  "oh oh");
    using traits = MaterialMuSpectre_traits<typename Fix::Mat>;
    {  // block to contain not-constant gradient map
      typename traits::StressMap_t grad_map(
          globalfields.get_field("Transformation Gradient"));
      for (auto F_ : grad_map) {
        F_.setRandom();
      }
      grad_map[0] = grad_map[0].Identity();  // identifiable gradients for debug
      grad_map[1] = 1.2 * grad_map[1].Identity();  // ditto
    }

    // compute stresses using material
    mat.compute_stresses(globalfields.get_field("Transformation Gradient"),
                         globalfields.get_field("Nominal Stress1"),
                         Formulation::finite_strain);

    // compute stresses and tangent moduli using material
    BOOST_CHECK_THROW(mat.compute_stresses_tangent(
                          globalfields.get_field("Transformation Gradient"),
                          globalfields.get_field("Nominal Stress2"),
                          globalfields.get_field("Nominal Stress2"),
                          Formulation::finite_strain),
                      muGrid::NFieldError);

    mat.compute_stresses_tangent(
        globalfields.get_field("Transformation Gradient"),
        globalfields.get_field("Nominal Stress2"),
        globalfields.get_field("Tangent Moduli"), Formulation::finite_strain);

    typename traits::StrainMap_t Fmap(
        globalfields.get_field("Transformation Gradient"));
    typename traits::StressMap_t Pmap_ref(
        globalfields.get_field("Nominal Stress reference"));
    typename traits::TangentMap_t Kmap_ref(
        globalfields.get_field("Tangent Moduli reference"));

    for (auto tup : akantu::zip(Fmap, Pmap_ref, Kmap_ref)) {
      auto F_ = std::get<0>(tup);
      auto P_ = std::get<1>(tup);
      auto K_ = std::get<2>(tup);
      std::tie(P_, K_) = muGrid::testGoodies::objective_hooke_explicit<
          Fix::Mat::MaterialDimension()>(Fix::lambda, Fix::mu, F_);
    }

    typename traits::StressMap_t Pmap_1(
        globalfields.get_field("Nominal Stress1"));
    for (auto tup : akantu::zip(Pmap_ref, Pmap_1)) {
      auto P_r = std::get<0>(tup);
      auto P_1 = std::get<1>(tup);
      Real error = (P_r - P_1).norm();
      BOOST_CHECK_LT(error, tol);
    }

    typename traits::StressMap_t Pmap_2(
        globalfields.get_field("Nominal Stress2"));
    typename traits::TangentMap_t Kmap(
        globalfields.get_field("Tangent Moduli"));
    for (auto tup : akantu::zip(Pmap_ref, Pmap_2, Kmap_ref, Kmap)) {
      auto P_r = std::get<0>(tup);
      auto P = std::get<1>(tup);
      Real error = (P_r - P).norm();
      BOOST_CHECK_LT(error, tol);

      auto K_r = std::get<2>(tup);
      auto K = std::get<3>(tup);
      error = (K_r - K).norm();
      BOOST_CHECK_LT(error, tol);
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
