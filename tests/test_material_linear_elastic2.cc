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
 * Copyright © 2018 Till Junge
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

#include <libmugrid/field_collection_global.hh>
#include <libmugrid/mapped_field.hh>
#include <libmugrid/iterators.hh>
#include <materials/material_linear_elastic2.hh>

#include <type_traits>
#include <boost/mpl/list.hpp>
#include <boost/range/combine.hpp>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(material_linear_elastic_2);

  template <class Mat_t>
  struct MaterialFixture {
    using Mat = Mat_t;
    constexpr static Real lambda{2}, mu{1.5};
    constexpr static Real young{mu * (3 * lambda + 2 * mu) / (lambda + mu)};
    constexpr static Real poisson{lambda / (2 * (lambda + mu))};
    constexpr static Index_t NbQuadPts() { return 2; }
    constexpr static Index_t sdim{twoD};
    constexpr static Index_t mdim() { return Mat_t::MaterialDimension(); }
    MaterialFixture() : mat("Name", mdim(), NbQuadPts(), young, poisson) {}

    Mat_t mat;
  };
  template <class Mat_t>
  constexpr Index_t MaterialFixture<Mat_t>::sdim;

  using mat_list =
      boost::mpl::list<MaterialFixture<MaterialLinearElastic2<twoD>>,
                       MaterialFixture<MaterialLinearElastic2<threeD>>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mat_list, Fix) {
    BOOST_CHECK_EQUAL("Name", Fix::mat.get_name());
    auto & mat{Fix::mat};
    auto mdim{Fix::mdim()};
    BOOST_CHECK_EQUAL(mdim, mat.MaterialDimension());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_add_pixel, Fix, mat_list, Fix) {
    auto & mat{Fix::mat};
    muGrid::testGoodies::RandRange<size_t> rng;

    const Index_t nb_pixel{7}, box_size{17};
    for (Index_t i = 0; i < nb_pixel; ++i) {
      const size_t c{rng.randval(0, box_size)};
      Eigen::Matrix<Real, Fix::mdim(), Fix::mdim()> Zero =
          Eigen::Matrix<Real, Fix::mdim(), Fix::mdim()>::Zero();
      BOOST_CHECK_NO_THROW(mat.add_pixel(c, Zero));
    }

    BOOST_CHECK_NO_THROW(mat.initialise());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_eigenstrain_equivalence, Fix, mat_list,
                                   Fix) {
    auto & mat{Fix::mat};

    const Index_t nb_pixel{2};
    constexpr auto cube{muGrid::CcoordOps::get_cube<Fix::sdim>(nb_pixel)};
    constexpr auto loc{muGrid::CcoordOps::get_cube<Fix::sdim>(Index_t{0})};

    using Mat_t = Eigen::Matrix<Real, Fix::mdim(), Fix::mdim()>;
    using FC_t = muGrid::GlobalFieldCollection;
    FC_t globalfields{Fix::mdim()};
    globalfields.set_nb_sub_pts(QuadPtTag, Fix::NbQuadPts());
    globalfields.initialise(cube, loc);

    Mat_t zero{Mat_t::Zero()};
    Mat_t F{Mat_t::Random() / 100 + Mat_t::Identity()};
    Mat_t strain{-.5 * (F + F.transpose()) - Mat_t::Identity()};

    Index_t pix0{0};
    Index_t pix1{1};

    mat.add_pixel(pix0, zero);
    mat.add_pixel(pix1, strain);
    mat.initialise();

    muGrid::MappedT2Field<Real, Mapping::Mut, Fix::mdim(), IterUnit::SubPt>
        F_f{"Transformation Gradient", globalfields, QuadPtTag};
    muGrid::MappedT2Field<Real, Mapping::Mut, Fix::mdim(), IterUnit::SubPt>
        P1_f{"Nominal Stress1", globalfields,
             QuadPtTag};  // to be computed alone
    muGrid::MappedT4Field<Real, Mapping::Mut, Fix::mdim(), IterUnit::SubPt>
        K_f{"Tangent Moduli", globalfields,
            QuadPtTag};  // to be computed with tangent
    for (Index_t quad_id{0}; quad_id < Fix::NbQuadPts(); ++quad_id) {
      F_f.get_map()[pix0 * Fix::NbQuadPts() + quad_id] = -strain;
      F_f.get_map()[pix1 * Fix::NbQuadPts() + quad_id] = zero;
    }

    const auto & F_field{F_f.get_field()};
    auto & P_field{P1_f.get_field()};
    auto & K_field{K_f.get_field()};
    mat.compute_stresses_tangent(F_field, P_field, K_field,
                                 Formulation::small_strain);

    Real error{(P1_f.get_map()[pix0] - P1_f.get_map()[pix1]).norm()};

    Real tol{1e-12};
    if (error >= tol) {
      std::cout << "error = " << error << " >= " << tol << " = tol"
                << std::endl;
      std::cout << "P(0) =" << std::endl << P1_f.get_map()[pix0] << std::endl;
      std::cout << "P(1) =" << std::endl
                << P1_f.get_map()[Fix::NbQuadPts()] << std::endl;
    }
    BOOST_CHECK_LT(error, tol);
  }

  template <class Mat_t>
  struct MaterialFixtureFilled : public MaterialFixture<Mat_t> {
    using Par = MaterialFixture<Mat_t>;
    using Mat = Mat_t;
    constexpr static Index_t box_size{3};
    MaterialFixtureFilled() : MaterialFixture<Mat_t>() {
      using Ccoord = Ccoord_t<Par::sdim>;
      Ccoord cube{muGrid::CcoordOps::get_cube<Par::sdim>(box_size)};
      muGrid::CcoordOps::Pixels<Par::sdim> pixels(cube);
      for (auto && id_pixel : akantu::enumerate(pixels)) {
        auto && id{std::get<0>(id_pixel)};
        Eigen::Matrix<Real, Par::mdim(), Par::mdim()> Zero =
            Eigen::Matrix<Real, Par::mdim(), Par::mdim()>::Zero();
        this->mat.add_pixel(id, Zero);
      }
      this->mat.initialise();
    }
  };

  using mat_fill =
      boost::mpl::list<MaterialFixtureFilled<MaterialLinearElastic2<twoD>>,
                       MaterialFixtureFilled<MaterialLinearElastic2<threeD>>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_law, Fix, mat_fill, Fix) {
    constexpr auto cube{muGrid::CcoordOps::get_cube<Fix::sdim>(Fix::box_size)};
    constexpr auto loc{muGrid::CcoordOps::get_cube<Fix::sdim>(Index_t{0})};
    auto & mat{Fix::mat};

    using FC_t = muGrid::GlobalFieldCollection;
    FC_t globalfields{Fix::sdim};
    globalfields.set_nb_sub_pts(QuadPtTag, Fix::NbQuadPts());
    globalfields.register_real_field("Transformation Gradient",
                                     Fix::mdim() * Fix::mdim(),
                                     QuadPtTag);
    globalfields.register_real_field("Nominal Stress1",
                                     Fix::mdim() * Fix::mdim(),
                                     QuadPtTag);  // to be computed alone
    globalfields.register_real_field("Nominal Stress2",
                                     Fix::mdim() * Fix::mdim(),
                                     QuadPtTag);  // to be computed with tangent
    globalfields.register_real_field("Tangent Moduli",
                                     muGrid::ipow(Fix::mdim(), 4),
                                     QuadPtTag);  // to be computed with tangent
    globalfields.register_real_field("Nominal Stress reference",
                                     muGrid::ipow(Fix::mdim(), 2), QuadPtTag);
    globalfields.register_real_field("Tangent Moduli reference",
                                     muGrid::ipow(Fix::mdim(), 4),
                                     QuadPtTag);  // to be computed with tangent

    globalfields.initialise(cube, loc);

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
                      std::runtime_error);

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
      std::tie(P_, K_) =
          muGrid::testGoodies::objective_hooke_explicit<Fix::mdim()>(
              Fix::lambda, Fix::mu, F_);
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
      if (error >= tol) {
        std::cout << "error = " << error << " >= " << tol << " = tol"
                  << std::endl;
        std::cout << "P(0) =" << std::endl << P_r << std::endl;
        std::cout << "P(1) =" << std::endl << P << std::endl;
      }
      BOOST_CHECK_LT(error, tol);

      auto K_r = std::get<2>(tup);
      auto K = std::get<3>(tup);
      error = (K_r - K).norm();
      BOOST_CHECK_LT(error, tol);
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
