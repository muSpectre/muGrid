/**
 * @file   test_s_t_material_linear_elastic_generic1.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   20 Jan 2020
 *
 * @brief  Testing the equality of the result of
 * MaterialLinearElasticGeneric1<DimM> and STMaterialLinearElasticGeneric1<DimM,
 * GreenLagrnage, PK2>
 *
 * Copyright © 2020 Ali Falsafi
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
 * Boston, MA 02111-1307, USA.
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

#include "materials/s_t_material_linear_elastic_generic1.hh"

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(s_t_material_linear_elasrtic_generic1);

  template <Dim_t Dim>
  struct MatFixture {
    using Mat_ref_t = MaterialLinearElasticGeneric1<Dim>;
    using Mat_t =
        STMaterialLinearElasticGeneric1<Dim, StrainMeasure::GreenLagrange,
                                        StressMeasure::PK2>;
    using T2_t = Eigen::Matrix<Real, Dim, Dim>;
    using T4_t = muGrid::T4Mat<Real, Dim>;
    using V_t = Eigen::Matrix<Real, vsize(Dim), vsize(Dim)>;
    constexpr static Real lambda{2}, mu{1.5};
    constexpr static Real get_lambda() { return lambda; }
    constexpr static Real get_mu() { return mu; }
    constexpr static Real young{mu * (3 * lambda + 2 * mu) / (lambda + mu)};
    constexpr static Real poisson{lambda / (2 * (lambda + mu))};
    using Hooke = MatTB::Hooke<Dim, T2_t, T4_t>;

    constexpr static Dim_t mdim() { return Dim; }
    constexpr static Dim_t NbQuadPts() { return 2; }

    MatFixture()
        : C_voigt{get_C_voigt()},
          mat_ref("material_ref", mdim(), NbQuadPts(), this->C_voigt),
          mat("material", mdim(), NbQuadPts(), this->C_voigt) {}

    static V_t get_C_voigt() {
      V_t C{};
      C.setZero();
      C.template topLeftCorner<Dim, Dim>().setConstant(get_lambda());
      C.template topLeftCorner<Dim, Dim>() += 2 * get_mu() * T2_t::Identity();
      constexpr Dim_t Rest{vsize(Dim) - Dim};
      using Rest_t = Eigen::Matrix<Real, Rest, Rest>;
      C.template bottomRightCorner<Rest, Rest>() +=
          get_mu() * Rest_t::Identity();
      return C;
    }

    V_t C_voigt;
    Mat_ref_t mat_ref;
    Mat_t mat;
  };

  using mats = boost::mpl::list<MatFixture<twoD>, MatFixture<threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(stress_test, Fix, mats, Fix) {
    using T2_t = typename Fix::T2_t;

    auto & mat{Fix::mat};
    auto & mat_ref{Fix::mat_ref};

    T2_t F{(T2_t::Random() - (T2_t::Ones() * .5)) * 1e-4 + T2_t::Identity()};
    const T2_t E{.5 *
                 ((F - T2_t::Identity()) + (F - T2_t::Identity()).transpose())};
    mat.set_F(F);
    auto && S{mat.evaluate_stress(E)};
    auto && S_ref{mat_ref.evaluate_stress(E)};

    constexpr Real tol{1e-8};
    Real err{rel_error(S, S_ref)};
    BOOST_CHECK_LT(err, tol);
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muSpectre
