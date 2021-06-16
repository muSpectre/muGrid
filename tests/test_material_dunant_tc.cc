/**
 * @file   test_material_dunant_tc.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   09 Sep 2020
 *
 * @brief  the test for the MaterialDunantTC
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

#include "materials/material_dunant_tc.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/iterable_proxy.hh"

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(material_dunant_tc)
  template <Dim_t Dim, Index_t c_c = 1, Index_t c_t = 1>
  struct MaterialFixture {
    using MatDamTC = MaterialDunantTC<Dim>;
    using MatLin = MaterialLinearElastic1<Dim>;
    const Real young{1.0e5};
    const Real poisson{0.2};
    const Real kappa{1e-3};
    const Real kappa_var{kappa * 1e-3};
    const Real alpha{0.2};

    const Real coeff_c{1.0 * c_c};
    const Real coeff_t{1.0 * c_t};

    //! Constructor
    MaterialFixture()
        : mat_dam_eval_tc{MatDamTC::make_evaluator(young, poisson, kappa, alpha,
                                                   coeff_c, coeff_t)},
          mat_lin_eval{MatLin::make_evaluator(young, poisson)} {}

    constexpr static Dim_t mdim() { return Dim; }
    constexpr static Dim_t sdim() { return mdim(); }
    constexpr static Dim_t NbQuadPts() { return 1; }

    std::tuple<std::shared_ptr<MatDamTC>, MaterialEvaluator<Dim>>
        mat_dam_eval_tc;
    std::tuple<std::shared_ptr<MatLin>, MaterialEvaluator<Dim>> mat_lin_eval;
  };

  using mats = boost::mpl::list<MaterialFixture<twoD>, MaterialFixture<threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_add_pixel, Fix, mats, Fix) {
    auto & mat_dam_tc{*std::get<0>(Fix::mat_dam_eval_tc)};
    muGrid::testGoodies::RandRange<size_t> rng;

    const Dim_t nb_pixel{7}, box_size{17};
    for (Dim_t i = 0; i < nb_pixel; ++i) {
      const size_t c{rng.randval(0, box_size)};
      BOOST_CHECK_NO_THROW(mat_dam_tc.add_pixel(c, Fix::kappa_var));
    }
  }

  template <Index_t Dim, Index_t c_c = 1, Index_t c_t = 1>
  struct MaterialFixtureFilled : public MaterialFixture<Dim, c_c, c_t> {
    using Parent = MaterialFixture<Dim, c_c, c_t>;
    using MatDamTC = typename Parent::MatDamTC;
    using MatLin = typename Parent::MatLin;

    constexpr static Index_t box_size{1};
    MaterialFixtureFilled() : Parent() {
      auto & mat_dam_tc{*std::get<0>(Parent::mat_dam_eval_tc)};
      auto & mat_lin{*std::get<0>(Parent::mat_lin_eval)};
      mat_dam_tc.add_pixel(0);
      mat_lin.add_pixel(0);
      mat_dam_tc.initialise();
      mat_lin.initialise();
    }

    constexpr static Real get_tol() { return tol; }
    constexpr static Real get_tol_diff() { return finite_diff_tol; }
    constexpr static Dim_t mdim() { return Parent::mdim(); }
    constexpr static Dim_t sdim() { return mdim(); }
    const std::array<Real, 3> steps{{1.e-9, 1.e-8, 1.e-7}};
    Real get_step() { return this->step; }
  };

  using mats_fill = boost::mpl::list<MaterialFixtureFilled<twoD>,
                                     MaterialFixtureFilled<threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evalaute_stress_identical_coeffs, Fix,
                                   mats_fill, Fix) {
    constexpr Dim_t mdim{Fix::mdim()};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;

    auto & mat_dam_tc{*std::get<0>(Fix::mat_dam_eval_tc)};
    auto & mat_lin{*std::get<0>(Fix::mat_lin_eval)};

    // create state fields
    muGrid::LocalFieldCollection::SubPtMap_t map_tc{};
    map_tc[QuadPtTag] = Fix::NbQuadPts();
    muGrid::LocalFieldCollection coll_tc{mdim, map_tc};
    coll_tc.add_pixel({0});
    coll_tc.initialise();

    muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt>
        kappa_tc_{"Kappa TC", coll_tc, QuadPtTag};

    muGrid::MappedScalarField<Real, Mapping::Mut, IterUnit::SubPt> kappa_init_{
        "Kappa_init", coll_tc, QuadPtTag};

    kappa_init_[0] = Fix::kappa;

    auto & kappa_tc{kappa_tc_.get_map()};
    kappa_tc[0].current() = Fix::kappa;

    Strain_t F{2e-3 * Strain_t::Random() + Strain_t::Identity()};
    Strain_t E{0.5 * ((F * F.transpose()) - Strain_t::Identity())};

    kappa_tc_.get_state_field().cycle();

    auto && S_dam_tc{
        mat_dam_tc.evaluate_stress(E, kappa_tc_[0], kappa_init_[0])};

    auto && S_lin{mat_lin.evaluate_stress(E, 0)};

    auto && kap_ref{std::sqrt(muGrid::Matrices::ddot<Fix::mdim()>(E, E) /
                              (Fix::coeff_c + Fix::coeff_t))};

    if (kap_ref > Fix::kappa) {
      auto && kap{kappa_tc_[0].current()};
      auto && err{rel_error(kap, kap_ref)};

      if (err > Fix::get_tol()) {
        std::cout << "kappa: " << kappa_tc_[0].current() << "\n";
        std::cout << "kap_ref: " << kap_ref << "\n";
      }

      BOOST_CHECK_LT(err, Fix::get_tol());
      auto && red{kap > Fix::kappa
                      ? (((1 + Fix::alpha) * (Fix::kappa / kap)) - Fix::alpha)
                      : 1.0};

      auto && S_dam_ref{red * S_lin};
      err = rel_error(S_dam_tc, S_dam_ref);
      BOOST_CHECK_LT(err, Fix::get_tol());
    }
  }

  using mats_fill_tangent = boost::mpl::list<
      MaterialFixtureFilled<twoD, 1, 1>, MaterialFixtureFilled<threeD, 1, 1>,
      MaterialFixtureFilled<twoD, 1, 2>, MaterialFixtureFilled<threeD, 1, 2>,
      MaterialFixtureFilled<twoD, 1, 3>, MaterialFixtureFilled<threeD, 1, 3>,
      MaterialFixtureFilled<twoD, 2, 2>, MaterialFixtureFilled<threeD, 2, 2>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evalaute_stress_tangent_diff, Fix,
                                   mats_fill_tangent, Fix) {
    constexpr Dim_t mdim{Fix::mdim()};

    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;
    using Stiffness_t = muGrid::T4Mat<Real, mdim>;

    auto & mat_dam{*std::get<0>(Fix::mat_dam_eval_tc)};
    auto & evaluator{std::get<1>(Fix::mat_dam_eval_tc)};

    Real strain_amp{8.56584651465e-4};
    Strain_t E{strain_amp * Strain_t::Identity()};
    E(1, 1) = -1.7 * E(1, 1);
    E(0, 1) = 2.1241e-3;
    E(1, 0) = E(0, 1);

    if (mdim == 3) {
      E(0, 2) = 1.12345e-3;
      E(2, 0) = E(0, 2);
      E(1, 2) = 1.1356e-3;
      E(2, 1) = E(1, 2);
      E(2, 2) = 0.6 * E(2, 2);
    }

    Strain_t S_dam{Strain_t::Zero()};
    Stiffness_t C_mat{std::get<1>(mat_dam.evaluate_stress_tangent(E, 0))};
    mat_dam.save_history_variables();

    for (auto && step : Fix::steps) {
      Stiffness_t C_estim_inward{evaluator.estimate_tangent(
          E, Formulation::small_strain, step, FiniteDiff::inward)};
      Stiffness_t C_estim_outward{evaluator.estimate_tangent(
          E, Formulation::small_strain, step, FiniteDiff::outward)};

      Strain_t out_incr{0.5 * step * Strain_t::Ones()};
      for (Index_t i = 0; i < mdim; ++i) {
        for (Index_t j = 0; j < mdim; ++j) {
          if (E(i, j) < 0) {
            out_incr(i, j) *= -1;
          }
        }
      }

      Stiffness_t C_mat_out{
          std::get<1>(mat_dam.evaluate_stress_tangent(E + out_incr, 0))};

      Stiffness_t C_mat_in{
          std::get<1>(mat_dam.evaluate_stress_tangent(E - out_incr, 0))};

      auto err_tmp_out{rel_error(C_estim_outward, C_mat_out)};
      auto err_tmp_in{rel_error(C_estim_inward, C_mat_in)};

      // the inward finite difference estimation and the backward material
      // tangent evaluation are actually linear elastic steps and therefore
      // can satisfy the get_tol_diff() tolerance easily.
      BOOST_CHECK_LT(err_tmp_in, Fix::get_tol_diff());
      if (err_tmp_in > Fix::get_tol_diff()) {
        std::cout << "finite_diff_step_size:" << step << "\n";
        std::cout << "the C calculation error is :" << err_tmp_in << "\n";
        std::cout << "C_estim_inward:" << std::endl
                  << C_estim_inward << std::endl;
        std::cout << "C_backward:" << std::endl << C_mat_in << std::endl;
        std::cout << "DIFF_inward:" << std::endl
                  << C_estim_inward - C_mat_in << std::endl;
      }

      // Apparently, the reason that using get_tol_diff tolerance is not
      // possible here is that in tangent eigen-value and eigen-vector
      // calculation is carried out via numerical solver that results in
      // numerical errors that might add up and exceed the tolerance here;
      // however, the validity of tol_out defined here shows that tangent
      // evaluation error decays linearly w.r.t. to finite difference step size.
      Real tol_out{Fix::get_tol_diff() * (step / 1.e-10)};
      BOOST_CHECK_LT(err_tmp_out, tol_out);
      if (err_tmp_out > tol_out) {
        std::cout << "finite_diff_step_size:" << step << "\n";
        std::cout << "the C calculation error is :" << err_tmp_out << "\n";
        std::cout << "C_estim_outward:" << std::endl
                  << C_estim_outward << std::endl;
        std::cout << "C_forward:" << std::endl << C_mat_out << std::endl;
        std::cout << "DIFF_outward:" << std::endl
                  << C_estim_outward - C_mat_out << std::endl;
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END()
}  // namespace muSpectre
