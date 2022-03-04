/**
 * @file  test_material_dunant.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   04 May 2020
 *
 * @brief  the test for the MaterialDunant
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

#include "materials/material_dunant.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/iterable_proxy.hh"

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(material_dunant)
  template <Dim_t Dim>
  struct MaterialFixture {
    using MatDam = MaterialDunant<Dim>;
    using MatLin = MaterialLinearElastic1<Dim>;
    const Real young{1.0e10};
    const Real poisson{0.2};
    const Real kappa{1e-3};
    const Real kappa_var{kappa * 1e-3};
    const Real alpha{0.5};

    //! Constructor
    MaterialFixture()
        : mat_dam_eval{MatDam::make_evaluator(young, poisson, kappa, alpha)},
          mat_lin_eval{MatLin::make_evaluator(young, poisson)} {}

    constexpr static Dim_t mdim() { return Dim; }
    constexpr static Dim_t sdim() { return mdim(); }
    constexpr static Dim_t NbQuadPts() { return 1; }

    std::tuple<std::shared_ptr<MatDam>, MaterialEvaluator<Dim>> mat_dam_eval;
    std::tuple<std::shared_ptr<MatLin>, MaterialEvaluator<Dim>> mat_lin_eval;
  };

  using mats = boost::mpl::list<MaterialFixture<twoD>, MaterialFixture<threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_add_pixel, Fix, mats, Fix) {
    auto & mat_dam{*std::get<0>(Fix::mat_dam_eval)};
    muGrid::testGoodies::RandRange<size_t> rng;

    const Dim_t nb_pixel{7}, box_size{17};
    for (Dim_t i = 0; i < nb_pixel; ++i) {
      const size_t c{rng.randval(0, box_size)};
      BOOST_CHECK_NO_THROW(mat_dam.add_pixel(c, Fix::kappa_var));
    }
  }

  template <Index_t Dim>
  struct MaterialFixtureFilled : public MaterialFixture<Dim> {
    using Parent = MaterialFixture<Dim>;
    using MatDam = typename Parent::MatDam;
    using MatLin = typename Parent::MatLin;

    constexpr static Index_t box_size{1};
    MaterialFixtureFilled() : Parent() {
      auto & mat_dam{*std::get<0>(Parent::mat_dam_eval)};
      auto & mat_lin{*std::get<0>(Parent::mat_lin_eval)};
      mat_dam.add_pixel(0);
      mat_lin.add_pixel(0);
      mat_dam.initialise();
      mat_lin.initialise();
      mat_dam.initialise();
      mat_lin.initialise();
    }

    constexpr static Real tol{1.6e-5};
    constexpr static Real get_tol() { return tol; }
    constexpr static Dim_t mdim() { return Parent::mdim(); }
    constexpr static Dim_t sdim() { return mdim(); }
    constexpr static Real step{1.e-8};
  };

  using mats_fill = boost::mpl::list<MaterialFixtureFilled<twoD>,
                                     MaterialFixtureFilled<threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evalaute_stress, Fix, mats_fill, Fix) {
    constexpr Dim_t mdim{Fix::mdim()}, sdim{Fix::sdim()};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;

    auto & mat_dam{*std::get<0>(Fix::mat_dam_eval)};
    auto & mat_lin{*std::get<0>(Fix::mat_lin_eval)};

    // create state fields
    muGrid::LocalFieldCollection coll{sdim};
    coll.set_nb_sub_pts(QuadPtTag, Fix::NbQuadPts());
    coll.add_pixel({0});
    coll.initialise();

    muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt> kappa_{
        "Kappa", coll, QuadPtTag};

    muGrid::MappedScalarField<Real, Mapping::Mut, IterUnit::SubPt> kappa_init_{
        "Kappa_init", coll, QuadPtTag};

    kappa_init_[0] = Fix::kappa;

    auto & kappa{kappa_.get_map()};
    kappa[0].current() = Fix::kappa;

    Strain_t F{1.5e-3 * Strain_t::Random() + Strain_t::Identity()};
    Strain_t E{0.5 * ((F * F.transpose()) - Strain_t::Identity())};

    kappa_.get_state_field().cycle();

    Strain_t S_dam{Strain_t::Zero()};
    Strain_t S_lin{Strain_t::Zero()};

    S_dam = mat_dam.evaluate_stress(E, kappa_[0], kappa_init_[0]);
    S_lin = mat_lin.evaluate_stress(E, 0);

    Real kap_ref{std::sqrt(muGrid::Matrices::ddot<Fix::mdim()>(E, E))};

    Real kap{kappa_[0].current()};

    Real err{rel_error(kap, kap_ref)};
    if (kap > Fix::kappa) {
      BOOST_CHECK_LT(err, Fix::get_tol());
    }

    Real dam{kap > Fix::kappa
                 ? (((1 + Fix::alpha) * (Fix::kappa / kap)) - Fix::alpha)
                 : 1.0};

    Strain_t S_dam_ref{dam * S_lin};
    Real err2{rel_error(S_dam, S_dam_ref)};

    BOOST_CHECK_LT(err2, Fix::get_tol());

    if (not(err2 <= Fix::get_tol())) {
      std::cout << "Dam:" << dam << "\n";
      std::cout << "S reference:"
                << "\n"
                << S_dam_ref << "\n";

      std::cout << "S material:"
                << "\n"
                << S_dam << "\n";
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evalaute_stress_tangent, Fix, mats_fill,
                                   Fix) {
    constexpr Dim_t mdim{Fix::mdim()};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;
    using Stiffness_t = muGrid::T4Mat<Real, mdim>;

    auto & mat_dam{*std::get<0>(Fix::mat_dam_eval)};
    auto & evaluator{std::get<1>(Fix::mat_dam_eval)};

    Real initial_strain{0.96e-3};
    Strain_t E{initial_strain * Strain_t::Identity()};
    Strain_t E_f{(initial_strain + Fix::step) * Strain_t::Identity()};
    Strain_t E_b{(initial_strain - Fix::step) * Strain_t::Identity()};

    Strain_t S_dam{Strain_t::Zero()};

    Stiffness_t C_mat{std::get<1>(mat_dam.evaluate_stress_tangent(E, 0))};

    mat_dam.save_history_variables();

    Stiffness_t C_mat_backward{
        std::get<1>(mat_dam.evaluate_stress_tangent(E_b, 0))};
    Stiffness_t C_mat_forward{
        std::get<1>(mat_dam.evaluate_stress_tangent(E_f, 0))};

    Stiffness_t C_estim_backward{evaluator.estimate_tangent(
        E, Formulation::small_strain, Fix::step, FiniteDiff::backward)};

    Stiffness_t C_estim_forward{evaluator.estimate_tangent(
        E, Formulation::small_strain, Fix::step, FiniteDiff::forward)};

    Real err_f{rel_error(C_estim_forward, C_mat_forward)};

    BOOST_CHECK_LT(err_f, Fix::get_tol());
    if (not(err_f <= Fix::get_tol())) {
      std::cout << "C estimated forward:"
                << "\n"
                << C_estim_forward << "\n";
      std::cout << "C material forward:"
                << "\n"
                << C_mat_forward << "\n";
    }

    Real err_b{rel_error(C_estim_backward, C_mat_backward)};
    BOOST_CHECK_LT(err_b, Fix::get_tol());
    if (not(err_b <= Fix::get_tol())) {
      std::cout << "C estimated backward:"
                << "\n"
                << C_estim_backward << "\n";

      std::cout << "C material backward:"
                << "\n"
                << C_mat_backward << "\n";
    }
  }

  BOOST_AUTO_TEST_SUITE_END()
}  // namespace muSpectre
