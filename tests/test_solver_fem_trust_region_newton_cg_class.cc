/**
 * @file   test_solver_fem_trust_region_newton_cg_class.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   01 Sep 2020
 *
 * @brief  tests for the un-preconditioned newton-cg FEM solver class
 *
 * Copyright © 2020 Till Junge, Martin Ladecký
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
#include "test_cell_data.hh"
#include "libmugrid/test_goodies.hh"

#include "projection/fem_library.hh"
#include "projection/discretisation.hh"
#include "solver/solver_fem_newton_cg.hh"
#include "solver/solver_fem_newton_pcg.hh"
#include "solver/solver_fem_trust_region_newton_cg.hh"
#include "solver/solver_fem_trust_region_newton_pcg.hh"

#include "solver/solvers.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/material_dunant.hh"

#include <cell/cell_factory.hh>
#include <solver/krylov_solver_cg.hh>
#include <solver/krylov_solver_trust_region_cg.hh>
#include <solver/krylov_solver_trust_region_pcg.hh>
#include <solver/krylov_solver_pcg.hh>

#include <cstdlib>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(fem_trust_region_newton_cg_solverclass);

  BOOST_FIXTURE_TEST_CASE(constructor_test, CellDataFixtureSquare<twoD>) {
    using Fix = CellDataFixture<twoD>;
    auto stencil{FEMLibrary::linear_triangle_straight(this->cell_data)};
    auto discretisation{std::make_shared<Discretisation>(stencil)};
    MaterialLinearElastic1<Fix::SpatialDim>::make(this->cell_data, "material",
                                                  4, .3);
    auto krylov_solver{std::make_shared<KrylovSolverCG>(1e-8, 100)};
    BOOST_TEST_CHECKPOINT("Before constructor");
    auto solver{std::make_shared<SolverFEMNewtonCG>(
        discretisation, krylov_solver, muGrid::Verbosity::Silent, 1e-10, 1e-10,
        100)};
  }

  BOOST_FIXTURE_TEST_CASE(precond_solver_test, CellDataFixtureSquare<twoD>) {
    std::cout << "number of grid points: "
              << this->cell_data->get_nb_domain_grid_pts() << "\n";

    auto stencil{FEMLibrary::bilinear_quadrangle(this->cell_data)};
    auto discretisation{std::make_shared<Discretisation>(stencil)};

    constexpr Formulation Form{Formulation::small_strain};
    using Fix = CellDataFixtureSquare<twoD>;
    const Real young{2.0e10};
    // const Real young2{0.8 * 2.0e10};
    const Real poisson{0.0};
    const Real alpha{2.e-1};
    const Real kappa{1.e-1};
    const Real eta{1.e-4};
    const Real tr{1.e-2};
    const Real strain_mean{1.2e-1};
    using MatElastic_t = MaterialLinearElastic1<Fix::SpatialDim>;
    // using MatDamage_t = MaterialLinearElastic1<Fix::SpatialDim>;
    using MatDamage_t = MaterialDunant<Fix::SpatialDim>;
    // this->cell_data->set_nb_quad_pts(OneQuadPt);

    auto & dam{MatDamage_t::make(this->cell_data, "damage", young, poisson,
                                 kappa, alpha)};
    // auto & dam{MatDamage_t::make(this->cell_data, "damage", young2,
    // poisson)};
    auto & elas{MatElastic_t::make(this->cell_data, "elastic", young, poisson)};
    {
      Index_t nb_dam{this->cell_data->get_nb_domain_grid_pts()[0]};
      for (auto && index_pixel : this->cell_data->get_pixels().enumerate()) {
        auto && index{std::get<0>(index_pixel)};
        if (nb_dam) {
          --nb_dam;
          dam.add_pixel(index);
        } else {
          elas.add_pixel(index);
        }
      }
    }

    BOOST_TEST_CHECKPOINT("after material assignment");

    constexpr Real cg_tol{1e-8}, newton_tol{1e-8}, equil_tol{2e-7};
    const Uint maxiter{static_cast<Uint>(this->cell_data->get_spatial_dim()) *
                       20};
    constexpr Verbosity verbose{Verbosity::Full};

    auto krylov_solver{std::make_shared<KrylovSolverTrustRegionPCG>(
        cg_tol, maxiter, tr, verbose, ResetCG::iter_count)};
    auto solver{std::make_shared<SolverFEMTrustRegionNewtonPCG>(
        discretisation, krylov_solver, verbose, newton_tol, equil_tol, maxiter,
        tr, eta)};

    solver->set_formulation(Form);
    solver->initialise_cell();

    BOOST_TEST_CHECKPOINT("before ref material assignment");
    solver->get_set_eval_grad().get_map() =
        Eigen::MatrixXd::Identity(twoD, twoD);
    solver->evaluate_stress_tangent();
    auto ref_material{solver->get_tangent().get_map().mean()};
    solver->set_reference_material(ref_material);

    BOOST_TEST_CHECKPOINT("after ref material assignment");
    auto && symmetric{[](Eigen::MatrixXd mat) -> Eigen::MatrixXd {
      return 0.5 * (mat + mat.transpose());
    }};

    Eigen::MatrixXd strain{
        symmetric(Eigen::MatrixXd::Zero(Fix::SpatialDim, Fix::SpatialDim)) / 2};
    strain(1, 1) = strain_mean;

    BOOST_TEST_CHECKPOINT("before load increment");

    solver->solve_load_increment(strain);
    // std::cout << new_result.grad << "\n";
    // std::cout << new_result.stress << "\n";
    BOOST_TEST_CHECKPOINT("after load increment");
  }

  BOOST_FIXTURE_TEST_CASE(pre_conditioned_constructor_test,
                          CellDataFixtureSquare<twoD>) {
    using Fix = CellDataFixture<twoD>;
    auto stencil{FEMLibrary::linear_triangle_straight(this->cell_data)};
    auto discretisation{std::make_shared<Discretisation>(stencil)};
    MaterialLinearElastic1<Fix::SpatialDim>::make(this->cell_data, "material",
                                                  4, .3);
    auto krylov_solver{std::make_shared<KrylovSolverPCG>(1e-8, 100)};
    BOOST_TEST_CHECKPOINT("Before constructor");
    auto solver{std::make_shared<SolverFEMNewtonCG>(
        discretisation, krylov_solver, muGrid::Verbosity::Full, 1e-10, 1e-10,
        100)};
  }

  BOOST_FIXTURE_TEST_CASE(pre_conditioned_solver_test,
                          CellDataFixtureSquare<twoD>) {
    std::cout << "number of grid points: "
              << this->cell_data->get_nb_domain_grid_pts() << "\n";

    auto stencil{FEMLibrary::bilinear_quadrangle(this->cell_data)};
    auto discretisation{std::make_shared<Discretisation>(stencil)};

    constexpr Formulation Form{Formulation::small_strain};
    using Fix = CellDataFixtureSquare<twoD>;
    const Real young{2.0e10};
    // const Real young2{0.8 * 2.0e10};
    const Real poisson{0.0};
    const Real alpha{2.e-1};
    const Real kappa{1.e-1};
    const Real eta{1.e-4};
    const Real tr{1.e-2};
    const Real strain_mean{1.2e-1};
    using MatElastic_t = MaterialLinearElastic1<Fix::SpatialDim>;
    // using MatDamage_t = MaterialLinearElastic1<Fix::SpatialDim>;
    using MatDamage_t = MaterialDunant<Fix::SpatialDim>;
    // this->cell_data->set_nb_quad_pts(OneQuadPt);

    auto & dam{MatDamage_t::make(this->cell_data, "damage", young, poisson,
                                 kappa, alpha)};
    // auto & dam{MatDamage_t::make(this->cell_data, "damage", young2,
    // poisson)};
    auto & elas{MatElastic_t::make(this->cell_data, "elastic", young, poisson)};
    {
      Index_t nb_dam{this->cell_data->get_nb_domain_grid_pts()[0]};
      for (auto && index_pixel : this->cell_data->get_pixels().enumerate()) {
        auto && index{std::get<0>(index_pixel)};
        if (nb_dam) {
          --nb_dam;
          dam.add_pixel(index);
        } else {
          elas.add_pixel(index);
        }
      }
    }

    BOOST_TEST_CHECKPOINT("after material assignment");

    constexpr Real cg_tol{1e-8}, newton_tol{1e-8}, equil_tol{1e-7};
    const Uint maxiter{static_cast<Uint>(this->cell_data->get_spatial_dim()) *
                       20};
    constexpr Verbosity verbose{Verbosity::Silent};

    auto krylov_solver{std::make_shared<KrylovSolverTrustRegionCG>(
        cg_tol, maxiter, tr, verbose)};
    auto solver{std::make_shared<SolverFEMTrustRegionNewtonCG>(
        discretisation, krylov_solver, verbose, newton_tol, equil_tol, maxiter,
        tr, eta)};

    auto && symmetric{[](Eigen::MatrixXd mat) -> Eigen::MatrixXd {
      return 0.5 * (mat + mat.transpose());
    }};

    Eigen::MatrixXd strain{
        symmetric(Eigen::MatrixXd::Zero(Fix::SpatialDim, Fix::SpatialDim)) / 2};
    strain(1, 1) = strain_mean;

    solver->set_formulation(Form);
    solver->initialise_cell();

    BOOST_TEST_CHECKPOINT("before load increment");

    solver->solve_load_increment(strain);
    // std::cout << new_result.grad << "\n";
    // std::cout << new_result.stress << "\n";
    BOOST_TEST_CHECKPOINT("after load increment");
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muSpectre
