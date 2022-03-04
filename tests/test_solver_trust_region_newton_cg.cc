/**
 * @file   test_solver_trust_region_newton_cg.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   29 Mär 2021
 *
 * @brief  description
 *
 * Copyright © 2021 Ali Falsafi
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

#include "solver/solver_newton_cg.hh"
#include "solver/solver_trust_region_newton_cg.hh"
#include "solver/solvers.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/material_dunant.hh"

#include <cell/cell_factory.hh>
#include <solver/krylov_solver_eigen.hh>
#include <solver/krylov_solver_trust_region_cg.hh>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(trust_region_newton_cg_solverclass);

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, Fix, CellDataFixtures,
                                   Fix) {
    this->cell_data->set_nb_quad_pts(OneQuadPt);
    MaterialLinearElastic1<Fix::SpatialDim>::make(this->cell_data, "material",
                                                  4, .3);
    std::shared_ptr<KrylovSolverTrustRegionCG> krylov_solver{
        std::make_shared<KrylovSolverTrustRegionCG>(1e-8, 100, 1.0)};
    // std::shared_ptr<KrylovSolverTrustRegionCG> krylov_solver{nullptr};
    auto solver{std::make_shared<SolverTrustRegionNewtonCG>(
        this->cell_data, krylov_solver, muGrid::Verbosity::Full, 1e-10, 1e-10,
        100, 10.0, 1e-3)};
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(solver_test, Fix, CellDataFixtureSquares,
                                   Fix) {
    const Real young{2.0e10};
    // const Real young2{0.8 * 2.0e10};
    const Real poisson{0.0};
    const Real alpha{2.e-1};
    const Real kappa{1.e-1};
    const Real eta{1.e-4};
    const Real tr{1.e0};
    const Real strain_mean{1.2e-1};
    using MatElastic_t = MaterialLinearElastic1<Fix::SpatialDim>;
    // using MatDamage_t = MaterialLinearElastic1<Fix::SpatialDim>;
    using MatDamage_t = MaterialDunant<Fix::SpatialDim>;
    this->cell_data->set_nb_quad_pts(OneQuadPt);
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

    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equil_tol{1e-10};
    const Uint maxiter{static_cast<Uint>(this->cell_data->get_spatial_dim()) *
                       10};
    constexpr Verbosity verbose{Verbosity::Full};

    std::shared_ptr<KrylovSolverTrustRegionCG> krylov_solver{
        std::make_shared<KrylovSolverTrustRegionCG>(cg_tol, maxiter, tr,
                                                    verbose)};
    auto solver{std::make_shared<SolverTrustRegionNewtonCG>(
        this->cell_data, krylov_solver, verbose, newton_tol, equil_tol, maxiter,
        tr, eta)};

    auto homo_solver{std::make_shared<SolverTrustRegionNewtonCG>(
        this->cell_data, krylov_solver, verbose, newton_tol, equil_tol, maxiter,
        tr, eta)};

    auto && symmetric{[](Eigen::MatrixXd mat) -> Eigen::MatrixXd {
      return 0.5 * (mat + mat.transpose());
    }};

    Eigen::MatrixXd strain{
        symmetric(Eigen::MatrixXd::Zero(Fix::SpatialDim, Fix::SpatialDim)) / 2};
    strain(1, 1) = strain_mean;
    // strain(1, 1) = strain_mean;
    solver->set_formulation(Formulation::small_strain);
    solver->initialise_cell();

    homo_solver->set_formulation(Formulation::small_strain);
    homo_solver->initialise_cell();

    BOOST_TEST_CHECKPOINT("before load increment");
    std::cout << std::endl
              << "strain:" << std::endl
              << strain << std::endl
              << std::endl;
    std::cout << std::endl
              << "symmetric(strain):" << std::endl
              << symmetric(strain) << std::endl
              << std::endl;
    auto && new_result{solver->solve_load_increment(strain)};
    auto && C_eff{homo_solver->compute_effective_stiffness()};

    std::cout << new_result.grad << "\n";
    std::cout << new_result.stress << "\n";
    BOOST_TEST_CHECKPOINT("after load increment");
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
