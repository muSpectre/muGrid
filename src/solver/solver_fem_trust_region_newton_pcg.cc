/**
 * @file   solver_fem_trust_region_newton_pcg.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   03 Sep 2020
 *
 * @brief  Implementation for Newton-PCG FEM solver
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

#include "solver_fem_trust_region_newton_pcg.hh"
#include "materials/material_mechanics_base.hh"
#include "projection/discrete_greens_operator.hh"

#include <libmugrid/grid_common.hh>

#include <iomanip>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  SolverFEMTrustRegionNewtonPCG::SolverFEMTrustRegionNewtonPCG(
      std::shared_ptr<Discretisation> discretisation,
      std::shared_ptr<KrylovSolverTrustRegionPCG> krylov_solver,
      const muGrid::Verbosity & verbosity, const Real & newton_tol,
      const Real & equil_tol, const Uint & max_iter,
      const Real & max_trust_radius, const Real & eta)
      : Parent{discretisation, krylov_solver, verbosity,        newton_tol,
               equil_tol,      max_iter,      max_trust_radius, eta} {}

  /* ---------------------------------------------------------------------- */
  void SolverFEMTrustRegionNewtonPCG::set_reference_material(
      Eigen::Ref<const Eigen::MatrixXd> material_properties) {
    this->ref_material = material_properties;
    auto pcg_krylov_solver{
        std::dynamic_pointer_cast<KrylovSolverTrustRegionPCG>(
            this->krylov_solver)};

    auto impulse_response{this->discretisation->compute_impulse_response(
        this->get_displacement_rank(), this->ref_material)};

    auto greens_operator{std::make_shared<DiscreteGreensOperator>(
        discretisation->get_cell()->get_FFT_engine(), *impulse_response,
        this->get_displacement_rank())};

    pcg_krylov_solver->set_preconditioner(greens_operator);
  }
}  // namespace muSpectre
