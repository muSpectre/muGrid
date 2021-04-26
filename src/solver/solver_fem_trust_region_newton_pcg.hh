/**
 * @file   solver_fem_trust_region_newton_pcg.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   31 Aug 2020
 *
 * @brief  Newton-PCG solver for single-physics finite-element problems
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

#include "solver_fem_trust_region_newton_cg.hh"
#include "krylov_solver_trust_region_pcg.hh"

// #include "krylov_solver_pcg.hh"

#include "projection/discretisation.hh"

#ifndef SRC_SOLVER_SOLVER_FEM_TRUST_REGION_NEWTON_PCG_HH_
#define SRC_SOLVER_SOLVER_FEM_TRUST_REGION_NEWTON_PCG_HH_

namespace muSpectre {

  class SolverFEMTrustRegionNewtonPCG : public SolverFEMTrustRegionNewtonCG {
   public:
    using Parent = SolverFEMTrustRegionNewtonCG;
    //! Default constructor
    SolverFEMTrustRegionNewtonPCG() = delete;

    //!
    SolverFEMTrustRegionNewtonPCG(
        std::shared_ptr<Discretisation> discretisation,
        std::shared_ptr<KrylovSolverTrustRegionPCG> krylov_solver,
        const muGrid::Verbosity & verbosity, const Real & newton_tol,
        const Real & equil_tol, const Uint & max_iter,
        const Real & max_trust_radius, const Real & eta);

    //! Copy constructor
    SolverFEMTrustRegionNewtonPCG(const SolverFEMTrustRegionNewtonPCG & other) =
        delete;

    //! Move constructor
    SolverFEMTrustRegionNewtonPCG(SolverFEMTrustRegionNewtonPCG && other) =
        default;

    //! Destructor
    virtual ~SolverFEMTrustRegionNewtonPCG() = default;

    //! Copy assignment operator
    SolverFEMTrustRegionNewtonPCG &
    operator=(const SolverFEMTrustRegionNewtonPCG & other) = delete;

    //! Move assignment operator
    SolverFEMTrustRegionNewtonPCG &
    operator=(SolverFEMTrustRegionNewtonPCG && other) = delete;

    //! set reference_material, sets also the preconditioner
    void set_reference_material(
        Eigen::Ref<const Eigen::MatrixXd> material_properties);

   protected:
    Eigen::MatrixXd ref_material{};
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_SOLVER_FEM_TRUST_REGION_NEWTON_PCG_HH_
