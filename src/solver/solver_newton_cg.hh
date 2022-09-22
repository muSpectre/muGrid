/**
 * @file   solver_newton_cg.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   10 Jul 2020
 *
 * @brief  Newton-CG solver for single-physics problems
 *
 * Copyright © 2020 Till Junge
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

#include "solver_single_physics_projection_base.hh"

#include <libmugrid/units.hh>

#ifndef SRC_SOLVER_SOLVER_NEWTON_CG_HH_
#define SRC_SOLVER_SOLVER_NEWTON_CG_HH_

namespace muSpectre {

  class SolverNewtonCG : public SolverSinglePhysicsProjectionBase {
    using Parent = SolverSinglePhysicsProjectionBase;
    using Gradient_t = ProjectionBase::Gradient_t;
    using Weights_t = ProjectionBase::Weights_t;
    using EigenStrainFunc_ref = Parent::EigenStrainFunc_ref;
    using CellExtractFieldFunc_ref = Parent::CellExtractFieldFunc_ref;

   public:
    //! Default constructor
    SolverNewtonCG() = delete;

    //! constructor
    SolverNewtonCG(
        std::shared_ptr<CellData> cell_data,
        std::shared_ptr<KrylovSolverBase> krylov_solver,
        const muGrid::Verbosity & verbosity, const Real & newton_tol,
        const Real & equil_tol, const Uint & max_iter,
        const Gradient_t & gradient, const Weights_t & weights,
        const MeanControl & mean_control = MeanControl::StrainControl);

    //! constructor
    SolverNewtonCG(
        std::shared_ptr<CellData> cell_data,
        std::shared_ptr<KrylovSolverBase> krylov_solver,
        const muGrid::Verbosity & verbosity, const Real & newton_tol,
        const Real & equil_tol, const Uint & max_iter,
        const MeanControl & mean_control = MeanControl::StrainControl);

    //! Copy constructor
    SolverNewtonCG(const SolverNewtonCG & other) = delete;

    //! Move constructor
    SolverNewtonCG(SolverNewtonCG && other) = default;

    //! Destructor
    virtual ~SolverNewtonCG() = default;

    //! Copy assignment operator
    SolverNewtonCG & operator=(const SolverNewtonCG & other) = delete;

    //! Move assignment operator
    SolverNewtonCG & operator=(SolverNewtonCG && other) = delete;

    using Parent::solve_load_increment;
    //! solve for a single increment of strain
    OptimizeResult solve_load_increment(
        const LoadStep & load_step,
        EigenStrainFunc_ref eigen_strain_func = muGrid::nullopt,
        CellExtractFieldFunc_ref cell_extract_func = muGrid::nullopt) final;

    //! initialise cell data for this solver
    void initialise_cell() final;

    //! getter for Krylov solver object
    KrylovSolverBase & get_krylov_solver() final;

   protected:
    std::shared_ptr<KrylovSolverBase> krylov_solver;
  };
}  // namespace muSpectre

#endif  // SRC_SOLVER_SOLVER_NEWTON_CG_HH_
