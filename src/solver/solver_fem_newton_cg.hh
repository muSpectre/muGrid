/**
 * @file   solver_fem_newton_cg.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   31 Aug 2020
 *
 * @brief  Newton-CG solver for single-physics finite-element problems
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

#include "solver_single_physics.hh"
#include "krylov_solver_base.hh"

#include "projection/discretisation.hh"

#ifndef SRC_SOLVER_SOLVER_FEM_NEWTON_CG_HH_
#define SRC_SOLVER_SOLVER_FEM_NEWTON_CG_HH_

namespace muSpectre {

  class SolverFEMNewtonCG : public SolverSinglePhysics {
   public:
    using Parent = SolverSinglePhysics;

    using EigenStrainFunc_ref = Parent::EigenStrainFunc_ref;
    using CellExtractFieldFunc_ref = Parent::CellExtractFieldFunc_ref;

    //! Default constructor
    SolverFEMNewtonCG() = delete;

    //!
    SolverFEMNewtonCG(std::shared_ptr<Discretisation> discretisation,
                      std::shared_ptr<KrylovSolverBase> krylov_solver,
                      const muGrid::Verbosity & verbosity,
                      const Real & newton_tol, const Real & equil_tol,
                      const Uint & max_iter);

    //! Copy constructor
    SolverFEMNewtonCG(const SolverFEMNewtonCG & other) = delete;

    //! Move constructor
    SolverFEMNewtonCG(SolverFEMNewtonCG && other) = default;

    //! Destructor
    virtual ~SolverFEMNewtonCG() = default;

    //! Copy assignment operator
    SolverFEMNewtonCG & operator=(const SolverFEMNewtonCG & other) = delete;

    //! Move assignment operator
    SolverFEMNewtonCG & operator=(SolverFEMNewtonCG && other) = delete;

    using Parent::solve_load_increment;
    //! solve for a single increment of strain
    OptimizeResult solve_load_increment(
        const LoadStep & load_step,
        EigenStrainFunc_ref eigen_strain_func = muGrid::nullopt,
        CellExtractFieldFunc_ref cell_extract_func = muGrid::nullopt) final;

    //! return the number of degrees of freedom of the solver problem
    Index_t get_nb_dof() const final;

    //! implementation of the action of the stiffness matrix
    void action_increment(EigenCVecRef delta_u, const Real & alpha,
                          EigenVecRef delta_f) final;

    //! initialise cell data for this solver
    void initialise_cell() final;

    //! return the rank of the displacement field for this PhysicsDomain
    Index_t get_displacement_rank() const;

    //! Displacement field
    const MappedField_t & get_disp_fluctuation() const;

    //! getter for rhs field
    MappedField_t & get_rhs() final;

    //! getter for increment field (field in this solver)
    MappedField_t & get_incr() final;

    //! getter for Krylov solver object
    KrylovSolverBase & get_krylov_solver() final;

   protected:
    void initialise_eigen_strain_storage();
    bool has_eigen_strain_storage() const;
    using MappedField_t =
        muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>>;
    //! Newton loop displacement fluctuation increment δũ
    std::shared_ptr<MappedField_t> disp_fluctuation_incr{nullptr};
    //! displacement field fluctuation ũ
    std::shared_ptr<MappedField_t> disp_fluctuation{nullptr};
    //! Gradient (or strain) field
    std::shared_ptr<MappedField_t> grad{nullptr};
    /**
     * Gradient (or strain) field evaluated by the materials this pointer points
     * usually to the same field as `grad` does, but in the case of problems
     * with an eigen strain, these point to different fields
     */
    std::shared_ptr<MappedField_t> eval_grad{nullptr};
    //! Flux (or stress) field
    std::shared_ptr<MappedField_t> flux{nullptr};
    //! Force field
    std::shared_ptr<MappedField_t> force{nullptr};
    //! Tangent moduli field
    std::shared_ptr<MappedField_t> tangent{nullptr};
    //! right-hand-side field
    std::shared_ptr<MappedField_t> rhs{nullptr};

    Eigen::MatrixXd previous_macro_load{};
    std::array<Index_t, 2> displacement_shape{};
    std::array<Index_t, 2> grad_shape{};

    std::shared_ptr<KrylovSolverBase> krylov_solver;
    std::shared_ptr<Discretisation> discretisation;
    StiffnessOperator K;
    Real newton_tol;
    Real equil_tol;
    Uint max_iter;
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_SOLVER_FEM_NEWTON_CG_HH_
