/**
 * @file   solver_single_physics_projection_base.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   13 Feb 2022
 *
 * @brief  Base class for single physics projection based solvers
 *
 * Copyright © 2022 Ali Falsafi
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

#include "projection/projection_base.hh"
#include "materials/material_mechanics_base.hh"

#ifndef SRC_SOLVER_SOLVER_SINGLE_PHYSICS_PROJECTION_BASE_HH_
#define SRC_SOLVER_SOLVER_SINGLE_PHYSICS_PROJECTION_BASE_HH_

namespace muSpectre {

  class SolverSinglePhysicsProjectionBase : public SolverSinglePhysics {
   public:
    using Parent = SolverSinglePhysics;
    using Gradient_t = ProjectionBase::Gradient_t;
    using Weights_t = ProjectionBase::Weights_t;
    using EigenStrainFunc_ref = Parent::EigenStrainFunc_ref;
    using CellExtractFieldFunc_ref = Parent::CellExtractFieldFunc_ref;
    using RealField = muGrid::TypedField<Real>;

    //! Ref to input/output vector
    using EigenVecRef = Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, 1>>;
    //! Ref to input/output matrix
    using EigenMatRef =
        Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>;

    //! Ref to input/output vector
    using EigenVec_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    //! Ref to input/output matrix
    using EigenMat_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

    //! Ref to input/output matrix
    using EigenArr_t = Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic>;

    //! Default constructor
    SolverSinglePhysicsProjectionBase() = delete;

    //! constructor with units of input fields
    SolverSinglePhysicsProjectionBase(
        std::shared_ptr<CellData> cell_data,
        const muGrid::Verbosity & verbosity, const Real & newton_tol,
        const Real & equil_tol, const Uint & max_iter,
        const Gradient_t & gradient,
        const Weights_t & weights,
        const MeanControl & mean_control = MeanControl::StrainControl);

    //! constructor with units of input fields
    SolverSinglePhysicsProjectionBase(
        std::shared_ptr<CellData> cell_data,
        const muGrid::Verbosity & verbosity, const Real & newton_tol,
        const Real & equil_tol, const Uint & max_iter,
        const MeanControl & mean_control = MeanControl::StrainControl);

    //! Copy constructor
    SolverSinglePhysicsProjectionBase(
        const SolverSinglePhysicsProjectionBase & other) = delete;

    //! Move constructor
    SolverSinglePhysicsProjectionBase(
        SolverSinglePhysicsProjectionBase && other) = default;

    //! Destructor
    virtual ~SolverSinglePhysicsProjectionBase() = default;

    //! Copy assignment operator
    SolverSinglePhysicsProjectionBase &
    operator=(const SolverSinglePhysicsProjectionBase & other) = delete;

    //! Move assignment operator
    SolverSinglePhysicsProjectionBase &
    operator=(SolverSinglePhysicsProjectionBase && other) = delete;

    //! return the rank of the displacement field for this PhysicsDomain
    Index_t get_displacement_rank() const;

    //! return the number of degrees of freedom of the solver problem
    Index_t get_nb_dof() const final;

    //! return the projection operator
    const ProjectionBase & get_projection();

    //! evaluated gradient field
    MappedField_t & get_eval_grad() const;

    //! getter for rhs field
    MappedField_t & get_rhs() final;

    //! getter for increment field (field in this solver)
    MappedField_t & get_incr() final;

    //! computes the effective tangent of the cell at the current eq. state:
    EigenMat_t compute_effective_stiffness();

    //! create unit test strains
    EigenMat_t create_unit_test_strain();

    //! implementation of the action of the stiffness matrix
    void action_increment(EigenCVecRef delta_grad, const Real & alpha,
                          EigenVecRef del_flux) final;

   protected:
    void initialise_cell_worker();
    void initialise_eigen_strain_storage();
    bool has_eigen_strain_storage() const;
    /**
     * statically dimensioned worker for evaluating the incremental tangent
     * operator
     */
    template <Dim_t DimM>
    static void action_increment_worker_prep(
        const muGrid::TypedFieldBase<Real> & delta_strain,
        const muGrid::TypedFieldBase<Real> & tangent, const Real & alpha,
        muGrid::TypedFieldBase<Real> & delta_stress,
        const Index_t & displacement_rank);

    /**
     * statically dimensioned worker for evaluating the incremental tangent
     * operator
     */
    template <Dim_t DimM, Index_t DisplacementRank>
    static void
    action_increment_worker(const muGrid::TypedFieldBase<Real> & delta_strain,
                            const muGrid::TypedFieldBase<Real> & tangent,
                            const Real & alpha,
                            muGrid::TypedFieldBase<Real> & delta_stress);

    //! create a mechanics projection
    template <Dim_t Dim>
    void create_mechanics_projection_worker();
    void create_mechanics_projection();

    //! create a generic gradient projection
    void create_gradient_projection();

    //! making a mapped_field by making a new field or fetching an existing
    //! name with the unique_name from the field_collection
    RealField & fetch_or_register_field(const std::string & unique_name,
                                        const Index_t & nb_rows,
                                        const Index_t & nb_cols,
                                        muGrid::FieldCollection & collection,
                                        const std::string & sub_division_tag);

    std::shared_ptr<ProjectionBase> projection{nullptr};

    using MappedField_t =
        muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>>;
    //! Newton loop gradient (or strain) increment
    std::shared_ptr<MappedField_t> grad_incr{nullptr};
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
    //! Tangent moduli field
    std::shared_ptr<MappedField_t> tangent{nullptr};
    //! right-hand-side field
    std::shared_ptr<MappedField_t> rhs{nullptr};

    Eigen::MatrixXd previous_macro_load{};
    std::array<Index_t, 2> grad_shape{};

    Real newton_tol;
    Real equil_tol;
    Uint max_iter;

    //! The gradient operator used in the projection
    std::shared_ptr<Gradient_t> gradient;

    //! The weights used in the projection;
    std::shared_ptr<Weights_t> weights;

    //! number of quadrature points
    Index_t nb_quad_pts{1};

    //! The type of the mean controlled variable on the RVE:
    MeanControl mean_control;
  };
}  // namespace muSpectre

#endif  // SRC_SOLVER_SOLVER_SINGLE_PHYSICS_PROJECTION_BASE_HH_
