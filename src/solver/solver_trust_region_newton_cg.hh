/**
 * @file   solver_trust_region_newton_cg.hh
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

#include "solver_single_physics.hh"
#include "krylov_solver_trust_region_base.hh"
#include "projection/projection_base.hh"

#include <libmugrid/units.hh>

#ifndef SRC_SOLVER_SOLVER_TRUST_REGION_NEWTON_CG_HH_
#define SRC_SOLVER_SOLVER_TRUST_REGION_NEWTON_CG_HH_

namespace muSpectre {

  class SolverTrustRegionNewtonCG : public SolverSinglePhysics {
    using Parent = SolverSinglePhysics;
    using Gradient_t = muFFT::Gradient_t;
    using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using Matrix_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    using EigenStrainFunc_ref = Parent::EigenStrainFunc_ref;
    using CellExtractFieldFunc_ref = Parent::CellExtractFieldFunc_ref;

   public:
    //! Default constructor
    SolverTrustRegionNewtonCG() = delete;

    //! constructor
    SolverTrustRegionNewtonCG(
        std::shared_ptr<CellData> cell_data,
        std::shared_ptr<KrylovSolverTrustRegionBase> krylov_solver,
        const muGrid::Verbosity & verbosity, const Real & newton_tol,
        const Real & equil_tol, const Uint & max_iter,
        const Real & max_trust_radius, const Real & eta);

    //! constructor
    SolverTrustRegionNewtonCG(
        std::shared_ptr<CellData> cell_data,
        std::shared_ptr<KrylovSolverTrustRegionBase> krylov_solver,
        const muGrid::Verbosity & verbosity, const Real & newton_tol,
        const Real & equil_tol, const Uint & max_iter,
        const Real & max_trust_radius, const Real & eta,
        const Gradient_t & gradient);

    //! Copy constructor
    SolverTrustRegionNewtonCG(const SolverTrustRegionNewtonCG & other) = delete;

    //! Move constructor
    SolverTrustRegionNewtonCG(SolverTrustRegionNewtonCG && other) = default;

    //! Destructor
    virtual ~SolverTrustRegionNewtonCG() = default;

    //! Copy assignment operator
    SolverTrustRegionNewtonCG &
    operator=(const SolverTrustRegionNewtonCG & other) = delete;

    //! Move assignment operator
    SolverTrustRegionNewtonCG &
    operator=(SolverTrustRegionNewtonCG && other) = delete;

    using Parent::solve_load_increment;
    //! solve for a single increment of strain
    OptimizeResult solve_load_increment(
        const LoadStep & load_step,
        EigenStrainFunc_ref eigen_strain_func = muGrid::nullopt,
        CellExtractFieldFunc_ref cell_extract_func = muGrid::nullopt) final;

    //! return the number of degrees of freedom of the solver problem
    Index_t get_nb_dof() const final;

    //! implementation of the action of the stiffness matrix
    void action_increment(EigenCVec_t delta_grad, const Real & alpha,
                          EigenVec_t del_flux) final;

    //! initialise cell data for this solver
    void initialise_cell() final;

    //! return a const ref to the projection implementation
    const ProjectionBase & get_projection() const;

   protected:
    void initialise_eigen_strain_storage();
    bool has_eigen_strain_storage() const;
    /**
     * statically dimensioned worker for evaluating the incremental tangent
     * operator
     */
    template <Dim_t DimM>
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

    //! holder of the previous macroscopic load
    Eigen::MatrixXd previous_macro_load{};
    std::array<Index_t, 2> grad_shape{};

    std::shared_ptr<KrylovSolverTrustRegionBase> krylov_solver;
    Real newton_tol;
    Real equil_tol;
    Uint max_iter;

    //! maximum radius of trust region
    Real max_trust_radius;
    //! threshold used in accepting or rejecting a sub-problem solution
    Real eta;

    //! The gradient operator used in the projection
    std::shared_ptr<Gradient_t> gradient;
    Index_t nb_quad_pts{1};
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_SOLVER_TRUST_REGION_NEWTON_CG_HH_
