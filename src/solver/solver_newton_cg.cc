/**
 * @file   solver_newton_cg.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   14 Jul 2020
 *
 * @brief  implementation of Newton-CG solver class
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

#include "solver_newton_cg.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "projection/projection_small_strain.hh"
#include "materials/material_mechanics_base.hh"

#include <iomanip>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  SolverNewtonCG::SolverNewtonCG(
      std::shared_ptr<CellData> cell_data,
      std::shared_ptr<KrylovSolverBase> krylov_solver,
      const muGrid::Verbosity & verbosity, const Real & newton_tol,
      const Real & equil_tol, const Uint & max_iter)
      : Parent{cell_data, verbosity}, krylov_solver{krylov_solver},
        newton_tol{newton_tol}, equil_tol{equil_tol}, max_iter{max_iter} {}

  /* ---------------------------------------------------------------------- */
  void SolverNewtonCG::initialise_cell() {
    if (this->is_initialised) {
      return;
    }
    // make sure the number of subpoints is correct
    this->cell_data->get_fields().set_nb_sub_pts(QuadPtTag, OneQuadPt);
    this->cell_data->get_fields().set_nb_sub_pts(NodalPtTag, OneNode);
    //! check whether formulation has been set
    if (this->is_mechanics()) {
      if (this->get_formulation() == Formulation::not_set) {
        throw SolverError(
            "Can't run a mechanics calculation without knowing the "
            "formulation. please use the `set_formulation()` to "
            "choose between finite and small strain");
      } else {
        for (auto && mat :
             this->cell_data->get_domain_materials().at(this->domain)) {
          auto mech{std::dynamic_pointer_cast<MaterialMechanicsBase>(mat)};
          if (this->get_formulation() == Formulation::small_strain) {
            mech->check_small_strain_capability();
          }

          mech->set_formulation(this->get_formulation());
        }
      }
    }

    this->grad_shape =
        gradient_shape(this->domain.rank(), this->cell_data->get_spatial_dim(),
                       this->is_mechanics(), this->get_formulation());

    //! store solver fields in cell
    auto & field_collection{cell_data->get_fields()};
    // Corresponds to symbol δF or δε
    this->grad_incr = std::make_shared<MappedField_t>(
        "incrF", this->grad_shape[0], this->grad_shape[1], IterUnit::SubPt,
        field_collection, QuadPtTag);
    // Corresponds to symbol F or ε
    this->grad = std::make_shared<MappedField_t>(
        "grad", this->grad_shape[0], this->grad_shape[1], IterUnit::SubPt,
        field_collection, QuadPtTag);
    this->grads[this->domain] = this->grad;
    this->eval_grad = this->grad;
    // Corresponds to symbol P or σ
    this->flux = std::make_shared<MappedField_t>(
        "flux", this->grad_shape[0], this->grad_shape[1], IterUnit::SubPt,
        field_collection, QuadPtTag);
    this->fluxes[this->domain] = this->flux;
    // Corresponds to symbol K or C
    Index_t tangent_size{this->grad_shape[0] * this->grad_shape[1]};
    this->tangent = std::make_shared<MappedField_t>(
        "tangent", tangent_size, tangent_size, IterUnit::SubPt,
        field_collection, QuadPtTag);
    this->tangents[this->domain] = this->tangent;
    // field to store the rhs for cg calculations
    this->rhs = std::make_shared<MappedField_t>(
        "rhs", this->grad_shape[0], this->grad_shape[1], IterUnit::SubPt,
        field_collection, QuadPtTag);

    Eigen::MatrixXd default_grad_val{};
    if (this->is_mechanics()) {
      switch (this->get_formulation()) {
      case Formulation::finite_strain: {
        default_grad_val =
            Eigen::MatrixXd::Identity(this->grad_shape[0], this->grad_shape[1]);
        break;
      }
      case Formulation::small_strain: {
        default_grad_val =
            Eigen::MatrixXd::Zero(this->grad_shape[0], this->grad_shape[1]);
        break;
      }
      default:
        std::stringstream error_msg{};
        error_msg << "I don't know how to handle the Formulation '"
                  << this->get_formulation() << "'.";
        throw SolverError{error_msg.str()};
        break;
      }
    } else {
      default_grad_val =
          Eigen::MatrixXd::Zero(this->grad_shape[0], this->grad_shape[1]);
    }
    this->grad->get_map() = default_grad_val;

    this->previous_macro_load.setZero(this->grad_shape[0], this->grad_shape[1]);

    // make sure all materials are initialised and every pixel is covered
    this->cell_data->check_material_coverage();

    if (this->is_mechanics()) {
      this->create_mechanics_projection();
    } else {
      this->create_gradient_projection();
    }
    this->projection->initialise();
    this->is_initialised = true;
    // here at the end, because set_matrix checks whether this is initialised
    std::weak_ptr<MatrixAdaptable> w_ptr{this->shared_from_this()};
    this->krylov_solver->set_matrix(w_ptr);
    this->krylov_solver->initialise();
  }

  /* ---------------------------------------------------------------------- */
  OptimizeResult
  SolverNewtonCG::solve_load_increment(const LoadStep & load_step) {
    // check whether this solver's cell has been initialised already
    if (not this->is_initialised) {
      this->initialise_cell();
    }

    auto && comm{this->cell_data->get_communicator()};

    // obtain this iteration's load and check its shape
    const auto & macro_load{load_step.at(this->domain)};
    if (macro_load.rows() != this->grad_shape[0] or
        macro_load.cols() != this->grad_shape[1]) {
      std::stringstream error_message{};
      error_message << "Expected a macroscopic load step of shape "
                    << this->grad_shape
                    << ", but you've supplied a matrix of shape ("
                    << macro_load.rows() << ", " << macro_load.cols() << ")."
                    << std::endl;
      throw SolverError{error_message.str()};
    }

    ++this->get_counter();
    if (load_step.size() > 1 and this->verbosity > Verbosity::Silent and
        comm.rank() == 0) {
      std::cout << "at Load step " << std::setw(this->default_count_width)
                << this->get_counter() << std::endl;
    }

    // updating cell grad with the difference of the current and previous
    // grad input.
    this->grad->get_map() += macro_load - this->previous_macro_load;

    // define the number of newton iteration (initially equal to 0)
    std::string message{"Has not converged"};
    Real incr_norm{2 * newton_tol}, grad_norm{1};
    Real stress_norm{2 * equil_tol};
    bool has_converged{false};
    bool last_step_was_nonlinear{true};
    bool newton_tol_test{false};
    bool equil_tol_test{false};

    //! Checks two loop breaking criteria (newton tolerance and equilibrium
    //! tolerance)
    auto && early_convergence_test{[&incr_norm, &grad_norm, this, &stress_norm,
                                    &message, &has_converged, &newton_tol_test,
                                    &equil_tol_test] {
      newton_tol_test = (incr_norm / grad_norm) <= this->newton_tol;
      equil_tol_test = stress_norm < this->equil_tol;

      if (newton_tol_test) {
        message = "Residual  tolerance reached";
      } else if (equil_tol_test) {
        message = "Reached stress divergence tolerance";
      }
      has_converged = newton_tol_test or equil_tol_test;
      return has_converged;
    }};

    const std::string strain_symb{[this]() -> std::string {
      if (this->is_mechanics()) {
        if (this->get_formulation() == Formulation::finite_strain) {
          return "F";
        } else {
          return "ε";
        }
      } else {
        return "Grad";
      }
    }()};
    //! Checks all convergence criteria, including detection of linear
    //! problems
    auto && full_convergence_test{[&early_convergence_test, this,
                                   &has_converged, &message,
                                   &last_step_was_nonlinear]() {
      // the last step was nonlinear if either this is a finite strain
      // mechanics problem or the material evaluation was non-linear
      bool is_finite_strain_mechanics{this->is_mechanics() and
                                      this->get_formulation() ==
                                          Formulation::finite_strain};
      last_step_was_nonlinear = is_finite_strain_mechanics or
                                this->cell_data->was_last_eval_non_linear();
      if (not last_step_was_nonlinear) {
        message = "Linear problem, no more iteration necessary";
      }
      has_converged = early_convergence_test() or not last_step_was_nonlinear;
      return has_converged;
    }};

    // TODO(junge): Handle eigenstrain here, see issue #146

    auto res_tup{this->evaluate_stress_tangent()};
    auto & flux{std::get<0>(res_tup)};

    *this->rhs = -flux.get_field();
    this->projection->apply_projection(this->rhs->get_field());

    stress_norm =
        std::sqrt(comm.sum(this->rhs->get_field().eigen_vec().squaredNorm()));

    if (early_convergence_test()) {
      has_converged = true;
    }

    Uint newt_iter{0};
    for (; newt_iter < this->max_iter and not has_converged; ++newt_iter) {
      // calling solver for solving the current (iteratively approximated)
      // linear equilibrium problem
      try {
        this->grad_incr->get_field() =
            this->krylov_solver->solve(this->rhs->get_field().eigen_vec());
      } catch (ConvergenceError & error) {
        std::stringstream err{};
        err << "Failure at load step " << this->get_counter()
            << ". In Newton-Raphson step " << newt_iter << ":" << std::endl
            << error.what() << std::endl
            << "The applied boundary condition is Δ" << strain_symb << " ="
            << std::endl
            << macro_load << std::endl
            << "and the load increment is Δ" << strain_symb << " =" << std::endl
            << macro_load - this->previous_macro_load << std::endl;
        throw ConvergenceError(err.str());
      }

      // updating cell strain with the periodic (non-constant) solution
      // resulted from imposing the new macro_strain
      std::cout << "<δgrad> =" << std::endl
                << this->grad_incr->get_map().mean() << std::endl;
      this->grad->get_field() += this->grad_incr->get_field();

      // updating the incremental differences for checking the termination
      // criteria
      incr_norm = std::sqrt(
          comm.sum(this->grad_incr->get_field().eigen_vec().squaredNorm()));
      grad_norm =
          std::sqrt(comm.sum(grad->get_field().eigen_vec().squaredNorm()));

      if ((this->verbosity >= Verbosity::Detailed) and (comm.rank() == 0)) {
        std::cout << "at Newton step " << std::setw(this->default_count_width)
                  << newt_iter << ", |δ" << strain_symb << "|/|Δ" << strain_symb
                  << "|" << std::setw(17) << incr_norm / grad_norm
                  << ", tol = " << newton_tol << std::endl;

        if (this->verbosity > Verbosity::Detailed) {
          std::cout << "<Grad> =" << std::endl
                    << this->grad->get_map().mean() << std::endl;
        }
      }

      // TODO(junge): Handle eigenstrain here, see issue #146

      auto res_tup{this->evaluate_stress_tangent()};
      auto & flux{std::get<0>(res_tup)};

      *this->rhs = -flux.get_field();
      this->projection->apply_projection(this->rhs->get_field());

      stress_norm =
          std::sqrt(comm.sum(this->rhs->get_field().eigen_vec().squaredNorm()));

      full_convergence_test();
    }
    // throwing meaningful error message if the number of iterations for
    // newton_cg is exploited
    if (newt_iter == this->krylov_solver->get_maxiter()) {
      std::stringstream err{};
      err << "Failure at load step " << this->get_counter()
          << ". Newton-Raphson failed to converge. "
          << "The applied boundary condition is Δ" << strain_symb << " ="
          << std::endl
          << macro_load << std::endl
          << "and the load increment is Δ" << strain_symb << " =" << std::endl
          << macro_load - this->previous_macro_load << std::endl;
      throw ConvergenceError(err.str());
    }

    // update previous macroscopic strain
    this->previous_macro_load = macro_load;

    // store results

    OptimizeResult ret_val{this->grad->get_field().eigen_sub_pt(),
                           this->flux->get_field().eigen_sub_pt(),
                           full_convergence_test(),
                           Int(full_convergence_test()),
                           message,
                           newt_iter,
                           this->krylov_solver->get_counter(),
                           this->get_formulation()};

    // store history variables for next load increment
    this->cell_data->save_history_variables();

    return ret_val;
  }

  /* ---------------------------------------------------------------------- */
  Index_t SolverNewtonCG::get_nb_dof() const {
    if (not this->is_initialised) {
      throw SolverError{"Can't determine the number of degrees of freedom "
                        "until I have been "
                        "initialised!"};
    }
    return this->cell_data->get_pixels().size() *
           this->cell_data->get_nb_quad_pts() * this->grad_shape[0] *
           this->grad_shape[1];
  }

  /* ---------------------------------------------------------------------- */
  void SolverNewtonCG::action_increment(EigenCVec_t delta_grad,
                                        const Real & alpha,
                                        EigenVec_t del_flux) {
    auto && grad_size{this->grad_shape[0] * this->grad_shape[1]};
    auto delta_grad_ptr{muGrid::WrappedField<Real>::make_const(
        "delta Grad", this->cell_data->get_fields(), grad_size, delta_grad,
        QuadPtTag)};

    muGrid::WrappedField<Real> del_flux_field{"delta_flux",
                                              this->cell_data->get_fields(),
                                              grad_size, del_flux, QuadPtTag};

    switch (this->cell_data->get_material_dim()) {
    case twoD: {
      this->template action_increment_worker<twoD>(
          *delta_grad_ptr, this->tangent->get_field(), alpha, del_flux_field);
      break;
    }
    case threeD: {
      this->template action_increment_worker<threeD>(
          *delta_grad_ptr, this->tangent->get_field(), alpha, del_flux_field);
      break;
    }
    default:
      std::stringstream err{};
      err << "unknown dimension " << this->cell_data->get_material_dim()
          << std::endl;
      throw SolverError(err.str());
      break;
    }
    this->projection->apply_projection(del_flux_field);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM>
  void SolverNewtonCG::action_increment_worker(
      const muGrid::TypedFieldBase<Real> & delta_grad,
      const muGrid::TypedFieldBase<Real> & tangent, const Real & alpha,
      muGrid::TypedFieldBase<Real> & delta_flux) {
    muGrid::T2FieldMap<Real, muGrid::Mapping::Const, DimM, IterUnit::SubPt>
        grad_map{delta_grad};
    muGrid::T4FieldMap<Real, muGrid::Mapping::Const, DimM, IterUnit::SubPt>
        tangent_map{tangent};
    muGrid::T2FieldMap<Real, muGrid::Mapping::Mut, DimM, IterUnit::SubPt>
        flux_map{delta_flux};
    for (auto && tup : akantu::zip(grad_map, tangent_map, flux_map)) {
      auto & df = std::get<0>(tup);
      auto & k = std::get<1>(tup);
      auto & dp = std::get<2>(tup);
      dp += alpha * Matrices::tensmult(k, df);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  void SolverNewtonCG::create_mechanics_projection_worker() {
    switch (this->get_formulation()) {
    case Formulation::finite_strain: {
      this->projection = std::make_shared<ProjectionFiniteStrainFast<Dim>>(
          this->cell_data->get_FFT_engine(),
          this->cell_data->get_domain_lengths());
      break;
    }
    case Formulation::small_strain: {
      this->projection = std::make_shared<ProjectionSmallStrain<Dim>>(
          this->cell_data->get_FFT_engine(),
          this->cell_data->get_domain_lengths());
      break;
    }
    default:
      std::stringstream error_msg{};
      error_msg << "I don't know how to handle the Formulation '"
                << this->get_formulation() << "'.";
      throw SolverError{error_msg.str()};
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  void SolverNewtonCG::create_mechanics_projection() {
    switch (this->cell_data->get_spatial_dim()) {
    case twoD: {
      this->template create_mechanics_projection_worker<twoD>();
      break;
    }
    case threeD: {
      this->template create_mechanics_projection_worker<threeD>();
      break;
    }
    default:
      std::stringstream error_message{};
      error_message << "Only 2- and 3-dimensional projections are currently "
                       "supported, you chose "
                    << this->cell_data->get_spatial_dim() << '.';
      throw SolverError{error_message.str()};
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  void SolverNewtonCG::create_gradient_projection() {
    switch (this->cell_data->get_spatial_dim()) {
    case twoD: {
      // fall_through
    }
    case threeD: {
      // fall_through
    }
    default:
      std::stringstream error_message{};
      error_message << "generic gradient projection is not implemented yet";
      throw SolverError{error_message.str()};
      break;
    }
  }
}  // namespace muSpectre
