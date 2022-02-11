/**
 * @file   solver_fem_trust_region_newton_cg.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   31 Aug 2020
 *
 * @brief  Implementation for Trust Region Newton-CG FEM solver
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
#include "materials/material_mechanics_base.hh"

#include <libmugrid/grid_common.hh>
#include <libmugrid/iterators.hh>

#include <iomanip>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  SolverFEMTrustRegionNewtonCG::SolverFEMTrustRegionNewtonCG(
      std::shared_ptr<Discretisation> discretisation,
      std::shared_ptr<KrylovSolverTrustRegionCG> krylov_solver,
      const muGrid::Verbosity & verbosity, const Real & newton_tol,
      const Real & equil_tol, const Uint & max_iter,
      const Real & max_trust_radius, const Real & eta)
      : Parent{discretisation->get_cell(), verbosity,
               SolverType::FiniteElements},
        krylov_solver{krylov_solver}, discretisation{discretisation},
        K{this->discretisation->get_stiffness_operator(
            this->get_displacement_rank())},
        newton_tol{newton_tol}, equil_tol{equil_tol}, max_iter{max_iter},
        max_trust_radius{max_trust_radius}, eta{eta} {}

  /* ---------------------------------------------------------------------- */
  void SolverFEMTrustRegionNewtonCG::initialise_cell() {
    if (this->is_initialised) {
      return;
    }

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
          mech->set_solver_type(SolverType::FiniteElements);
        }
      }
    }

    this->displacement_shape = {muGrid::ipow(this->cell_data->get_spatial_dim(),
                                             this->get_displacement_rank()),
                                1};
    this->grad_shape =
        gradient_shape(this->domain.rank(), this->cell_data->get_spatial_dim(),
                       this->is_mechanics(), this->get_formulation());

    //! store solver fields in cell
    auto & field_collection{cell_data->get_fields()};

    // Corresponds to symbol δũ
    this->disp_fluctuation_incr = std::make_shared<MappedField_t>(
        "incru", this->displacement_shape[0], this->displacement_shape[1],
        IterUnit::SubPt, field_collection, NodalPtTag);

    // Corresponds to symbol ũ
    this->disp_fluctuation = std::make_shared<MappedField_t>(
        "u", this->displacement_shape[0], this->displacement_shape[1],
        IterUnit::SubPt, field_collection, NodalPtTag);

    // Corresponds to symbol F or ε
    this->grad = std::make_shared<MappedField_t>(
        "grad", this->grad_shape[0], this->grad_shape[1], IterUnit::SubPt,
        field_collection, QuadPtTag);

    this->eval_grad = this->grad;

    this->eval_grads[this->domain] = this->eval_grad;
    this->grads[this->domain] = this->grad;

    // Corresponds to symbol P or σ
    this->flux = std::make_shared<MappedField_t>(
        "flux", this->grad_shape[0], this->grad_shape[1], IterUnit::SubPt,
        field_collection, QuadPtTag);
    this->fluxes[this->domain] = this->flux;

    // Corresponds to symbol f
    this->force = std::make_shared<MappedField_t>(
        "f", this->displacement_shape[0], this->displacement_shape[1],
        IterUnit::SubPt, field_collection, NodalPtTag);

    // Corresponds to symbol K or C
    Index_t tangent_size{this->grad_shape[0] * this->grad_shape[1]};
    this->tangent = std::make_shared<MappedField_t>(
        "tangent", tangent_size, tangent_size, IterUnit::SubPt,
        field_collection, QuadPtTag);
    this->tangents[this->domain] = this->tangent;

    // field to store the rhs for cg calculations
    this->rhs = std::make_shared<MappedField_t>(
        "rhs", this->displacement_shape[0], this->displacement_shape[1],
        IterUnit::SubPt, field_collection, NodalPtTag);

    this->previous_macro_load.setZero(this->grad_shape[0], this->grad_shape[1]);

    // make sure all materials are initialised and every pixel is covered
    this->cell_data->check_material_coverage();

    this->is_initialised = true;

    // here at the end, because set_matrix checks whether this is initialised
    std::weak_ptr<MatrixAdaptable> w_ptr{this->shared_from_this()};
    this->krylov_solver->set_matrix(w_ptr);
  }

  /* ---------------------------------------------------------------------- */
  OptimizeResult SolverFEMTrustRegionNewtonCG::solve_load_increment(
      const LoadStep & load_step, EigenStrainFunc_ref eigen_strain_func,
      CellExtractFieldFunc_ref cell_extract_func) {
    if (eigen_strain_func == muGrid::nullopt and
        this->has_eigen_strain_storage()) {
      std::stringstream err{};
      err << "eval_grad is different from the grad field of the cell"
          << ". Therefore, it does not get updated unless an eigen strain "
          << "function is passed to solve_load_increment"
          << "This is probably because you have already called this "
          << "previously with an eigenstrain function." << std::endl;
      throw SolverError(err.str());
    }
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

    ++this->counter_load_step;
    if (this->verbosity > Verbosity::Silent and comm.rank() == 0) {
      std::cout << "at Load step " << std::setw(this->default_count_width)
                << this->get_counter_load_step() << std::endl;
    }

    // define the number of newton iteration (initially equal to 0)
    std::string message{"Has not converged"};
    Real incr_norm{2 * newton_tol}, displacement_norm{1};
    Real force_norm{2 * equil_tol};
    bool has_converged{false};
    bool last_step_was_nonlinear{true};
    bool newton_tol_test{false};
    bool equil_tol_test{false};

    //! Checks two loop breaking criteria (newton tolerance and equilibrium
    //! tolerance)
    auto && early_convergence_test{[&incr_norm, &displacement_norm, this,
                                    &force_norm, &message, &has_converged,
                                    &newton_tol_test, &equil_tol_test] {
      newton_tol_test = (incr_norm / displacement_norm) <= this->newton_tol;
      equil_tol_test = force_norm < this->equil_tol;

      if (newton_tol_test) {
        message = "Residual tolerance reached";
      } else if (equil_tol_test) {
        message = "Reached force balance tolerance";
      }
      has_converged = newton_tol_test or equil_tol_test;
      return has_converged;
    }};

    auto && linear_convrgence_test{
        [this, &has_converged, &message,
         &last_step_was_nonlinear]() {  // the last step was nonlinear if either
                                        // this is a finite strain
          // mechanics problem or the material evaluation was non-linear
          bool is_finite_strain_mechanics{this->is_mechanics() and
                                          this->get_formulation() ==
                                              Formulation::finite_strain};
          last_step_was_nonlinear = is_finite_strain_mechanics or
                                    this->cell_data->was_last_eval_non_linear();
          if (not last_step_was_nonlinear) {
            message = "Linear problem, no more iteration necessary";
          }
          has_converged = not last_step_was_nonlinear;
          return has_converged;
        }};

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

    // create and initialise the radius of the trust region with
    // max_trust_region_radius
    Real trust_region_radius{this->max_trust_radius};

    if ((this->verbosity >= Verbosity::Detailed) and (comm.rank() == 0)) {
      std::cout << std::setw(this->default_count_width) << "STEP"
                << std::setw(15) << "ACTION" << std::setw(15) << "OLD TR. REG."
                << std::setw(15) << "NEW TR. REG." << std::setw(15)
                << "Δm model" << std::setw(15) << "ΔE estim." << std::setw(15)
                << " estim. ρ" << std::setw(15) << "|GP|"
                << "(equi tol)" << std::setw(15)
                << ("|δ" + this->strain_symb() + "|/|Δ" + this->strain_symb() +
                    "|")
                << "(newt tol)" << std::endl;
    }

    // The vector created for holding a copy of the force vector which will be
    // used in objective function reduction estimation
    Vector_t force_copy_vec;

    auto & grad_operator{*this->K.get_gradient_operator()};

    this->grad->get_map() = macro_load;
    grad_operator.apply_gradient_increment(this->disp_fluctuation->get_field(),
                                           1., this->grad->get_field());
    if (not(eigen_strain_func == muGrid::nullopt)) {
      this->initialise_eigen_strain_storage();
      this->eval_grad->get_field() = this->grad->get_field();
      (eigen_strain_func.value())(this->eval_grad->get_field());
    }

    // this fills the tangent and flux fields of this solver, using the
    // eval_grad field as input
    this->clear_last_step_nonlinear();
    auto res_tup{this->evaluate_stress_tangent()};
    auto & flux{std::get<0>(res_tup)};

    this->K.apply_divergence(flux.get_field(), this->force->get_field());

    // keep a copy of the flux in a vector shape (later needed for calculating
    // the estimation of energy function reduction)
    force_copy_vec = this->force->get_field().eigen_vec();

    force_norm =
        std::sqrt(this->squared_norm(this->force->get_field().eigen_vec()));
    *this->rhs = -this->force->get_field();

    if (early_convergence_test()) {
      has_converged = true;
    }

    Uint newt_iter{0};
    for (; newt_iter < this->max_iter and not has_converged; ++newt_iter) {
      // setting the trust region radius of the solver of the sub-problem
      try {
        krylov_solver->set_trust_region(trust_region_radius);
      } catch (SolverError & error) {
        std::stringstream err{};
        err << "The Krylov solver needs to be a trust-region sub-problem "
            << "solver (should have a trust region member to be set). "
            << "For instance, use KrylovTrustRegionCG solver" << std::endl;
        throw SolverError(err.str());
      }
      // calling solver for solving the current (iteratively approximated)
      // linear equilibrium problem
      try {
        this->disp_fluctuation_incr->get_field() =
            this->krylov_solver->solve(this->rhs->get_field().eigen_vec());
      } catch (ConvergenceError & error) {
        std::stringstream err{};
        err << "Failure at load step " << this->get_counter_load_step()
            << ". In Newton-Raphson step " << newt_iter << ":" << std::endl
            << error.what() << std::endl
            << "The applied boundary condition is Δ" << this->strain_symb()
            << " =" << std::endl
            << macro_load << std::endl
            << "and the load increment is Δ" << this->strain_symb() << " ="
            << std::endl
            << macro_load - this->previous_macro_load << std::endl;
        throw ConvergenceError(err.str());
      }

      // δuₖ
      Vector_t displacement_incr_vec{
          this->disp_fluctuation_incr->get_field().eigen_vec()};

      // Fₖᵀ:δuₖ (will be used in calculating Δ̱E)
      Real force_copy_displacement_incr{
          this->dot(force_copy_vec, displacement_incr_vec)};

      // δuₖᵀ: Kₖ : δuₖ (will be used in model reduction calculation Δ̱m)
      Real disp_incr_K_disp_incr{this->dot(
          displacement_incr_vec,
          this->krylov_solver->get_matrix_ptr().lock()->get_adaptor() *
              displacement_incr_vec)};

      // Δmₖ = mₖ(δuₖ) - mₖ(0) = 0.5 * δuₖᵀ : Kₖ : δuₖ + Fₖᵀ : δuₖ
      Real del_m{0.5 * disp_incr_K_disp_incr + force_copy_displacement_incr};

      // due to the trial newton step (it might be rejected)
      // increment cell displacement and eval grad field according to trial
      // displacement (result of subporblem solution)
      grad_operator.apply_gradient_increment(
          this->disp_fluctuation_incr->get_field(), 1.,
          this->eval_grad->get_field());

      // Fᵗʳⁱᵃˡₖ₊₁:
      auto && flux_trial{this->evaluate_stress()};

      this->K.apply_divergence(flux_trial.get_field(),
                               this->force->get_field());

      Vector_t force_trial_vec{this->force->get_field().eigen_vec()};

      // restore cell displacement back
      grad_operator.apply_gradient_increment(
          this->disp_fluctuation_incr->get_field(), -1.,
          this->eval_grad->get_field());

      // TODO(afalsafi): I have to check whether applying the divergence on top
      // of the trial strain does not mess up with the force field
      this->K.apply_divergence(flux.get_field(), this->force->get_field());

      // Fᵀₖ₊₁ : δuₖ
      Real force_trial_displacement_incr{
          this->dot(force_trial_vec, displacement_incr_vec)};

      // ΔE = 0.5 * (Fᵀₖ : δuₖ + Fᵀₖ₊₁ : δuₖ)
      Real del_E_bar{
          0.5 * (force_copy_displacement_incr + force_trial_displacement_incr)};

      // ρₖ estimated ratio  of the reduction of the energy of the
      // system (Actual reduction) vs. its quadratic model (sub-problem
      // estimation) in the trust region
      Real rho_bar{del_E_bar / del_m};

      // snapshot of trust region size history (printing purposes)
      Real pre_tr{trust_region_radius};

      // string descriptor for this step (for printing to screen)
      std::string action_str{""};
      // std::cout << "Normalized_diff_of_step_size_and_the_Tr_radius:"
      //           << ((incr_norm - trust_region_radius) / trust_region_radius)
      //           << "\n";

      Real eps{1.0e-2};
      if (rho_bar < 0.25) {
        // reduce trust region if the model estimation does not match the system
        // energy reduction or the model does not predict reduction at all
        trust_region_radius *= (1.0 / 4.0);
        action_str = "decrease tr";
      } else if (rho_bar > 0.75 and std::abs((incr_norm - trust_region_radius) /
                                             trust_region_radius) < eps) {
        // expand the trust region if estimated energy loss matches with
        // the model energy reduction
        trust_region_radius =
            std::min(2.0 * trust_region_radius, this->max_trust_radius);
        action_str = "increase tr";
      } else {
        action_str = "keep tr";
      }

      if ((this->verbosity >= Verbosity::Detailed) and (comm.rank() == 0)) {
        std::cout << std::setw(3) << (newt_iter) << std::setw(15) << action_str
                  << std::setw(15) << pre_tr << std::setw(15)
                  << trust_region_radius << std::setw(15) << del_m
                  << std::setw(15) << del_E_bar << std::setw(15) << rho_bar
                  << std::setw(15) << force_norm << "(" << equil_tol << ")"
                  << std::setw(15) << incr_norm / displacement_norm << "("
                  << newton_tol << ")" << std::setw(15);
      }

      if (rho_bar < eta) {
        if ((this->verbosity >= Verbosity::Detailed) and (comm.rank() == 0)) {
          std::cout << "  (Rejected)" << std::endl;
        }
        linear_convrgence_test();
        continue;
      } else {
        if ((this->verbosity >= Verbosity::Detailed) and (comm.rank() == 0)) {
          std::cout << "  (Accepted)" << std::endl;
        }

        // updating cell strain with the periodic (non-constant) solution
        // resulted from imposing the new macro_strain
        this->disp_fluctuation->get_field() +=
            this->disp_fluctuation_incr->get_field();

        // updating the incremental differences for checking the termination
        // criteria
        // ||δuₖ||
        incr_norm = std::sqrt(this->squared_norm(
            this->disp_fluctuation_incr->get_field().eigen_vec()));
        // ||uₖ||
        displacement_norm = std::sqrt(
            this->squared_norm(disp_fluctuation->get_field().eigen_vec()));

        if ((this->verbosity >= Verbosity::Detailed) and (comm.rank() == 0)) {
          std::cout << "at Newton step " << std::setw(this->default_count_width)
                    << newt_iter << ", |δu|/|Δu|" << std::setw(17)
                    << incr_norm / displacement_norm << ", tol = " << newton_tol
                    << std::endl;

          if (this->verbosity > Verbosity::Detailed) {
            std::cout << "<Grad> =" << std::endl
                      << this->grad->get_map().mean() << std::endl;
          }
        }

        this->grad->get_map() = macro_load;
        grad_operator.apply_gradient_increment(
            this->disp_fluctuation->get_field(), 1., this->grad->get_field());

        if (not(eigen_strain_func == muGrid::nullopt)) {
          this->eval_grad->get_field() = this->grad->get_field();
          (eigen_strain_func.value())(this->eval_grad->get_field());
        }

        // Calculating the Flux and Tangent using eval_grad and updating the
        // force Field with according to the Flux Field
        this->clear_last_step_nonlinear();
        auto res_tup{this->evaluate_stress_tangent()};
        auto & flux_loop{std::get<0>(res_tup)};
        this->K.apply_divergence(flux_loop.get_field(),
                                 this->force->get_field());

        // Keeping a copy of the Force field (be used in ρ calculation)
        force_copy_vec = this->force->get_field().eigen_vec();
        force_norm =
            std::sqrt(this->squared_norm(this->force->get_field().eigen_vec()));

        // updating RHS for feeding to CG
        *this->rhs = -this->force->get_field();

        if ((this->verbosity >= Verbosity::Detailed) and (comm.rank() == 0)) {
          std::cout << std::setw(3) << (newt_iter) << std::setw(15)
                    << action_str << std::setw(15) << pre_tr << std::setw(15)
                    << trust_region_radius << std::setw(15) << del_m
                    << std::setw(15) << del_E_bar << std::setw(15) << rho_bar
                    << std::setw(15) << force_norm << "(" << equil_tol << ")"
                    << std::setw(15) << incr_norm / displacement_norm << "("
                    << newton_tol << ")" << std::setw(15) << "\n";
        }
        // if (not this->krylov_solver->get_is_on_bound()) {
        // bool full_has_converged{full_convergence_test()};
        full_convergence_test();
        // std::cout << "FULL_HAS_Converged: " << full_has_converged << "\n";
        // } else {
        //   std::cout << "ON_TRUST_REGION_BOUNDARY: 1"
        //             << "\n";
        // }
      }
    }

    // incrementing the number of Newton steps so far by the number of newton
    // steps of current load step
    this->counter_iteration += newt_iter;

    // throwing meaningful error message if the number of iterations for
    // newton_cg is exploited
    if (newt_iter == krylov_solver->get_maxiter()) {
      std::stringstream err{};
      err << "Failure at load step " << this->get_counter_load_step()
          << ". Newton-Raphson failed to converge. "
          << "The applied boundary condition is Δ" << this->strain_symb()
          << " =" << std::endl
          << macro_load << std::endl
          << "and the load increment is Δ" << this->strain_symb() << " ="
          << std::endl
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
                           incr_norm / displacement_norm,
                           force_norm,
                           this->get_formulation()};

    // store history variables for next load increment
    this->cell_data->save_history_variables();
    if (not(cell_extract_func == muGrid::nullopt)) {
      (cell_extract_func.value())(this->get_cell_data());
    }
    return ret_val;
  }

  /* ---------------------------------------------------------------------- */
  Index_t SolverFEMTrustRegionNewtonCG::get_nb_dof() const {
    if (not this->is_initialised) {
      throw SolverError{"Can't determine the number of degrees of freedom "
                        "until I have been "
                        "initialised!"};
    }
    return this->cell_data->get_pixels().size() *
           this->cell_data->get_nb_nodal_pts() *
           muGrid::ipow(this->cell_data->get_spatial_dim(),
                        this->get_displacement_rank());
  }

  /* ---------------------------------------------------------------------- */
  void SolverFEMTrustRegionNewtonCG::action_increment(EigenCVecRef delta_u,
                                                      const Real & alpha,
                                                      EigenVecRef delta_f) {
    this->K.apply_increment(this->tangent->get_field(), delta_u, alpha,
                            delta_f);
  }

  /* ---------------------------------------------------------------------- */
  Index_t SolverFEMTrustRegionNewtonCG::get_displacement_rank() const {
    return this->domain.rank() - 1;
  }

  /* ---------------------------------------------------------------------- */
  auto SolverFEMTrustRegionNewtonCG::get_disp_fluctuation() const
      -> const MappedField_t & {
    return *this->disp_fluctuation;
  }

  /* ---------------------------------------------------------------------- */
  void SolverFEMTrustRegionNewtonCG::initialise_eigen_strain_storage() {
    if (not this->has_eigen_strain_storage()) {
      this->eval_grad = std::make_shared<MappedField_t>(
          "eval_grad", this->grad_shape[0], this->grad_shape[1],
          IterUnit::SubPt, this->cell_data->get_fields(), QuadPtTag);
      this->eval_grads[this->domain] = this->eval_grad;
    }
  }

  /* ---------------------------------------------------------------------- */
  bool SolverFEMTrustRegionNewtonCG::has_eigen_strain_storage() const {
    return this->eval_grad != this->grad;
  }

  /* ---------------------------------------------------------------------- */
  auto SolverFEMTrustRegionNewtonCG::get_rhs() -> MappedField_t & {
    return *this->rhs;
  }

  /* ---------------------------------------------------------------------- */
  auto SolverFEMTrustRegionNewtonCG::get_incr() -> MappedField_t & {
    return *this->disp_fluctuation_incr;
  }

  /* ---------------------------------------------------------------------- */
  auto SolverFEMTrustRegionNewtonCG::get_krylov_solver() -> KrylovSolverBase & {
    return *this->krylov_solver;
  }
}  // namespace muSpectre
