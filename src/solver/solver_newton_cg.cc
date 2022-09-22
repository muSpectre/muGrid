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

#include <iomanip>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  SolverNewtonCG::SolverNewtonCG(
      std::shared_ptr<CellData> cell_data,
      std::shared_ptr<KrylovSolverBase> krylov_solver,
      const muGrid::Verbosity & verbosity, const Real & newton_tol,
      const Real & equil_tol, const Uint & max_iter,
      const Gradient_t & gradient, const Weights_t & weights,
      const MeanControl & mean_control)
      : Parent{cell_data, verbosity, newton_tol, equil_tol,
               max_iter, gradient, weights, mean_control},
        krylov_solver{krylov_solver} {}

  /* ---------------------------------------------------------------------- */
  SolverNewtonCG::SolverNewtonCG(
      std::shared_ptr<CellData> cell_data,
      std::shared_ptr<KrylovSolverBase> krylov_solver,
      const muGrid::Verbosity & verbosity, const Real & newton_tol,
      const Real & equil_tol, const Uint & max_iter,
      const MeanControl & mean_control)
      : Parent{cell_data, verbosity, newton_tol,
               equil_tol, max_iter,  mean_control},
        krylov_solver{krylov_solver} {}

  /* ---------------------------------------------------------------------- */
  void SolverNewtonCG::initialise_cell() {
    if (this->is_initialised) {
      return;
    }
    this->initialise_cell_worker();
    this->projection->initialise();
    this->is_initialised = true;

    // here at the end, because set_matrix checks whether this is initialised
    std::weak_ptr<MatrixAdaptable> w_ptr{this->shared_from_this()};
    this->krylov_solver->set_matrix(w_ptr);
    this->krylov_solver->initialise();
  }

  /* ---------------------------------------------------------------------- */
  OptimizeResult SolverNewtonCG::solve_load_increment(
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

    if (this->verbosity > Verbosity::Silent and comm.rank() == 0) {
      for (auto & elem : load_step) {
        std::cout << "the load_step for the physics domain: "
                  << elem.first.get_name() << " is:\n"
                  << elem.second << "\n";
      }
    }

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
    if (load_step.size() > 1 and this->verbosity > Verbosity::Silent and
        comm.rank() == 0) {
      std::cout << "at Load step " << std::setw(this->default_count_width)
                << this->get_counter_load_step() << std::endl;
    }

    // updating cell grad with the difference of the current and previous
    // grad input if the mean control is on the strain
    if (this->mean_control == MeanControl::StrainControl) {
      this->grad->get_map() += macro_load - this->previous_macro_load;
    }

    // define the number of newton iteration (initially equal to 0)
    std::string message{"Has not converged"};
    Real incr_second_norm{2 * newton_tol}, incr_inf_norm{2 * newton_tol},
        incr_norm_mean_control{2 * newton_tol}, grad_norm{1};
    // Real incr_max{2 * newton_tol};
    Real rhs_norm{2 * equil_tol};
    bool has_converged{false};
    bool last_step_was_nonlinear{true};
    bool newton_tol_test{false};
    bool equil_tol_test{false};

    //! Checks two loop breaking criteria (newton tolerance and equilibrium
    //! tolerance)
    auto && early_convergence_test{[&incr_norm_mean_control, &grad_norm, this,
                                    &rhs_norm, &message, &has_converged,
                                    &newton_tol_test, &equil_tol_test] {
      newton_tol_test =
          (incr_norm_mean_control / grad_norm) <= this->newton_tol;
      equil_tol_test = rhs_norm < this->equil_tol;

      if (newton_tol_test) {
        message = "Residual tolerance reached";
      } else if (equil_tol_test) {
        message = "Reached stress divergence tolerance";
      }
      has_converged = newton_tol_test or equil_tol_test;
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

    if (not(eigen_strain_func == muGrid::nullopt)) {
      this->initialise_eigen_strain_storage();
      this->eval_grad->get_field() = this->grad->get_field();
      (eigen_strain_func.value())(this->eval_grad->get_field());
    }

    this->clear_last_step_nonlinear();
    auto res_tup{this->evaluate_stress_tangent()};
    auto & flux{std::get<0>(res_tup)};

    *this->rhs = -flux.get_field();

    // Here we need to subtract the mean stress value if the control
    // is on the mean value of stress (we do not to subtract the macroscopic
    // load the previous step since the RHS is overwritten in each step)
    // (for more clarification refer to equation 15 in "An algorithm for
    // stress and mixed control in
    // Galerkin-based FFT homogenization", 2019,  By: S. Lucarini , and J.
    // Segurado, doi: 10.1002/nme.6069)
    if (this->mean_control == MeanControl::StressControl) {
      this->rhs->get_map() += macro_load;
    }

    this->projection->apply_projection(this->rhs->get_field());

    rhs_norm =
        std::sqrt(this->squared_norm(this->rhs->get_field().eigen_vec()));

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

      // updating cell strain with the periodic (non-constant) solution
      // resulted from imposing the new macro_strain
      if ((this->verbosity > Verbosity::Silent) and (comm.rank() == 0)) {
        std::cout << "<δgrad> =" << std::endl
                  << this->grad_incr->get_map().mean() << std::endl;
      }

      this->grad->get_field() += this->grad_incr->get_field();

      grad_norm = std::sqrt(this->squared_norm(grad->get_field().eigen_vec()));

      incr_second_norm = std::sqrt(
          this->squared_norm(this->grad_incr->get_field().eigen_vec()));
      switch (this->mean_control) {
      case MeanControl::StrainControl: {
        incr_norm_mean_control = incr_second_norm;
        break;
      }
      case MeanControl::StressControl: {
        // criteria according to "An algorithm for stress and mixed control in
        // Galerkin-based FFT homogenization" by: S.Lucarini, J. Segurado
        // doi: 10.1002/nme.6069
        incr_inf_norm = this->inf_norm(grad_incr);
        incr_norm_mean_control =
            incr_inf_norm / grad_incr->get_field().get_nb_entries();
        break;
      }
      case MeanControl::MixedControl: {
        muGrid::RuntimeError("Mixed control projection is not implemented yet");
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown value for mean_control value");
        break;
      }

      if ((this->verbosity >= Verbosity::Detailed) and (comm.rank() == 0)) {
        std::cout << "at Newton step " << std::setw(this->default_count_width)
                  << newt_iter << ", |δ" << this->strain_symb() << "|/|Δ"
                  << this->strain_symb() << "|" << std::setw(17)
                  << incr_second_norm / grad_norm << ", tol = " << newton_tol
                  << std::endl;

        if (this->verbosity > Verbosity::Detailed) {
          switch (this->mean_control) {
          case MeanControl::StrainControl: {
            std::cout << "<Grad> =" << std::endl
                      << this->grad->get_map().mean() << std::endl;
            break;
          }
          case MeanControl::StressControl: {
            std::cout << "<Flux> =" << std::endl
                      << this->flux->get_map().mean() + macro_load << std::endl;
            std::cout << "<Grad> =" << std::endl
                      << this->grad->get_map().mean() << std::endl;
            break;
          }
          default:
            throw muGrid::RuntimeError("Unknown value for mean_control value");
            break;
          }
        }
      }

      if (not(eigen_strain_func == muGrid::nullopt)) {
        this->eval_grad->get_field() = this->grad->get_field();
        (eigen_strain_func.value())(this->eval_grad->get_field());
      }

      this->clear_last_step_nonlinear();
      auto res_tup{this->evaluate_stress_tangent()};
      auto & flux{std::get<0>(res_tup)};

      *this->rhs = -flux.get_field();

      // Here we need to subtract the mean stress value if the control
      // is on the mean value of stress (we do not to subtract the macroscopic
      // load the previous step since the RHS is overwritten in each step)
      // (for more clarification refer to equation 15 in "An algorithm for
      // stress and mixed control in
      // Galerkin-based FFT homogenization", 2019,  By: S. Lucarini , and J.
      // Segurado, doi: 10.1002/nme.6069)
      if (this->mean_control == MeanControl::StressControl) {
        this->rhs->get_map() += macro_load;
      }

      this->projection->apply_projection(this->rhs->get_field());

      rhs_norm =
          std::sqrt(this->squared_norm(this->rhs->get_field().eigen_vec()));

      full_convergence_test();
    }

    // incrementing the number of Newton steps so far by the number of newton
    // steps of current load step
    this->counter_iteration += newt_iter;

    // throwing meaningful error message if the number of iterations for
    // newton_cg is exploited
    if (newt_iter == this->krylov_solver->get_maxiter()) {
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
                           incr_second_norm / grad_norm,
                           rhs_norm,
                           this->get_formulation()};

    // store history variables for next load increment
    this->cell_data->save_history_variables();
    if (not(cell_extract_func == muGrid::nullopt)) {
      (cell_extract_func.value())(this->get_cell_data());
    }

    return ret_val;
  }

  /* ---------------------------------------------------------------------- */
  auto SolverNewtonCG::get_krylov_solver() -> KrylovSolverBase & {
    return *this->krylov_solver;
  }

}  // namespace muSpectre
