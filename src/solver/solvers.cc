/**
 * @file   solvers.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Apr 2018
 *
 * @brief  implementation of dynamic newton-cg solver
 *
 * Copyright © 2018 Till Junge
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "solver/solvers.hh"

#include "cell/cell_adaptor.hh"

#include <libmugrid/iterators.hh>
#include <libmugrid/mapped_field.hh>

#include <Eigen/Dense>

#include <iomanip>
#include <iostream>
#include <algorithm>

namespace muSpectre {

  //! produces numpy-compatible full precision text output. great for debugging
  Eigen::IOFormat format(Eigen::FullPrecision, 0, ", ", ",\n", "[", "]", "[",
                         "]");

  /* ---------------------------------------------------------------------- */
  ConvergenceCriterion::ConvergenceCriterion()
      : was_last_step_linear_test{false}, equil_tol_test{false},
        newton_tol_test{false} {}

  /* ---------------------------------------------------------------------- */
  bool & ConvergenceCriterion::get_was_last_step_linear_test() {
    return this->was_last_step_linear_test;
  }

  /* ---------------------------------------------------------------------- */
  bool & ConvergenceCriterion::get_equil_tol_test() {
    return this->equil_tol_test;
  }

  /* ---------------------------------------------------------------------- */
  bool & ConvergenceCriterion::get_newton_tol_test() {
    return this->newton_tol_test;
  }

  /* ---------------------------------------------------------------------- */
  void ConvergenceCriterion::reset() {
    this->was_last_step_linear_test = false;
    this->equil_tol_test = false;
    this->newton_tol_test = false;
  }

  //--------------------------------------------------------------------------//
  std::vector<OptimizeResult>
  newton_cg(Cell & cell, const LoadSteps_t & load_steps,
            KrylovSolverBase & solver, const Real & newton_tol,
            const Real & equil_tol, const Verbosity & verbose,
            const IsStrainInitialised & strain_init,
            EigenStrainOptFunc_ref eigen_strain_func) {
    if (load_steps.size() == 0) {
      throw SolverError("No load steps specified.");
    }

    const auto & comm = cell.get_communicator();

    using Matrix_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    auto shape{cell.get_strain_shape()};

    ConvergenceCriterion convergence_criterion;

    // create a field collection to store workspaces
    auto field_collection{cell.get_fields().get_empty_clone()};
    // Corresponds to symbol δF or δε
    muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>> incrF_field{
        "incrF",         shape[0],         shape[1],
        IterUnit::SubPt, field_collection, QuadPtTag};

    // field to store the rhs for cg calculations
    muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>> rhs_field{
        "rhs",           shape[0],         shape[1],
        IterUnit::SubPt, field_collection, QuadPtTag};

    // Standard strain field without the eigen strain: typically the strain
    // field given to the cell for stress/tangent evaluation, but in case of
    // presence of eigen strain, it is the strain field before adding the eigen
    // strain
    muGrid::TypedFieldBase<Real> & general_strain_field{
        [&cell, &field_collection,
         &eigen_strain_func]() -> muGrid::TypedFieldBase<Real> & {
          if (not(eigen_strain_func == muGrid::nullopt)) {
            // general strain field
            muGrid::RealField & general_strain_field{
                field_collection.register_real_field(
                    "general strain",
                    shape_for_formulation(cell.get_formulation(),
                                          cell.get_material_dim()),
                    QuadPtTag)};
            return general_strain_field;
          } else {
            return cell.get_strain();
          }
        }()};

    solver.initialise();

    size_t count_width{};
    const auto form{cell.get_formulation()};
    std::string strain_symb{};
    if (verbose > Verbosity::Silent && comm.rank() == 0) {
      // setup of algorithm 5.2 in Nocedal, Numerical Optimization (p. 111)
      std::cout << "Newton-" << solver.get_name() << " for ";
      switch (form) {
      case Formulation::small_strain: {
        strain_symb = "ε";
        std::cout << "small";
        break;
      }
      case Formulation::finite_strain: {
        strain_symb = "F";
        std::cout << "finite";
        break;
      }
      default:
        throw SolverError("unknown formulation");
        break;
      }
      std::cout << " strain with" << std::endl
                << "newton_tol = " << newton_tol
                << ", cg_tol = " << solver.get_tol()
                << " maxiter = " << solver.get_maxiter() << " and "
                << strain_symb << " from " << strain_symb << "₁ =" << std::endl
                << load_steps.front() << std::endl
                << " to " << strain_symb << "ₙ =" << std::endl
                << load_steps.back() << std::endl
                << "in increments of Δ" << strain_symb << " =" << std::endl
                << (load_steps.back() - load_steps.front()) / load_steps.size()
                << std::endl;
      count_width = static_cast<size_t>(std::log10(solver.get_maxiter())) + 1;
    }

    Matrix_t default_strain_val{};
    switch (form) {
    case Formulation::finite_strain: {
      if (strain_init == IsStrainInitialised::False) {
        // initilasing cell placement gradient (F) with identity matrix in
        // case of finite strain
        default_strain_val = Matrix_t::Identity(shape[0], shape[1]);
        cell.set_uniform_strain(default_strain_val);

        if (verbose > Verbosity::Silent && comm.rank() == 0) {
          std::cout << "\nThe strain is initialised by default to the identity "
                       "matrix!\n"
                    << std::endl;
        }
      } else if (verbose > Verbosity::Silent && comm.rank() == 0 &&
                 strain_init == IsStrainInitialised::True) {
        std::cout << "\nThe strain was initialised by the user!\n" << std::endl;
      }
      // Checking the consistancy of input load_steps and cell shape
      for (const auto & delF : load_steps) {
        if (not((delF.rows() == shape[0]) and (delF.cols() == shape[1]))) {
          std::stringstream err{};
          err << "Load increments need to be given in " << shape[0] << "×"
              << shape[1] << " matrices, but I got a " << delF.rows() << "×"
              << delF.cols() << " matrix:" << std::endl
              << delF;
          throw SolverError(err.str());
        }
      }
      break;
    }
    case Formulation::small_strain: {
      if (strain_init == IsStrainInitialised::False) {
        // initilasing cell strain (ε) with zero-filled matrix in case of small
        // strain
        default_strain_val = Matrix_t::Zero(shape[0], shape[1]);
        cell.set_uniform_strain(default_strain_val);
        if (verbose > Verbosity::Silent && comm.rank() == 0) {
          std::cout << "\nThe strain is initialised by default to the zero "
                       "matrix!\n"
                    << std::endl;
        }
      } else if (verbose > Verbosity::Silent && comm.rank() == 0 &&
                 strain_init == IsStrainInitialised::True) {
        std::cout << "\nThe strain was initialised by the user!\n" << std::endl;
      }
      // Checking the consistancy of input load_steps and cell shape
      for (const auto & delF : load_steps) {
        if (not((delF.rows() == shape[0]) and (delF.cols() == shape[1]))) {
          std::stringstream err{};
          err << "Load increments need to be given in " << shape[0] << "×"
              << shape[1] << " matrices, but I got a " << delF.rows() << "×"
              << delF.cols() << " matrix:" << std::endl
              << delF;
          throw SolverError(err.str());
        }
        if (not check_symmetry(delF)) {
          throw SolverError("all Δε must be symmetric!");
        }
      }
      break;
    }
    default:
      throw SolverError("Unknown strain measure");
      break;
    }

    // initialise return value
    std::vector<OptimizeResult> ret_val{};

    // storage for the previous mean strain (to compute ΔF or Δε )
    Matrix_t previous_macro_strain{load_steps.back().Zero(shape[0], shape[1])};

    // strain field used by the cell for evaluating stresses/tangents.
    auto & eval_strain_field{cell.get_strain()};

    //! incremental loop (load steps)
    for (const auto & tup : akantu::enumerate(load_steps)) {
      const auto & strain_step{std::get<0>(tup)};
      const auto & macro_strain{std::get<1>(tup)};
      if ((verbose > Verbosity::Silent) and (comm.rank() == 0)) {
        std::cout << "at Load step " << std::setw(count_width)
                  << strain_step + 1 << std::endl;
      }

      // updating cell strain with the difference of the current and previous
      // strain input.
      auto && F_general_map{muGrid::FieldMap<Real, Mapping::Mut>(
          general_strain_field, shape[0], muGrid::IterUnit::SubPt)};
      for (auto && strain_general : F_general_map) {
        strain_general += macro_strain - previous_macro_strain;
      }

      // define the number of newton iteration (initially equal to 0)
      Uint newt_iter{0};
      std::string message{"Has not converged"};
      Real incr_norm{2 * newton_tol}, grad_norm{1};
      Real stress_norm{2 * equil_tol};
      bool has_converged{false};

      //! Checks two loop breaking criteria (newton tolerance and equilibrium
      //! tolerance)
      auto && early_convergence_test{
          [&incr_norm, &grad_norm, &newton_tol, &stress_norm, &equil_tol,
           &message, &has_converged, &convergence_criterion]() {
            convergence_criterion.get_newton_tol_test() =
                (incr_norm / grad_norm) <= newton_tol;
            convergence_criterion.get_equil_tol_test() =
                stress_norm < equil_tol;

            if (convergence_criterion.get_newton_tol_test()) {
              message = "Residual  tolerance reached";
            } else if (convergence_criterion.get_equil_tol_test()) {
              message = "Reached stress divergence tolerance";
            }
            has_converged = convergence_criterion.get_newton_tol_test() or
                            convergence_criterion.get_equil_tol_test();
            return has_converged;
          }};

      //! Checks all convergence criteria, including detection of linear
      //! problems
      auto && full_convergence_test{[&early_convergence_test, &cell,
                                     &has_converged, &message,
                                     &convergence_criterion]() {
        convergence_criterion.get_was_last_step_linear_test() =
            not cell.was_last_eval_non_linear();
        if (convergence_criterion.get_was_last_step_linear_test()) {
          message = "Linear problem, no more iteration necessary";
        }
        has_converged = early_convergence_test() or
                        convergence_criterion.get_was_last_step_linear_test();
        return has_converged;
      }};

      // Iterative update for the non-linear case
      for (; newt_iter < solver.get_maxiter() && !has_converged; ++newt_iter) {
        // updating the strain fields with the  eigen_strain field calculated by
        // the functor called eigen_strain_func

        if (not(eigen_strain_func == muGrid::nullopt)) {
          eval_strain_field = general_strain_field;
          (eigen_strain_func.value())(strain_step, eval_strain_field);
        }
        auto res_tup{cell.evaluate_stress_tangent()};
        auto & P{std::get<0>(res_tup)};

        auto & rhs{rhs_field.get_field()};

        rhs = -P;
        cell.apply_projection(rhs);
        stress_norm = std::sqrt(comm.sum(rhs.eigen_vec().squaredNorm()));

        if (early_convergence_test()) {
          break;
        }

        // calling sovler for solving the current (iteratively approximated)
        // linear equilibrium problem
        auto & incrF{incrF_field.get_field()};
        try {
          incrF = solver.solve(rhs.eigen_vec());
        } catch (ConvergenceError & error) {
          std::stringstream err{};
          err << "Failure at load step " << strain_step + 1 << " of "
              << load_steps.size() << ". In Newton-Raphson step " << newt_iter
              << ":" << std::endl
              << error.what() << std::endl
              << "The applied boundary condition is Δ" << strain_symb << " ="
              << std::endl
              << macro_strain << std::endl
              << "and the load increment is Δ" << strain_symb << " ="
              << std::endl
              << macro_strain - previous_macro_strain << std::endl;
          throw ConvergenceError(err.str());
        }

        // updating cell strain with the periodic (non-constant) solution
        // resulted from imposing the new macro_strain
        general_strain_field += incrF;

        // updating the incremental differences for checking the termination
        // criteria
        incr_norm = std::sqrt(comm.sum(incrF.eigen_vec().squaredNorm()));
        grad_norm =
            std::sqrt(comm.sum(eval_strain_field.eigen_vec().squaredNorm()));

        if ((verbose >= Verbosity::Detailed) and (comm.rank() == 0)) {
          std::cout << "at Newton step " << std::setw(count_width) << newt_iter
                    << ", |δ" << strain_symb << "|/|Δ" << strain_symb
                    << "| = " << std::setw(17) << incr_norm / grad_norm
                    << ", tol = " << newton_tol << std::endl;

          using StrainMap_t = muGrid::FieldMap<Real, Mapping::Const>;
          if (verbose > Verbosity::Detailed) {
            std::cout << "<" << strain_symb << "> =" << std::endl
                      << StrainMap_t{eval_strain_field, shape[0]}.mean()
                      << std::endl;
          }
        }
        full_convergence_test();
      }

      // throwing meaningful error message if the number of iterations for
      // newton_cg is exploited
      if (newt_iter == solver.get_maxiter()) {
        std::stringstream err{};
        err << "Failure at load step " << strain_step + 1 << " of "
            << load_steps.size() << ". Newton-Raphson failed to converge. "
            << "The applied boundary condition is Δ" << strain_symb << " ="
            << std::endl
            << macro_strain << std::endl
            << "and the load increment is Δ" << strain_symb << " =" << std::endl
            << macro_strain - previous_macro_strain << std::endl;
        throw ConvergenceError(err.str());
      }

      // update previous macroscopic strain
      previous_macro_strain = macro_strain;

      // re-evaluate cell in case the loop was terminated because of linearity
      if (convergence_criterion.get_was_last_step_linear_test() and
          not(convergence_criterion.get_newton_tol_test() or
              convergence_criterion.get_equil_tol_test())) {
        if (not(eigen_strain_func == muGrid::nullopt)) {
          eval_strain_field = general_strain_field;
          (eigen_strain_func.value())(strain_step, eval_strain_field);
        }
        cell.evaluate_stress_tangent();
      }

      // store results
      ret_val.emplace_back(OptimizeResult{
          general_strain_field.eigen_vec(), cell.get_stress().eigen_vec(),
          full_convergence_test(), Int(full_convergence_test()), message,
          newt_iter, solver.get_counter(), form});

      // store history variables for next load increment
      cell.save_history_variables();
      convergence_criterion.reset();
    }
    return ret_val;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<OptimizeResult>
  de_geus(Cell & cell, const LoadSteps_t & load_steps,
          KrylovSolverBase & solver, Real newton_tol, Real equil_tol,
          Verbosity verbose, IsStrainInitialised strain_init) {
    if (load_steps.size() == 0) {
      throw SolverError("No load steps specified.");
    }

    const auto & comm = cell.get_communicator();

    using Matrix_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    auto shape{cell.get_strain_shape()};

    ConvergenceCriterion convergence_criterion;

    // create a field collection to store workspaces
    auto field_collection{cell.get_fields().get_empty_clone()};
    // Corresponds to symbol δF or δε
    muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>> incrF_field{
        "incrF",         shape[0],         shape[1],
        IterUnit::SubPt, field_collection, QuadPtTag};

    // Corresponds to symbol ΔF or Δε
    muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>> DeltaF_field{
        "DeltaF",        shape[0],         shape[1],
        IterUnit::SubPt, field_collection, QuadPtTag};

    // field to store the rhs for cg calculations
    muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>> rhs_field{
        "rhs",           shape[0],         shape[1],
        IterUnit::SubPt, field_collection, QuadPtTag};
    solver.initialise();

    size_t count_width{};
    const auto form{cell.get_formulation()};
    std::string strain_symb{};
    if (verbose > Verbosity::Silent && comm.rank() == 0) {
      // setup of algorithm 5.2 in Nocedal, Numerical Optimization (p. 111)
      std::cout << "de Geus-" << solver.get_name() << " for ";
      switch (form) {
      case Formulation::small_strain: {
        strain_symb = "ε";
        std::cout << "small";
        break;
      }
      case Formulation::finite_strain: {
        strain_symb = "F";
        std::cout << "finite";
        break;
      }
      default:
        throw SolverError("unknown formulation");
        break;
      }
      std::cout << " strain with" << std::endl
                << "newton_tol = " << newton_tol
                << ", cg_tol = " << solver.get_tol()
                << " maxiter = " << solver.get_maxiter() << " and "
                << strain_symb << " from " << strain_symb << "₁ =" << std::endl
                << load_steps.front() << std::endl
                << " to " << strain_symb << "ₙ =" << std::endl
                << load_steps.back() << std::endl
                << "in increments of Δ" << strain_symb << " =" << std::endl
                << (load_steps.back() - load_steps.front()) / load_steps.size()
                << std::endl;
      count_width = static_cast<size_t>(std::log10(solver.get_maxiter())) + 1;
    }

    Matrix_t default_strain_val{};

    switch (form) {
    case Formulation::finite_strain: {
      if (strain_init == IsStrainInitialised::False) {
        // initilasing cell placement gradient (F) with identity matrix in case
        // of finite strain
        default_strain_val = Matrix_t::Identity(shape[0], shape[1]);
        cell.set_uniform_strain(default_strain_val);
        if (verbose > Verbosity::Silent && comm.rank() == 0) {
          std::cout << "The strain is initialised by default to the identity "
                       "matrix!"
                    << std::endl;
        }
      } else if (verbose > Verbosity::Silent && comm.rank() == 0 &&
                 strain_init == IsStrainInitialised::True) {
        std::cout << "The strain was initialised by the user!" << std::endl;
      }

      // Checking the consistancy of input load_steps and cell shape
      for (const auto & delF : load_steps) {
        auto rows = delF.rows();
        auto cols = delF.cols();
        if (not((rows == shape[0]) and (cols == shape[1]))) {
          std::stringstream err{};
          err << "Load increments need to be given in " << shape[0] << "×"
              << shape[1] << " matrices, but I got a " << delF.rows() << "×"
              << delF.cols() << " matrix:" << std::endl
              << delF;
          throw SolverError(err.str());
        }
      }
      break;
    }
    case Formulation::small_strain: {
      if (strain_init == IsStrainInitialised::False) {
        // initilasing cell strain (ε) with zero-filled matrix in case of small
        // strain
        default_strain_val = Matrix_t::Zero(shape[0], shape[1]);
        cell.set_uniform_strain(default_strain_val);
        if (verbose > Verbosity::Silent && comm.rank() == 0) {
          std::cout << "The strain is initialised by default to the zero "
                       "matrix!"
                    << std::endl;
        }
      } else if (verbose > Verbosity::Silent && comm.rank() == 0 &&
                 strain_init == IsStrainInitialised::True) {
        std::cout << "The strain was initialised by the user!" << std::endl;
      }

      // Checking the consistancy of input load_steps and cell shape
      for (const auto & delF : load_steps) {
        if (not((delF.rows() == shape[0]) and (delF.cols() == shape[1]))) {
          std::stringstream err{};
          err << "Load increments need to be given in " << shape[0] << "×"
              << shape[1] << " matrices, but I got a " << delF.rows() << "×"
              << delF.cols() << " matrix:" << std::endl
              << delF;
          throw SolverError(err.str());
        }
        if (not check_symmetry(delF)) {
          throw SolverError("all Δε must be symmetric!");
        }
      }
      break;
    }
    default:
      throw SolverError("Unknown strain measure");
      break;
    }

    // initialise return value
    std::vector<OptimizeResult> ret_val{};

    // storage for the previous mean strain (to compute ΔF or Δε)
    Matrix_t previous_macro_strain{load_steps.back().Zero(shape[0], shape[1])};

    // initialization of F (F in Formulation::finite_strain and ε in
    // Formuation::samll_strain)
    auto & F{cell.get_strain()};

    //! incremental loop
    for (const auto & tup : akantu::enumerate(load_steps)) {
      const auto & strain_step{std::get<0>(tup)};
      const auto & macro_strain{std::get<1>(tup)};
      if ((verbose > Verbosity::Silent) and (comm.rank() == 0)) {
        std::cout << "at Load step " << std::setw(count_width)
                  << strain_step + 1 << ", " << strain_symb << " =" << std::endl
                  << (macro_strain + default_strain_val).format(format)
                  << std::endl;
      }

      std::string message{"Has not converged"};
      Real incr_norm{2 * newton_tol}, grad_norm{1};
      Real stress_norm{2 * equil_tol};
      bool has_converged{false};

      //! Checks two loop breaking criteria (newton tolerance and equilibrium
      //! tolerance)
      auto && early_convergence_test{
          [&incr_norm, &grad_norm, &newton_tol, &stress_norm, &equil_tol,
           &message, &has_converged, &convergence_criterion]() {
            convergence_criterion.get_newton_tol_test() =
                (incr_norm / grad_norm) <= newton_tol;
            convergence_criterion.get_equil_tol_test() =
                stress_norm < equil_tol;

            if (convergence_criterion.get_newton_tol_test()) {
              message = "Residual  tolerance reached";
            } else if (convergence_criterion.get_equil_tol_test()) {
              message = "Reached stress divergence tolerance";
            }
            has_converged = convergence_criterion.get_newton_tol_test() or
                            convergence_criterion.get_equil_tol_test();
            return has_converged;
          }};

      //! Checks all convergence criteria, including detection of linear
      //! problems
      auto && full_convergence_test{[&early_convergence_test, &cell,
                                     &has_converged, &message,
                                     &convergence_criterion]() {
        convergence_criterion.get_was_last_step_linear_test() =
            not cell.was_last_eval_non_linear();
        if (convergence_criterion.get_was_last_step_linear_test()) {
          message = "Linear problem, no more iteration necessary";
        }
        has_converged = early_convergence_test() or
                        convergence_criterion.get_was_last_step_linear_test();
        return has_converged;
      }};

      Uint newt_iter{0};

      if (cell.was_last_eval_non_linear()) {
        // Iterative update in the nonlinear case
        for (; ((newt_iter < solver.get_maxiter()) and (!has_converged)) or
               (newt_iter < 2);
             ++newt_iter) {
          // obtain material response
          auto res_tup{cell.evaluate_stress_tangent()};
          auto & P{std::get<0>(res_tup)};

          auto & rhs{rhs_field.get_field()};
          auto & DeltaF{DeltaF_field.get_field()};
          auto & incrF{incrF_field.get_field()};
          try {
            if (newt_iter == 0) {
              DeltaF_field.get_map() = macro_strain - previous_macro_strain;
              // this corresponds to rhs=-G:K:δF
              cell.evaluate_projected_directional_stiffness(DeltaF, rhs);
              F += DeltaF;
              stress_norm =
                  std::sqrt(comm.sum(rhs.eigen_vec().matrix().squaredNorm()));
              if (stress_norm < equil_tol) {
                incrF.set_zero();
              } else {
                incrF = -solver.solve(rhs.eigen_vec());
              }
            } else {
              rhs = -P;
              cell.apply_projection(rhs);
              stress_norm =
                  std::sqrt(comm.sum(rhs.eigen_vec().matrix().squaredNorm()));
              if (stress_norm < equil_tol) {
                incrF.set_zero();
              } else {
                incrF = solver.solve(rhs.eigen_vec());
              }
            }
          } catch (ConvergenceError & error) {
            std::stringstream err{};
            err << "Failure at load step " << strain_step + 1 << " of "
                << load_steps.size() << ". In Newton-Raphson step " << newt_iter
                << ":" << std::endl
                << error.what() << std::endl
                << "The applied boundary condition is Δ" << strain_symb << " ="
                << std::endl
                << macro_strain << std::endl
                << "and the load increment is Δ" << strain_symb << " ="
                << std::endl
                << macro_strain - previous_macro_strain << std::endl;
            throw ConvergenceError(err.str());
          }

          // updating cell strain with the periodic (non-constant) solution
          // resulted from imposing the new macro_strain
          F += incrF;

          // updating the incremental differences for checking the termination
          // criteria
          incr_norm = std::sqrt(comm.sum(incrF.eigen_vec().squaredNorm()));
          grad_norm = std::sqrt(comm.sum(F.eigen_vec().squaredNorm()));

          if ((verbose >= Verbosity::Detailed) and (comm.rank() == 0)) {
            std::cout << "at Newton step " << std::setw(count_width)
                      << newt_iter << ", |δ" << strain_symb << "|/|Δ"
                      << strain_symb << "| = " << std::setw(17)
                      << incr_norm / grad_norm << ", tol = " << newton_tol
                      << std::endl;

            if (verbose > Verbosity::Detailed) {
              using StrainMap_t = muGrid::FieldMap<Real, Mapping::Const>;
              std::cout << "<" << strain_symb << "> =" << std::endl
                        << StrainMap_t{F, shape[0]}.mean() << std::endl;
            }
          }
          full_convergence_test();
        }
        if (newt_iter == solver.get_maxiter()) {
          std::stringstream err{};
          err << "Failure at load step " << strain_step + 1 << " of "
              << load_steps.size() << ". Newton-Raphson failed to converge. "
              << "The applied boundary condition is Δ" << strain_symb << " ="
              << std::endl
              << macro_strain << std::endl
              << "and the load increment is Δ" << strain_symb << " ="
              << std::endl
              << macro_strain - previous_macro_strain << std::endl;
          throw ConvergenceError(err.str());
        }
      } else {
        // Single strain update in the linear case

        // updating cell strain with the difference of the current and previous
        // strain input.
        auto && F_map{muGrid::FieldMap<Real, Mapping::Mut>(
            F, shape[0], muGrid::IterUnit::SubPt)};

        for (auto && strain : F_map) {
          strain += macro_strain - previous_macro_strain;
        }

        // obtain material response
        auto res_tup{cell.evaluate_stress_tangent()};
        auto & P{std::get<0>(res_tup)};

        auto & rhs{rhs_field.get_field()};

        rhs = -P;
        cell.apply_projection(rhs);
        stress_norm = std::sqrt(comm.sum(rhs.eigen_vec().squaredNorm()));

        if (!early_convergence_test()) {
          // calling sovler for solving the current (iteratively approximated)
          // linear equilibrium problem
          auto & incrF{incrF_field.get_field()};
          try {
            incrF = solver.solve(rhs.eigen_vec());
          } catch (ConvergenceError & error) {
            std::stringstream err{};
            err << "Failure at load step " << strain_step + 1 << " of "
                << load_steps.size() << ". In Newton-Raphson step " << newt_iter
                << ":" << std::endl
                << error.what() << std::endl
                << "The applied boundary condition is Δ" << strain_symb << " ="
                << std::endl
                << macro_strain << std::endl
                << "and the load increment is Δ" << strain_symb << " ="
                << std::endl
                << macro_strain - previous_macro_strain << std::endl;
            throw ConvergenceError(err.str());
          }
          // updating cell strain with the periodic (non-constant) solution
          // resulted from imposing the new macro_strain
          F += incrF;

          // updating the incremental differences for checking the termination
          // criteria
          incr_norm = std::sqrt(comm.sum(incrF.eigen_vec().squaredNorm()));
          grad_norm = std::sqrt(comm.sum(F.eigen_vec().squaredNorm()));

          if ((verbose >= Verbosity::Detailed) and (comm.rank() == 0)) {
            std::cout << "at Newton step " << std::setw(count_width)
                      << newt_iter << ", |δ" << strain_symb << "|/|Δ"
                      << strain_symb << "| = " << std::setw(17)
                      << incr_norm / grad_norm << ", tol = " << newton_tol
                      << std::endl;

            using StrainMap_t = muGrid::FieldMap<Real, Mapping::Const>;
            if (verbose > Verbosity::Detailed) {
              std::cout << "<" << strain_symb << "> =" << std::endl
                        << StrainMap_t{F, shape[0]}.mean() << std::endl;
            }
          }
          full_convergence_test();
        }
      }

      // update previous macroscopic strain
      previous_macro_strain = macro_strain;

      if (convergence_criterion.get_was_last_step_linear_test() and
          not(convergence_criterion.get_newton_tol_test() or
              convergence_criterion.get_equil_tol_test())) {
        cell.evaluate_stress_tangent();
      }

      // store results
      ret_val.emplace_back(
          OptimizeResult{F.eigen_vec(), cell.get_stress().eigen_vec(),
                         full_convergence_test(), Int(full_convergence_test()),
                         message, newt_iter, solver.get_counter(), form});

      // store history variables for next load increment
      cell.save_history_variables();
      convergence_criterion.reset();
    }

    return ret_val;
  }

  //--------------------------------------------------------------------------//
  std::vector<OptimizeResult> trust_region_newton_cg(
      Cell & cell, const LoadSteps_t & load_steps, KrylovSolverBase & solver,
      const Real & max_trust_region, const Real & newton_tol,
      const Real & equil_tol, const Real & inc_tr_tol, const Real & dec_tr_tol,
      const Verbosity & verbose, const IsStrainInitialised & strain_init,
      EigenStrainOptFunc_ref eigen_strain_func) {
    if (load_steps.size() == 0) {
      throw SolverError("No load steps specified.");
    }

    if (dec_tr_tol <= inc_tr_tol) {
      throw SolverError("dec_tr_tol must be larger than inc_tr_tol");
    }

    const auto & comm = cell.get_communicator();

    using Matrix_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    auto shape{cell.get_strain_shape()};

    ConvergenceCriterion convergence_criterion;

    // create a field collection to store workspaces
    auto field_collection{cell.get_fields().get_empty_clone()};
    // Corresponds to symbol δF or δε
    muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>> incrF_field{
        "incrF",         shape[0],         shape[1],
        IterUnit::SubPt, field_collection, QuadPtTag};

    // field to store the rhs for cg calculations
    muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>> rhs_field{
        "rhs",           shape[0],         shape[1],
        IterUnit::SubPt, field_collection, QuadPtTag};

    // field to store the predicted rhs from linear model calculations
    muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>> lin_rhs_field{
        "lin_rhs",           shape[0],         shape[1],
        IterUnit::SubPt, field_collection, QuadPtTag};

    // Standard strain field without the eigen strain: typically the strain
    // field given to the cell for stress/tangent evaluation, but in case of
    // presence of eigen strain, it is the strain field before adding the eigen
    // strain
    muGrid::TypedFieldBase<Real> & general_strain_field{
        [&cell, &field_collection,
            &eigen_strain_func]() -> muGrid::TypedFieldBase<Real> & {
          if (not(eigen_strain_func == muGrid::nullopt)) {
            // general strain field
            muGrid::RealField & general_strain_field{
                field_collection.register_real_field(
                    "general strain",
                    shape_for_formulation(cell.get_formulation(),
                                          cell.get_material_dim()),
                    QuadPtTag)};
            return general_strain_field;
          } else {
            return cell.get_strain();
          }
        }()};

    solver.initialise();

    size_t count_width{};
    const auto form{cell.get_formulation()};
    std::string strain_symb{};
    if (verbose > Verbosity::Silent && comm.rank() == 0) {
      // setup of algorithm 5.2 in Nocedal, Numerical Optimization (p. 111)
      std::cout << "Newton-" << solver.get_name() << " for ";
      switch (form) {
      case Formulation::small_strain: {
        strain_symb = "ε";
        std::cout << "small";
        break;
      }
      case Formulation::finite_strain: {
        strain_symb = "F";
        std::cout << "finite";
        break;
      }
      default:
        throw SolverError("unknown formulation");
        break;
      }
      std::cout << " strain with" << std::endl
                << "newton_tol = " << newton_tol
                << ", cg_tol = " << solver.get_tol()
                << " maxiter = " << solver.get_maxiter() << " and "
                << strain_symb << " from " << strain_symb << "₁ =" << std::endl
                << load_steps.front() << std::endl
                << " to " << strain_symb << "ₙ =" << std::endl
                << load_steps.back() << std::endl
                << "in increments of Δ" << strain_symb << " =" << std::endl
                << (load_steps.back() - load_steps.front()) / load_steps.size()
                << std::endl;
      count_width = static_cast<size_t>(std::log10(solver.get_maxiter())) + 1;
    }

    Matrix_t default_strain_val{};
    switch (form) {
    case Formulation::finite_strain: {
      if (strain_init == IsStrainInitialised::False) {
        // initialising cell placement gradient (F) with identity matrix in
        // case of finite strain
        default_strain_val = Matrix_t::Identity(shape[0], shape[1]);
        cell.set_uniform_strain(default_strain_val);

        if (verbose > Verbosity::Silent && comm.rank() == 0) {
          std::cout << "\nThe strain is initialised by default to the identity "
                       "matrix!\n"
                    << std::endl;
        }
      } else if (verbose > Verbosity::Silent && comm.rank() == 0 &&
                 strain_init == IsStrainInitialised::True) {
        std::cout << "\nThe strain was initialised by the user!\n" << std::endl;
      }
      // Checking the consistency of input load_steps and cell shape
      for (const auto & delF : load_steps) {
        if (not((delF.rows() == shape[0]) and (delF.cols() == shape[1]))) {
          std::stringstream err{};
          err << "Load increments need to be given in " << shape[0] << "×"
              << shape[1] << " matrices, but I got a " << delF.rows() << "×"
              << delF.cols() << " matrix:" << std::endl
              << delF;
          throw SolverError(err.str());
        }
      }
      break;
    }
    case Formulation::small_strain: {
      if (strain_init == IsStrainInitialised::False) {
        // initialising cell strain (ε) with zero-filled matrix in case of small
        // strain
        default_strain_val = Matrix_t::Zero(shape[0], shape[1]);
        cell.set_uniform_strain(default_strain_val);
        if (verbose > Verbosity::Silent && comm.rank() == 0) {
          std::cout << "\nThe strain is initialised by default to the zero "
                       "matrix!\n"
                    << std::endl;
        }
      } else if (verbose > Verbosity::Silent && comm.rank() == 0 &&
                 strain_init == IsStrainInitialised::True) {
        std::cout << "\nThe strain was initialised by the user!\n" << std::endl;
      }
      // Checking the consistency of input load_steps and cell shape
      for (const auto & delF : load_steps) {
        if (not((delF.rows() == shape[0]) and (delF.cols() == shape[1]))) {
          std::stringstream err{};
          err << "Load increments need to be given in " << shape[0] << "×"
              << shape[1] << " matrices, but I got a " << delF.rows() << "×"
              << delF.cols() << " matrix:" << std::endl
              << delF;
          throw SolverError(err.str());
        }
        if (not check_symmetry(delF)) {
          throw SolverError("all Δε must be symmetric!");
        }
      }
      break;
    }
    default:
      throw SolverError("Unknown strain measure");
      break;
    }

    // initialise return value
    std::vector<OptimizeResult> ret_val{};

    // storage for the previous mean strain (to compute ΔF or Δε )
    Matrix_t previous_macro_strain{load_steps.back().Zero(shape[0], shape[1])};

    // strain field used by the cell for evaluating stresses/tangents.
    auto & eval_strain_field{cell.get_strain()};

    //! incremental loop (load steps)
    for (const auto & tup : akantu::enumerate(load_steps)) {
      const auto & strain_step{std::get<0>(tup)};
      const auto & macro_strain{std::get<1>(tup)};
      if ((verbose > Verbosity::Silent) and (comm.rank() == 0)) {
        std::cout << "at Load step " << std::setw(count_width)
                  << strain_step + 1 << std::endl;
      }

      auto trust_region{max_trust_region};

      // updating cell strain with the difference of the current and previous
      // strain input.
      auto && F_general_map{muGrid::FieldMap<Real, Mapping::Mut>(
          general_strain_field, shape[0], muGrid::IterUnit::SubPt)};
      for (auto && strain_general : F_general_map) {
        strain_general += macro_strain - previous_macro_strain;
      }

      // define the number of newton iteration (initially equal to 0)
      std::string message{"Has not converged"};
      Real incr_norm{2 * newton_tol}, grad_norm{1};
      Real stress_norm{2 * equil_tol};
      bool has_converged{false};

      //! Checks two loop breaking criteria (newton tolerance and equilibrium
      //! tolerance)
      auto && early_convergence_test{
          [&incr_norm, &grad_norm, &newton_tol, &stress_norm, &equil_tol,
              &message, &has_converged, &convergence_criterion]() {
            convergence_criterion.get_newton_tol_test() =
                (incr_norm / grad_norm) <= newton_tol;
            convergence_criterion.get_equil_tol_test() =
                stress_norm < equil_tol;

            if (convergence_criterion.get_newton_tol_test()) {
              message = "Residual  tolerance reached";
            } else if (convergence_criterion.get_equil_tol_test()) {
              message = "Reached stress divergence tolerance";
            }
            has_converged = convergence_criterion.get_newton_tol_test() or
                            convergence_criterion.get_equil_tol_test();
            return has_converged;
          }};

      //! Checks all convergence criteria, including detection of linear
      //! problems
      auto && full_convergence_test{[&early_convergence_test, &cell,
                                        &has_converged, &message,
                                        &convergence_criterion]() {
        convergence_criterion.get_was_last_step_linear_test() =
            not cell.was_last_eval_non_linear();
        if (convergence_criterion.get_was_last_step_linear_test()) {
          message = "Linear problem, no more iteration necessary";
        }
        has_converged = early_convergence_test() or
                        convergence_criterion.get_was_last_step_linear_test();
        return has_converged;
      }};


      if ((verbose >= Verbosity::Detailed) and (comm.rank() == 0)) {
        std::cout << std::setw(count_width) << "STEP"
                  << std::setw(17) << "CG STATUS"
                  << std::setw(17) << "ACTION"
                  << std::setw(17) << "TRUST REGION"
                  << std::setw(17) << "|GP|"
                  << std::setw(17) << "|GP - (GP)_lin|"
                  << std::setw(19)
                  << ("|δ" + strain_symb + "|/|Δ" + strain_symb + "|")
                  << " (tol)" << std::endl;
      }

      // Iterative update for the non-linear case
      Uint newt_iter{0};
      for (; newt_iter < solver.get_maxiter() && !has_converged; ++newt_iter) {
        // updating the strain fields with the  eigen_strain field calculated by
        // the functor called eigen_strain_func

        if (not(eigen_strain_func == muGrid::nullopt)) {
          eval_strain_field = general_strain_field;
          (eigen_strain_func.value())(strain_step, eval_strain_field);
        }
        auto res_tup{cell.evaluate_stress_tangent()};
        auto & P{std::get<0>(res_tup)};

        // string descriptor for this step (for printing to screen)
        std::string action_str{""};
        std::string accept_str{
            solver.get_convergence() ==
                    KrylovSolverBase::Convergence::ReachedTolerance
                ? "newton"
            : solver.get_convergence() ==
              KrylovSolverBase::Convergence::HessianNotPositiveDefinite
              ? "neg. Hessian"
              : "tr exceeded"};

        auto & rhs{rhs_field.get_field()};
        auto & lin_rhs{lin_rhs_field.get_field()};
        auto & incrF{incrF_field.get_field()};

        rhs = -P;
        cell.apply_projection(rhs);
        stress_norm = std::sqrt(comm.sum(rhs.eigen_vec().squaredNorm()));
        auto stress_diff_norm{std::sqrt(
            comm.sum((rhs.eigen_vec() - lin_rhs.eigen_vec()).squaredNorm()))};

        // compare rhs prediction of linear model with actual reduction
        if (newt_iter > 0) {
          // Reduce the trust region if the norm of the stress difference
          // between the linear and the actual model is larger than dec_tr_tol.
          // Expand the trust region if this norm is smaller than inc_tr_tol.
          // This also means that dec_tr_tol > inc_tr_tol
          if (stress_diff_norm > dec_tr_tol) {
            // reduce trust region
            trust_region *= 0.25;
            action_str = "decrease tr";

            // If we need to reduce the trust region, do nothing and try again.
            // with the decreased trust region. This means we need to undo the
            // last step.
            general_strain_field -= incrF;
            // We do not need to update P here because it is a reference to the
            // internal stress field of the cell. Calling
            // evaluate_stress_tangent will simply update the internal buffer.
            // (We still know the rhs from the previous iterations.)
            cell.evaluate_stress_tangent();
          } else if (
              stress_diff_norm < inc_tr_tol and
              (solver.get_convergence() ==
                   KrylovSolverBase::Convergence::HessianNotPositiveDefinite or
               solver.get_convergence() ==
                   KrylovSolverBase::Convergence::ExceededTrustRegionBound)) {
            // Increase trust region for the next step
            trust_region = std::min(2 * trust_region, max_trust_region);
            action_str = "increase tr";
          }

          if ((verbose >= Verbosity::Detailed) and (comm.rank() == 0)) {
            std::cout << std::setw(count_width) << (newt_iter - 1)
                      << std::setw(17) << accept_str
                      << std::setw(17) << action_str
                      << std::setw(17) << trust_region
                      << std::setw(17) << stress_norm
                      << std::setw(17) << stress_diff_norm
                      << std::setw(17) << incr_norm / grad_norm << " ("
                      << newton_tol << ")" << std::endl;
          }
        }

        if (early_convergence_test()) {
          break;
        }

        // calling solver for solving the current (iteratively approximated)
        // linear equilibrium problem
        try {
          solver.set_trust_region(trust_region);
          KrylovSolverBase::ConstVector_ref rhs_vec{rhs.eigen_vec()};
          incrF = solver.solve(rhs_vec);
          KrylovSolverBase::ConstVector_ref incrF_vec{incrF.eigen_vec()};
          auto lin_rhs_vec{lin_rhs.eigen_vec()};
          lin_rhs_vec = rhs_vec - cell.get_adaptor() * incrF_vec;
        } catch (ConvergenceError & error) {
          std::stringstream err{};
          err << "Failure at load step " << strain_step + 1 << " of "
              << load_steps.size() << ". In Trust-region Newton-CG step "
              << newt_iter << ":" << std::endl
              << error.what() << std::endl
              << "The applied boundary condition is Δ" << strain_symb << " ="
              << std::endl
              << macro_strain << std::endl
              << "and the load increment is Δ" << strain_symb << " ="
              << std::endl
              << macro_strain - previous_macro_strain << std::endl;
          throw ConvergenceError(err.str());
        }

        // updating cell strain with the periodic (non-constant) solution
        // resulted from imposing the new macro_strain
        general_strain_field += incrF;

        // updating the incremental differences for checking the termination
        // criteria/model energy
        incr_norm = std::sqrt(comm.sum(incrF.eigen_vec().squaredNorm()));
        grad_norm =
            std::sqrt(comm.sum(eval_strain_field.eigen_vec().squaredNorm()));

        full_convergence_test();
      }

      // throwing meaningful error message if the number of iterations for
      // newton_cg is exploited
      if (newt_iter == solver.get_maxiter()) {
        std::stringstream err{};
        err << "Failure at load step " << strain_step + 1 << " of "
            << load_steps.size() << ". Newton-Raphson failed to converge. "
            << "The applied boundary condition is Δ" << strain_symb << " ="
            << std::endl
            << macro_strain << std::endl
            << "and the load increment is Δ" << strain_symb << " =" << std::endl
            << macro_strain - previous_macro_strain << std::endl;
        throw ConvergenceError(err.str());
      }

      // update previous macroscopic strain
      previous_macro_strain = macro_strain;

      // re-evaluate cell in case the loop was terminated because of linearity
      if (convergence_criterion.get_was_last_step_linear_test() and
          not(convergence_criterion.get_newton_tol_test() or
              convergence_criterion.get_equil_tol_test())) {
        if (not(eigen_strain_func == muGrid::nullopt)) {
          eval_strain_field = general_strain_field;
          (eigen_strain_func.value())(strain_step, eval_strain_field);
        }
        cell.evaluate_stress_tangent();
      }

      // store results
      ret_val.emplace_back(OptimizeResult{
          general_strain_field.eigen_vec(), cell.get_stress().eigen_vec(),
          full_convergence_test(), Int(full_convergence_test()), message,
          newt_iter, solver.get_counter(), form});

      // store history variables for next load increment
      cell.save_history_variables();
      convergence_criterion.reset();
    }
    return ret_val;
  }

  /* ---------------------------------------------------------------------- */

}  // namespace muSpectre
