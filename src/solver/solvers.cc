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

#include <libmugrid/iterators.hh>
#include <libmugrid/mapped_field.hh>

#include <Eigen/Dense>

#include <iomanip>
#include <iostream>

namespace muSpectre {

  //! produces numpy-compatible full precision text output. great for debugging
  Eigen::IOFormat format(Eigen::FullPrecision, 0, ", ", ",\n", "[", "]", "[",
                         "]");

  //--------------------------------------------------------------------------//
  std::vector<OptimizeResult>
  newton_cg(Cell & cell, const LoadSteps_t & load_steps,
            KrylovSolverBase & solver, Real newton_tol, Real equil_tol,
            Verbosity verbose, IsStrainInitialised strain_init) {
    const auto & comm = cell.get_communicator();

    using Matrix_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

    auto shape{cell.get_strain_shape()};

    // create a field collection to store workspaces
    auto field_collection{cell.get_fields().get_empty_clone()};
    // Corresponds to symbol δF or δε
    muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>> incrF_field{
        "incrF",          shape[0], shape[1], IterUnit::SubPt,
        field_collection, QuadPtTag};

    // field to store the rhs for cg calculations
    muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>> rhs_field{
        "rhs",    shape[0], shape[1], IterUnit::SubPt, field_collection,
        QuadPtTag};

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
      count_width = size_t(std::log10(solver.get_maxiter())) + 1;
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

    auto & F{cell.get_strain()};
    //! incremental loop
    for (const auto & tup : akantu::enumerate(load_steps)) {
      const auto & strain_step{std::get<0>(tup)};
      const auto & macro_strain{std::get<1>(tup)};
      if ((verbose > Verbosity::Silent) and (comm.rank() == 0)) {
        std::cout << "at Load step " << std::setw(count_width)
                  << strain_step + 1 << std::endl;
      }
      // updating cell strain with the difference of the current and previous
      // strain input.
      for (auto && strain : muGrid::FieldMap<Real, Mapping::Mut>(
               F, shape[0], muGrid::IterUnit::SubPt)) {
        strain += macro_strain - previous_macro_strain;
      }

      // define the number of newton iteration (initially equal to 0)
      Uint newt_iter{0};
      std::string message{"Has not converged"};
      Real incr_norm{2 * newton_tol}, grad_norm{1};
      Real stress_norm{2 * equil_tol};

      //! Checks two loop breaking criteria (newton tolerance and equilibrium
      //! tolerance)
      auto convergence_test = [&incr_norm, &grad_norm, &newton_tol,
                               &stress_norm, &equil_tol, &message]() {
        bool incr_test = incr_norm / grad_norm <= newton_tol;
        bool stress_test = stress_norm < equil_tol;
        if (incr_test) {
          message = "Residual  tolerance reached";
        } else if (stress_test) {
          message = "Reached stress divergence tolerance";
        }
        return incr_test || stress_test;
      };

      //! carries out a single newton_cg solving step
      auto iterate_solve = [&cell, &rhs_field, &incrF_field, &F, &stress_norm,
                            &incr_norm, &grad_norm, &comm, &convergence_test,
                            &solver, &strain_step, &load_steps, &macro_strain,
                            &previous_macro_strain, &newt_iter, &strain_symb,
                            &verbose, &count_width, &newton_tol, &shape]() {
        // obtain material response
        auto res_tup{cell.evaluate_stress_tangent()};
        auto & P{std::get<0>(res_tup)};
        auto & rhs{rhs_field.get_field()};

        rhs = -P;
        cell.apply_projection(rhs);
        stress_norm = std::sqrt(comm.sum(rhs.eigen_vec().squaredNorm()));

        if (convergence_test()) {
          return true;
        }
        // calling solver for solving the current (iteratively approximated)
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
          std::cout << "at Newton step " << std::setw(count_width) << newt_iter
                    << ", |δ" << strain_symb << "|/|Δ" << strain_symb
                    << "| = " << std::setw(17) << incr_norm / grad_norm
                    << ", tol = " << newton_tol << std::endl;

          using StrainMap_t = muGrid::FieldMap<Real, Mapping::Const>;
          if (verbose > Verbosity::Detailed) {
            std::cout << "<" << strain_symb << "> =" << std::endl
                      << StrainMap_t{F, shape[0]}.mean() << std::endl;
          }
        }
        return convergence_test();
      };

      if (cell.is_non_linear()) {
        // Iterative update for the non-linear case
        for (; newt_iter < solver.get_maxiter(); ++newt_iter) {
          bool converged{iterate_solve()};
          if (converged) {
            break;
          }
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
              << "and the load increment is Δ" << strain_symb << " ="
              << std::endl
              << macro_strain - previous_macro_strain << std::endl;
          throw ConvergenceError(err.str());
        }
      } else {
        // Single strain update for the linear case
        iterate_solve();
      }

      // update previous macroscopic strain
      previous_macro_strain = macro_strain;

      // store results
      ret_val.emplace_back(
          OptimizeResult{F.eigen_vec(), cell.get_stress().eigen_vec(),
                         convergence_test(), Int(convergence_test()), message,
                         newt_iter, solver.get_counter(), form});

      // store history variables for next load increment
      cell.save_history_variables();
    }

    return ret_val;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<OptimizeResult>
  de_geus(Cell & cell, const LoadSteps_t & load_steps,
          KrylovSolverBase & solver, Real newton_tol, Real equil_tol,
          Verbosity verbose, IsStrainInitialised strain_init) {
    const auto & comm = cell.get_communicator();

    using Matrix_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

    auto shape{cell.get_strain_shape()};

    // create a field collection to store workspaces
    auto field_collection{cell.get_fields().get_empty_clone()};
    // Corresponds to symbol δF or δε
    muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>> incrF_field{
        "incrF",          shape[0], shape[1], IterUnit::SubPt,
        field_collection, QuadPtTag};

    // Corresponds to symbol ΔF or Δε
    muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>> DeltaF_field{
        "DeltaF",           shape[0],         shape[1],
        IterUnit::SubPt, field_collection, QuadPtTag};

    // field to store the rhs for cg calculations
    muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>> rhs_field{
        "rhs",    shape[0], shape[1], IterUnit::SubPt, field_collection,
        QuadPtTag};
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
      count_width = size_t(std::log10(solver.get_maxiter())) + 1;
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

      auto convergence_test = [&incr_norm, &grad_norm, &newton_tol,
                               &stress_norm, &equil_tol, &message,
                               &has_converged]() {
        bool incr_test = incr_norm / grad_norm <= newton_tol;
        bool stress_test = stress_norm < equil_tol;
        if (incr_test) {
          message = "Residual  tolerance reached";
        } else if (stress_test) {
          message = "Reached stress divergence tolerance";
        }
        has_converged = incr_test || stress_test;
        return has_converged;
      };
      Uint newt_iter{0};

      if (cell.is_non_linear()) {
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
          convergence_test();
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
        for (auto && strain : muGrid::FieldMap<Real, Mapping::Mut>(
                 F, shape[0], muGrid::IterUnit::SubPt)) {
          strain += macro_strain - previous_macro_strain;
        }

        // obtain material response
        auto res_tup{cell.evaluate_stress_tangent()};
        auto & P{std::get<0>(res_tup)};

        auto & rhs{rhs_field.get_field()};

        rhs = -P;
        cell.apply_projection(rhs);
        stress_norm = std::sqrt(comm.sum(rhs.eigen_vec().squaredNorm()));

        if (!convergence_test()) {
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
        }
      }

      // update previous macroscopic strain
      previous_macro_strain = macro_strain;

      // store results
      ret_val.emplace_back(
          OptimizeResult{F.eigen_vec(), cell.get_stress().eigen_vec(),
                         convergence_test(), Int(convergence_test()), message,
                         newt_iter, solver.get_counter(), form});

      // store history variables for next load increment
      cell.save_history_variables();
    }

    return ret_val;
  }
}  // namespace muSpectre
