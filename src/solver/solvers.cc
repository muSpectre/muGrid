/**
 * file   solvers.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Apr 2018
 *
 * @brief  implementation of dynamic newton-cg solver
 *
 * @section LICENSE
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "solvers.hh"

#include <Eigen/Dense>

#include <iomanip>

namespace muSpectre {

  //----------------------------------------------------------------------------//
  std::vector<OptimizeResult>
  newton_cg(Cell & cell, const LoadSteps_t & load_steps,
            SolverBase & solver, Real newton_tol,
            Real equil_tol, Dim_t verbose) {
    const Communicator & comm = cell.get_communicator();

    using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using Matrix_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

    // Corresponds to symbol δF or δε
    Vector_t incrF(cell.get_nb_dof());

    // field to store the rhs for cg calculations
    Vector_t rhs(cell.get_nb_dof());

    solver.initialise();


    size_t count_width{};
    const auto form{cell.get_formulation()};
    std::string strain_symb{};
    if (verbose > 0 && comm.rank() == 0) {
      //setup of algorithm 5.2 in Nocedal, Numerical Optimization (p. 111)
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
                << "newton_tol = " << newton_tol << ", cg_tol = "
                << solver.get_tol() << " maxiter = " << solver.get_maxiter() << " and Δ"
                << strain_symb << " =" <<std::endl;
      for (auto&& tup: akantu::enumerate(load_steps)) {
        auto && counter{std::get<0>(tup)};
        auto && grad{std::get<1>(tup)};
        std::cout << "Step " << counter + 1 << ":" << std::endl
                  << grad << std::endl;
      }
      count_width = size_t(std::log10(solver.get_maxiter()))+1;
    }

    auto shape{cell.get_strain_shape()};

    switch (form) {
    case Formulation::finite_strain: {
      cell.set_uniform_strain(Matrix_t::Identity(shape[0], shape[1]));
      for (const auto & delF: load_steps) {
        if (not ((delF.rows() == shape[0]) and (delF.cols() == shape[1]))) {
          std::stringstream err{};
          err << "Load increments need to be given in " << shape[0] << "×"
              << shape[1] << " matrices, but I got a " << delF.rows() << "×"
              << delF.cols() << " matrix:" << std::endl << delF;
          throw SolverError(err.str());
        }
      }
      break;
    }
    case Formulation::small_strain: {
      cell.set_uniform_strain(Matrix_t::Zero(shape[0], shape[1]));
      for (const auto & delF: load_steps) {
        if (not ((delF.rows() == shape[0]) and (delF.cols() == shape[1]))) {
          std::stringstream err{};
          err << "Load increments need to be given in " << shape[0] << "×"
              << shape[1] << " matrices, but I got a " << delF.rows() << "×"
              << delF.cols() << " matrix:" << std::endl << delF;
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

    auto F{cell.get_strain_vector()};
    //! incremental loop
    for (const auto & macro_strain: load_steps) {
      using StrainMap_t = RawFieldMap<Eigen::Map<Eigen::MatrixXd>>;
      for (auto && strain: StrainMap_t(F, shape[0], shape[1])) {
        strain += macro_strain - previous_macro_strain;
      }

      std::string message{"Has not converged"};
      Real incr_norm{2*newton_tol}, grad_norm{1};
      Real stress_norm{2*equil_tol};
      bool has_converged{false};

      auto convergence_test = [&incr_norm, &grad_norm, &newton_tol,
                               &stress_norm, &equil_tol, &message, &has_converged] () {
        bool incr_test = incr_norm/grad_norm <= newton_tol;
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

      for (;
           newt_iter < solver.get_maxiter() && !has_converged;
           ++newt_iter) {
        // obtain material response
        auto res_tup{cell.evaluate_stress_tangent()};
        auto & P{std::get<0>(res_tup)};

        rhs = -P;
        cell.apply_projection(rhs);
        stress_norm = std::sqrt(comm.sum(rhs.squaredNorm()));

        if (convergence_test()) {
          break;
        }

        //! this is a potentially avoidable copy TODO: check this out
        incrF = solver.solve(rhs);

        F += incrF;

        incr_norm = std::sqrt(comm.sum(incrF.squaredNorm()));
        grad_norm = std::sqrt(comm.sum(F.squaredNorm()));

        if ((verbose > 0) and (comm.rank() == 0)) {
          std::cout << "at Newton step " << std::setw(count_width) << newt_iter
                    << ", |δ" << strain_symb << "|/|Δ" << strain_symb
                    << "| = " << std::setw(17) << incr_norm/grad_norm
                    << ", tol = " << newton_tol << std::endl;

          if (verbose-1>1) {
            std::cout << "<" << strain_symb << "> =" << std::endl
                      << StrainMap_t(F, shape[0], shape[1]).mean() << std::endl;
          }
        }
        convergence_test();
      }


      // update previous macroscopic strain
      previous_macro_strain = macro_strain;

      // store results
      ret_val.emplace_back(OptimizeResult{F, cell.get_stress_vector(),
            convergence_test(), Int(convergence_test()),
            message, newt_iter, solver.get_counter()});

      // store history variables for next load increment
      cell.save_history_variables();
    }

    return ret_val;
  }

  //----------------------------------------------------------------------------//
  std::vector<OptimizeResult>
  de_geus(Cell & cell,
          const LoadSteps_t & load_steps,
          SolverBase & solver, Real newton_tol,
          Real equil_tol,
          Dim_t verbose) {
    const Communicator & comm = cell.get_communicator();

    using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using Matrix_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

    // Corresponds to symbol δF or δε
    Vector_t incrF(cell.get_nb_dof());

    // Corresponds to symbol ΔF or Δε
    Vector_t DeltaF(cell.get_nb_dof());

    // field to store the rhs for cg calculations
    Vector_t rhs(cell.get_nb_dof());

    solver.initialise();


    size_t count_width{};
    const auto form{cell.get_formulation()};
    std::string strain_symb{};
    if (verbose > 0 && comm.rank() == 0) {
      //setup of algorithm 5.2 in Nocedal, Numerical Optimization (p. 111)
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
                << "newton_tol = " << newton_tol << ", cg_tol = "
                << solver.get_tol() << " maxiter = " << solver.get_maxiter() << " and Δ"
                << strain_symb << " =" <<std::endl;
      for (auto&& tup: akantu::enumerate(load_steps)) {
        auto && counter{std::get<0>(tup)};
        auto && grad{std::get<1>(tup)};
        std::cout << "Step " << counter + 1 << ":" << std::endl
                  << grad << std::endl;
      }
      count_width = size_t(std::log10(solver.get_maxiter()))+1;
    }

    auto shape{cell.get_strain_shape()};

    switch (form) {
    case Formulation::finite_strain: {
      cell.set_uniform_strain(Matrix_t::Identity(shape[0], shape[1]));
      for (const auto & delF: load_steps) {
        auto rows = delF.rows();
        auto cols = delF.cols();
        if (not ((rows == shape[0]) and (cols == shape[1]))) {
          std::stringstream err{};
          err << "Load increments need to be given in " << shape[0] << "×"
              << shape[1] << " matrices, but I got a " << delF.rows() << "×"
              << delF.cols() << " matrix:" << std::endl << delF;
          throw SolverError(err.str());
        }
      }
      break;
    }
    case Formulation::small_strain: {
      cell.set_uniform_strain(Matrix_t::Zero(shape[0], shape[1]));
      for (const auto & delF: load_steps) {
        if (not ((delF.rows() == shape[0]) and (delF.cols() == shape[1]))) {
          std::stringstream err{};
          err << "Load increments need to be given in " << shape[0] << "×"
              << shape[1] << " matrices, but I got a " << delF.rows() << "×"
              << delF.cols() << " matrix:" << std::endl << delF;
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

    auto F{cell.get_strain_vector()};
    //! incremental loop
    for (const auto & macro_strain: load_steps) {
      using StrainMap_t = RawFieldMap<Eigen::Map<Eigen::MatrixXd>>;

      std::string message{"Has not converged"};
      Real incr_norm{2*newton_tol}, grad_norm{1};
      Real stress_norm{2*equil_tol};
      bool has_converged{false};

      auto convergence_test = [&incr_norm, &grad_norm, &newton_tol,
                               &stress_norm, &equil_tol, &message, &has_converged] () {
        bool incr_test = incr_norm/grad_norm <= newton_tol;
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

      for (;
           newt_iter < solver.get_maxiter() && !has_converged;
           ++newt_iter) {
        // obtain material response
        auto res_tup{cell.evaluate_stress_tangent()};
        auto & P{std::get<0>(res_tup)};

        if (newt_iter == 0) {
          for (auto && strain: StrainMap_t(DeltaF, shape[0], shape[1])) {
            strain = macro_strain -previous_macro_strain;
          }
          rhs = - cell.evaluate_projected_directional_stiffness(DeltaF);
          stress_norm = std::sqrt(comm.sum(rhs.matrix().squaredNorm()));
          if (convergence_test()) {
            break;
          }
          incrF = solver.solve(rhs);
          F += DeltaF;
        }
        else {
          rhs = -P;
          cell.apply_projection(rhs);
          stress_norm = std::sqrt(comm.sum(rhs.matrix().squaredNorm()));
          if (convergence_test()) {
            break;
          }
          incrF = solver.solve(rhs);
        }

        F += incrF;

        incr_norm = std::sqrt(comm.sum(incrF.squaredNorm()));
        grad_norm = std::sqrt(comm.sum(F.squaredNorm()));

        if ((verbose > 0) and (comm.rank() == 0)) {
          std::cout << "at Newton step " << std::setw(count_width) << newt_iter
                    << ", |δ" << strain_symb << "|/|Δ" << strain_symb
                    << "| = " << std::setw(17) << incr_norm/grad_norm
                    << ", tol = " << newton_tol << std::endl;

          if (verbose-1>1) {
            std::cout << "<" << strain_symb << "> =" << std::endl
                      << StrainMap_t(F, shape[0], shape[1]).mean() << std::endl;
          }
        }
        convergence_test();
      }


      // update previous macroscopic strain
      previous_macro_strain = macro_strain;

      // store results
      ret_val.emplace_back(OptimizeResult{F, cell.get_stress_vector(),
            convergence_test(), Int(convergence_test()),
            message, newt_iter, solver.get_counter()});

      // store history variables for next load increment
      cell.save_history_variables();
    }

    return ret_val;
  }
}  // muSpectre
