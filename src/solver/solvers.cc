/**
 * @file   solvers.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2017
 *
 * @brief  implementation of solver functions
 *
 * Copyright © 2017 Till Junge
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

#include "solver/solvers.hh"
#include "solver/solver_cg.hh"
#include "common/iterators.hh"

#include <Eigen/IterativeLinearSolvers>

#include <iomanip>
#include <cmath>

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM>
  std::vector<OptimizeResult>
  de_geus (CellBase<DimS, DimM> & cell, const GradIncrements<DimM> & delFs,
           SolverBase<DimS, DimM> & solver, Real newton_tol,
           Real equil_tol,
           Dim_t verbose) {
    using Field_t = typename MaterialBase<DimS, DimM>::StrainField_t;
    auto solver_fields{std::make_unique<GlobalFieldCollection<DimS>>()};
    solver_fields->initialise(cell.get_resolutions());

    // Corresponds to symbol δF or δε
    auto & incrF{make_field<Field_t>("δF", *solver_fields)};

    // Corresponds to symbol ΔF or Δε
    auto & DeltaF{make_field<Field_t>("ΔF", *solver_fields)};

    // field to store the rhs for cg calculations
    auto & rhs{make_field<Field_t>("rhs", *solver_fields)};

    solver.initialise();


    if (solver.get_maxiter() == 0) {
      solver.set_maxiter(cell.size()*DimM*DimM*10);
    }

    size_t count_width{};
    const auto form{cell.get_formulation()};
    std::string strain_symb{};
    if (verbose > 0) {
      //setup of algorithm 5.2 in Nocedal, Numerical Optimization (p. 111)
      std::cout << "de Geus-" << solver.name() << " for ";
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
      for (auto&& tup: akantu::enumerate(delFs)) {
        auto && counter{std::get<0>(tup)};
        auto && grad{std::get<1>(tup)};
        std::cout << "Step " << counter + 1 << ":" << std::endl
                  << grad << std::endl;
      }
      count_width = size_t(std::log10(solver.get_maxiter()))+1;
    }

    // initialise F = I or ε = 0
    auto & F{cell.get_strain()};
    switch (form) {
    case Formulation::finite_strain: {
      F.get_map() = Matrices::I2<DimM>();
      break;
    }
    case Formulation::small_strain: {
      F.get_map() = Matrices::I2<DimM>().Zero();
      for (const auto & delF: delFs) {
        if (!check_symmetry(delF)) {
          throw SolverError("all Δε must be symmetric!");
        }
      }
      break;
    }
    default:
      throw SolverError("Unknown formulation");
      break;
    }

    // initialise return value
    std::vector<OptimizeResult> ret_val{};

    // initialise materials
    constexpr bool need_tangent{true};
    cell.initialise_materials(need_tangent);

    Grad_t<DimM> previous_grad{Grad_t<DimM>::Zero()};
    for (const auto & delF: delFs) { //incremental loop

      std::string message{"Has not converged"};
      Real incrNorm{2*newton_tol}, gradNorm{1};
      Real stressNorm{2*equil_tol};
      bool has_converged{false};
      auto convergence_test = [&incrNorm, &gradNorm, &newton_tol,
                               &stressNorm, &equil_tol, &message, &has_converged] () {
        bool incr_test = incrNorm/gradNorm <= newton_tol;
        bool stress_test = stressNorm < equil_tol;
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
           (newt_iter < solver.get_maxiter()) && (!has_converged ||
                                     (newt_iter==1));
           ++newt_iter) {

        // obtain material response
        auto res_tup{cell.evaluate_stress_tangent(F)};
        auto & P{std::get<0>(res_tup)};
        auto & K{std::get<1>(res_tup)};

        auto tangent_effect = [&cell, &K] (const Field_t & dF, Field_t & dP) {
          cell.directional_stiffness(K, dF, dP);
        };


        if (newt_iter == 0) {
          DeltaF.get_map() = -(delF-previous_grad); // neg sign because rhs
          tangent_effect(DeltaF, rhs);
          stressNorm = rhs.eigen().matrix().norm();
          if (convergence_test()) {
            break;
          }
          incrF.eigenvec() = solver.solve(rhs.eigenvec(), incrF.eigenvec());
          F.eigen() -= DeltaF.eigen();
        } else {
          rhs.eigen() = -P.eigen();
          cell.project(rhs);
          stressNorm = rhs.eigen().matrix().norm();
          if (convergence_test()) {
            break;
          }
          incrF.eigen() = 0;
          incrF.eigenvec() = solver.solve(rhs.eigenvec(), incrF.eigenvec());
        }

        F.eigen() += incrF.eigen();

        incrNorm = incrF.eigen().matrix().norm();
        gradNorm = F.eigen().matrix().norm();
        if (verbose>0) {
          std::cout << "at Newton step " << std::setw(count_width) << newt_iter
                    << ", |δ" << strain_symb << "|/|Δ" << strain_symb
                    << "| = " << std::setw(17) << incrNorm/gradNorm
                    << ", tol = " << newton_tol << std::endl;
          if (verbose-1>1) {
            std::cout << "<" << strain_symb << "> =" << std::endl
                      << F.get_map().mean() << std::endl;
          }
        }
        convergence_test();

      }
      // update previous gradient
      previous_grad = delF;

      ret_val.push_back(OptimizeResult{F.eigen(), cell.get_stress().eigen(),
            has_converged, Int(has_converged),
            message,
            newt_iter, solver.get_counter()});


      // store history variables here
      cell.save_history_variables();

    }

    return ret_val;

  }

  //! instantiation for two-dimensional cells
  template std::vector<OptimizeResult>
  de_geus (CellBase<twoD, twoD> & cell, const GradIncrements<twoD>& delF0,
           SolverBase<twoD, twoD> & solver, Real newton_tol,
           Real equil_tol,
           Dim_t verbose);

  // template typename CellBase<twoD, threeD>::StrainField_t &
  // de_geus (CellBase<twoD, threeD> & cell, const GradIncrements<threeD>& delF0,
  //            const Real cg_tol, const Real newton_tol, Uint maxiter,
  //            Dim_t verbose);

  //! instantiation for three-dimensional cells
  template std::vector<OptimizeResult>
  de_geus (CellBase<threeD, threeD> & cell, const GradIncrements<threeD>& delF0,
           SolverBase<threeD, threeD> & solver, Real newton_tol,
           Real equil_tol,
           Dim_t verbose);

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  std::vector<OptimizeResult>
  newton_cg (CellBase<DimS, DimM> & cell, const GradIncrements<DimM> & delFs,
             SolverBase<DimS, DimM> & solver, Real newton_tol,
             Real equil_tol,
             Dim_t verbose) {
    using Field_t = typename MaterialBase<DimS, DimM>::StrainField_t;
    auto solver_fields{std::make_unique<GlobalFieldCollection<DimS>>()};
    solver_fields->initialise(cell.get_resolutions());

    // Corresponds to symbol δF or δε
    auto & incrF{make_field<Field_t>("δF", *solver_fields)};

    // field to store the rhs for cg calculations
    auto & rhs{make_field<Field_t>("rhs", *solver_fields)};

    solver.initialise();

    if (solver.get_maxiter() == 0) {
      solver.set_maxiter(cell.size()*DimM*DimM*10);
    }

    size_t count_width{};
    const auto form{cell.get_formulation()};
    std::string strain_symb{};
    if (verbose > 0) {
      //setup of algorithm 5.2 in Nocedal, Numerical Optimization (p. 111)
      std::cout << "Newton-" << solver.name() << " for ";
      switch (form) {
      case Formulation::small_strain: {
        strain_symb = "ε";
        std::cout << "small";
        break;
      }
      case Formulation::finite_strain: {
        strain_symb = "Fy";
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
      for (auto&& tup: akantu::enumerate(delFs)) {
        auto && counter{std::get<0>(tup)};
        auto && grad{std::get<1>(tup)};
        std::cout << "Step " << counter + 1 << ":" << std::endl
                  << grad << std::endl;
      }
      count_width = size_t(std::log10(solver.get_maxiter()))+1;
    }

    // initialise F = I or ε = 0
    auto & F{cell.get_strain()};
    switch (cell.get_formulation()) {
    case Formulation::finite_strain: {
      F.get_map() = Matrices::I2<DimM>();
      break;
    }
    case Formulation::small_strain: {
      F.get_map() = Matrices::I2<DimM>().Zero();
      for (const auto & delF: delFs) {
        if (!check_symmetry(delF)) {
          throw SolverError("all Δε must be symmetric!");
        }
      }
      break;
    }
    default:
      throw SolverError("Unknown formulation");
      break;
    }

    // initialise return value
    std::vector<OptimizeResult> ret_val{};

    // initialise materials
    constexpr bool need_tangent{true};
    cell.initialise_materials(need_tangent);

    Grad_t<DimM> previous_grad{Grad_t<DimM>::Zero()};
    for (const auto & delF: delFs) { //incremental loop
      // apply macroscopic strain increment
      for (auto && grad: F.get_map()) {
        grad += delF - previous_grad;
      }

      std::string message{"Has not converged"};
      Real incrNorm{2*newton_tol}, gradNorm{1};
      Real stressNorm{2*equil_tol};
      bool has_converged{false};
      auto convergence_test = [&incrNorm, &gradNorm, &newton_tol,
                               &stressNorm, &equil_tol, &message, &has_converged] () {
        bool incr_test = incrNorm/gradNorm <= newton_tol;
        bool stress_test = stressNorm < equil_tol;
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
        auto res_tup{cell.evaluate_stress_tangent(F)};
        auto & P{std::get<0>(res_tup)};

        rhs.eigen() = -P.eigen();
        cell.project(rhs);
        stressNorm = rhs.eigen().matrix().norm();
        if (convergence_test()) {
          break;
        }
        incrF.eigen() = 0;

        incrF.eigenvec() = solver.solve(rhs.eigenvec(), incrF.eigenvec());


        F.eigen() += incrF.eigen();

        incrNorm = incrF.eigen().matrix().norm();
        gradNorm = F.eigen().matrix().norm();
        if (verbose > 0) {
          std::cout << "at Newton step " << std::setw(count_width) << newt_iter
                    << ", |δ" << strain_symb << "|/|Δ" << strain_symb
                    << "| = " << std::setw(17) << incrNorm/gradNorm
                    << ", tol = " << newton_tol << std::endl;

          if (verbose-1>1) {
            std::cout << "<" << strain_symb << "> =" << std::endl
                      << F.get_map().mean() << std::endl;
          }
        }
        convergence_test();

      }
      // update previous gradient
      previous_grad = delF;

      ret_val.push_back(OptimizeResult{F.eigen(), cell.get_stress().eigen(),
            convergence_test(), Int(convergence_test()),
            message,
            newt_iter, solver.get_counter()});

      //store history variables for next load increment
      cell.save_history_variables();

    }

    return ret_val;

  }

  //! instantiation for two-dimensional cells
  template std::vector<OptimizeResult>
  newton_cg (CellBase<twoD, twoD> & cell, const GradIncrements<twoD>& delF0,
             SolverBase<twoD, twoD> & solver, Real newton_tol,
             Real equil_tol,
             Dim_t verbose);

  // template typename CellBase<twoD, threeD>::StrainField_t &
  // newton_cg (CellBase<twoD, threeD> & cell, const GradIncrements<threeD>& delF0,
  //            const Real cg_tol, const Real newton_tol, Uint maxiter,
  //            Dim_t verbose);

  //! instantiation for three-dimensional cells
  template std::vector<OptimizeResult>
  newton_cg (CellBase<threeD, threeD> & cell, const GradIncrements<threeD>& delF0,
             SolverBase<threeD, threeD> & solver, Real newton_tol,
             Real equil_tol,
             Dim_t verbose);


  /* ---------------------------------------------------------------------- */
  bool check_symmetry(const Eigen::Ref<const Eigen::ArrayXXd>& eps,
                      Real rel_tol){
    return (rel_tol >= (eps-eps.transpose()).matrix().norm()/eps.matrix().norm() ||
            rel_tol >= eps.matrix().norm());
  }


}  // muSpectre
