/**
 * @file   krylov_solver_cg.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Apr 2018
 *
 * @brief  implements KrylovSolverCG
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

#include "solver/krylov_solver_cg.hh"
#include "cell/cell_adaptor.hh"
#include <libmugrid/communicator.hh>

#include <iomanip>
#include <sstream>
#include <iostream>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  KrylovSolverCG::KrylovSolverCG(std::shared_ptr<MatrixAdaptable> matrix_holder,
                                 const Real & tol, const Uint & maxiter,
                                 const Verbosity & verbose)
      : Parent{matrix_holder, tol, maxiter, verbose}, r_k(this->get_nb_dof()),
        p_k(this->get_nb_dof()), Ap_k(this->get_nb_dof()),
        x_k(this->get_nb_dof()) {}

  /* ---------------------------------------------------------------------- */
  KrylovSolverCG::KrylovSolverCG(const Real & tol, const Uint & maxiter,
                                 const Verbosity & verbose)
      : Parent{tol, maxiter, verbose}, r_k{}, p_k{}, Ap_k{}, x_k{} {}

  /* ---------------------------------------------------------------------- */
  void KrylovSolverCG::set_matrix(
      std::shared_ptr<MatrixAdaptable> matrix_adaptable) {
    Parent::set_matrix(matrix_adaptable);
    this->set_internal_arrays();
  }

  /* ---------------------------------------------------------------------- */
  void
  KrylovSolverCG::set_matrix(std::weak_ptr<MatrixAdaptable> matrix_adaptable) {
    Parent::set_matrix(matrix_adaptable);
    this->set_internal_arrays();
  }

  /* ---------------------------------------------------------------------- */
  void KrylovSolverCG::set_internal_arrays() {
    this->comm = this->matrix_ptr.lock()->get_communicator();
    auto && nb_dof{this->matrix_ptr.lock()->get_nb_dof()};
    this->r_k.resize(nb_dof);
    this->p_k.resize(nb_dof);
    this->Ap_k.resize(nb_dof);
    this->x_k.resize(nb_dof);
  }

  /* ---------------------------------------------------------------------- */
  std::string KrylovSolverCG::get_name() const { return "CG"; }

  /* ---------------------------------------------------------------------- */
  auto KrylovSolverCG::solve(const ConstVector_ref rhs) -> Vector_map {
    if (this->matrix_ptr.expired()) {
      std::stringstream error_message{};
      error_message << "The system matrix has been destroyed. Did you set the "
                       "matrix using a weak_ptr instead of a shared_ptr?";
      throw SolverError{error_message.str()};
    }
    this->x_k.setZero();
    // Following implementation of algorithm 5.2 in Nocedal's
    // Numerical Optimization (p. 112)

    // initialisation of algorithm
    /* initialisation:
     *           Set r₀ ← Ax₀-b
     *           Set p₀ ← -r₀, k ← 0 */
    this->r_k = this->matrix * this->x_k - rhs;
    this->p_k = -this->r_k;
    this->convergence = Convergence::DidNotConverge;

    Real rdr = this->squared_norm(r_k);
    Real rhs_norm2 = this->squared_norm(rhs);

    if (rhs_norm2 == 0) {
      std::stringstream msg{};
      msg << "You are invoking conjugate gradient"
          << "solver with absolute zero RHS.\n"
          << "Please check the load steps of your problem "
          << "to ensure nothing is missed.\n"
          << "You might need to set equilibrium tolerance to a positive "
          << "small value to avoid calling the conjugate gradient solver in "
          << "case of having zero RHS (relatively small RHS).\n"
          << std::endl;
      std::cout << "WARNING: ";
      std::cout << msg.str();
      this->convergence = Convergence::ReachedTolerance;
      return Vector_map(this->x_k.data(), this->x_k.size());
    }

    if (this->verbose > Verbosity::Silent && this->comm.rank() == 0) {
      std::cout << "Norm of rhs in CG = " << std::sqrt(rhs_norm2) << std::endl;
    }

    // Multiplication with the norm of the right hand side to get a relative
    // convergence criterion
    Real tol = this->tol;

    // Negative tolerance tells the solver to automatically adjust it
    if (tol < 0.0) {
      // See Nocedal, page 169
      tol = std::min(0.5, sqrt(sqrt(rhs_norm2)));
    }
    Real rel_tol2 = muGrid::ipow(tol, 2) * rhs_norm2;

    size_t count_width{};  // for output formatting in verbose case
    if (this->verbose > Verbosity::Silent) {
      count_width = size_t(std::log10(this->maxiter)) + 1;
    }

    // because of the early termination criterion, we never count the last
    // iteration
    ++this->counter;
    Uint current_counter{0};
    for (; current_counter < this->maxiter && rdr > rel_tol2;
         ++current_counter, ++this->counter) {
      this->Ap_k = this->matrix * this->p_k;

      Real pdAp{this->dot(this->p_k, this->Ap_k)};
      if (pdAp <= 0) {
        // Hessian is not positive definite
        throw SolverError("Hessian is not positive definite");
      }

      /*                    rᵀₖrₖ
       *             αₖ ← ————––
       *                    pᵀₖApₖ                                  */
      Real alpha{rdr / pdAp};

      //             xₖ₊₁ ← xₖ + αₖpₖ
      this->x_k += alpha * this->p_k;

      //             rₖ₊₁ ← rₖ + αₖApₖ
      this->r_k += alpha * this->Ap_k;

      Real new_rdr{this->squared_norm(this->r_k)};

      /*                      rᵀₖ₊₁rₖ₊₁
       *             βₖ₊₁ ← ————————–
       *                      rᵀₖyₖ                                */
      Real beta{new_rdr / rdr};
      rdr = new_rdr;

      if (this->verbose > Verbosity::Silent && this->comm.rank() == 0) {
        std::cout << "  at CG step " << std::setw(count_width)
                  << current_counter << ": |r|/|b| = " << std::setw(15)
                  << std::sqrt(rdr / rhs_norm2) << ", cg_tol = " << this->tol
                  << std::endl;
      }
      if (new_rdr < rel_tol2) {
        break;
      }

      //             pₖ₊₁ ← -rₖ₊₁ + βₖ₊₁pₖ
      this->p_k = -this->r_k + beta * this->p_k;
    }

    if (rdr < rel_tol2) {
      this->convergence = Convergence::ReachedTolerance;
    } else {
      std::stringstream err{};
      err << " After " << current_counter << " steps, the solver "
          << " FAILED with  |r|/|b| = " << std::setw(15)
          << std::sqrt(rdr / rhs_norm2) << ", cg_tol = " << this->tol
          << std::endl;
      throw ConvergenceError("Conjugate gradient has not converged." +
                             err.str());
    }
    return Vector_map(this->x_k.data(), this->x_k.size());
  }

}  // namespace muSpectre
