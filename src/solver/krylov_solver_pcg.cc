/**
 * @file   krylov_solver_pcg.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   28 Aug 2020
 *
 * @brief  Implementation for preconditioned conjugate gradient solver
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

#include "krylov_solver_pcg.hh"
#include "matrix_adaptor.hh"

#include <iomanip>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  KrylovSolverPCG::KrylovSolverPCG(
      std::shared_ptr<MatrixAdaptable> matrix_holder,
      std::shared_ptr<MatrixAdaptable> inv_preconditioner, const Real & tol,
      const Uint & maxiter, const Verbosity & verbose)
      : Parent{matrix_holder, tol, maxiter, verbose},
        FeaturesPC{inv_preconditioner}, r_k(this->get_nb_dof()),
        y_k(this->get_nb_dof()), p_k(this->get_nb_dof()),
        Ap_k(this->get_nb_dof()), x_k(this->get_nb_dof()) {}

  /* ---------------------------------------------------------------------- */
  KrylovSolverPCG::KrylovSolverPCG(const Real & tol, const Uint & maxiter,
                                   const Verbosity & verbose)
      : Parent{tol, maxiter, verbose}, r_k{}, y_k{}, p_k{}, Ap_k{}, x_k{} {}

  /* ---------------------------------------------------------------------- */
  auto KrylovSolverPCG::solve(const ConstVector_ref rhs) -> Vector_map {
    // Following implementation of algorithm 5.3 in Nocedal's
    // Numerical Optimization (p. 119)

    this->x_k.setZero();
    /* initialisation:
     *           Set r₀ ← Ax₀-b
     *           Set y₀ ← M⁻¹r₀
     *           Set p₀ ← -y₀, k ← 0 */
    this->r_k = this->matrix * this->x_k - rhs;
    this->y_k = this->preconditioner * this->r_k;
    this->p_k = -this->y_k;

    Real rdr{this->squared_norm(this->r_k)};
    Real rdy{this->dot(this->r_k, this->y_k)};
    Real rhs_norm2{this->squared_norm(rhs)};

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
      std::cout << "Norm of rhs in preconditioned CG = " << rhs_norm2
                << std::endl;
    }

    // Multiplication with the norm of the right hand side to get a relative
    // convergence criterion
    Real rel_tol2 = muGrid::ipow(this->tol, 2) * rhs_norm2;

    size_t count_width{};  // for output formatting in verbose case
    if (this->verbose > Verbosity::Silent) {
      count_width = size_t(std::log10(this->maxiter)) + 1;
    }

    // because of the early termination criterion, we never count the last
    // iteration
    ++this->counter;
    for (Uint i = 0; i < this->maxiter; ++i, ++this->counter) {
      this->Ap_k = matrix * this->p_k;
      Real pAp{this->dot(this->p_k, this->Ap_k)};

      if (pAp <= 0) {
        // Hessian is not positive definite
        throw SolverError("Hessian is not positive definite");
      }

      /*                    rᵀₖyₖ
       *             αₖ ← ————––
       *                    pᵀₖApₖ                                  */
      Real alpha{rdy / pAp};

      //             xₖ₊₁ ← xₖ + αₖpₖ
      this->x_k += alpha * this->p_k;

      //             rₖ₊₁ ← rₖ + αₖApₖ
      this->r_k += alpha * this->Ap_k;
      rdr = this->squared_norm(this->r_k);
      if (this->verbose > Verbosity::Silent && this->comm.rank() == 0) {
        std::cout << "  at CG step " << std::setw(count_width) << i
                  << ": |r|/|b| = " << std::setw(15)
                  << std::sqrt(rdr / rhs_norm2) << ", cg_tol = " << this->tol
                  << std::endl;
      }
      if (rdr < rel_tol2) {
        break;
      }

      //             yₖ₊₁ ← M⁻¹rₖ₊₁
      this->y_k = this->preconditioner * this->r_k;
      /*                     rᵀₖ₊₁yₖ₊₁
       *             βₖ₊₁ ← ————————–
       *                      rᵀₖyₖ                                */
      Real new_rdy{this->dot(this->r_k, this->y_k)};
      Real beta{new_rdy / rdy};
      rdy = new_rdy;

      //             pₖ₊₁ ← -yₖ₊₁ + βₖ₊₁pₖ
      this->p_k = -this->y_k + beta * this->p_k;
    }

    if (rdr < rel_tol2) {
      this->convergence = Convergence::ReachedTolerance;
    } else {
      std::stringstream err{};
      err << " After " << this->counter << " steps, the solver "
          << " FAILED with  |r|/|b| = " << std::setw(15)
          << std::sqrt(rdr / rhs_norm2) << ", cg_tol = " << this->tol
          << std::endl;
      throw ConvergenceError("Conjugate gradient has not converged." +
                             err.str());
    }
    return Vector_map(this->x_k.data(), this->x_k.size());
  }

  /* ---------------------------------------------------------------------- */
  std::string KrylovSolverPCG::get_name() const { return "PCG"; }

  /* ---------------------------------------------------------------------- */
  void KrylovSolverPCG::set_matrix(
      std::shared_ptr<MatrixAdaptable> matrix_adaptable) {
    Parent::set_matrix(matrix_adaptable);
    this->set_internal_arrays();
  }

  /* ---------------------------------------------------------------------- */
  void
  KrylovSolverPCG::set_matrix(std::weak_ptr<MatrixAdaptable> matrix_adaptable) {
    Parent::set_matrix(matrix_adaptable);
    this->set_internal_arrays();
  }

  /* ---------------------------------------------------------------------- */
  void KrylovSolverPCG::set_internal_arrays() {
    this->comm = this->matrix_ptr.lock()->get_communicator();
    auto && nb_dof{this->matrix_ptr.lock()->get_nb_dof()};
    this->r_k.resize(nb_dof);
    this->p_k.resize(nb_dof);
    this->Ap_k.resize(nb_dof);
    this->x_k.resize(nb_dof);
  }

  /* ---------------------------------------------------------------------- */
  void KrylovSolverPCG::set_preconditioner(
      std::shared_ptr<MatrixAdaptable> inv_preconditioner) {
    FeaturesPC::set_preconditioner(inv_preconditioner);
  }

}  // namespace muSpectre
