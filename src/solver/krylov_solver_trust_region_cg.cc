/**
 * @file   krylov_solver_trust_region_cg.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *         Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 July 2020
 *
 * @brief  Conjugate-gradient solver with a trust region
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

#include <libmugrid/communicator.hh>

#include "cell/cell_adaptor.hh"

#include "solver/krylov_solver_trust_region_cg.hh"

#include <iomanip>
#include <sstream>
#include <iostream>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  KrylovSolverTrustRegionCG::KrylovSolverTrustRegionCG(
      std::shared_ptr<MatrixAdaptable> matrix_holder, const Real & tol,
      const Uint & maxiter, const Real & trust_region,
      const Verbosity & verbose, const ResetCG & reset,
      const Index_t & reset_iter_count)
      : Parent{matrix_holder, tol, maxiter, verbose},
        FeaturesTR{trust_region, reset,
                   reset_iter_count == muGrid::Unknown ? this->get_nb_dof() / 4
                                                       : reset_iter_count},
        r_k(this->get_nb_dof()), p_k(this->get_nb_dof()),
        Ap_k(this->get_nb_dof()), x_k(this->get_nb_dof()) {
    if (this->reset == ResetCG::iter_count and this->reset_iter_count <= 0) {
      throw SolverError(
          "Positive valued reset_iter_count is needed to perform user "
          "defined iteration count restart for the CG solver");
    }
  }

  /* ---------------------------------------------------------------------- */
  KrylovSolverTrustRegionCG::KrylovSolverTrustRegionCG(
      const Real & tol, const Uint & maxiter, const Real & trust_region,
      const Verbosity & verbose, const ResetCG & reset,
      const Index_t & reset_iter_count)
      : Parent{tol, maxiter, verbose}, FeaturesTR{trust_region, reset,
                                                  reset_iter_count},
        r_k{}, p_k{}, Ap_k{}, x_k{} {}

  /* ---------------------------------------------------------------------- */
  void KrylovSolverTrustRegionCG::set_matrix(
      std::shared_ptr<MatrixAdaptable> matrix_adaptable) {
    Parent::set_matrix(matrix_adaptable);
    this->set_internal_arrays();
  }

  /* ---------------------------------------------------------------------- */
  void KrylovSolverTrustRegionCG::set_matrix(
      std::weak_ptr<MatrixAdaptable> matrix_adaptable) {
    Parent::set_matrix(matrix_adaptable);
    this->set_internal_arrays();
  }

  /* ---------------------------------------------------------------------- */
  void KrylovSolverTrustRegionCG::set_internal_arrays() {
    this->comm = this->matrix_ptr.lock()->get_communicator();
    auto && nb_dof{this->matrix_ptr.lock()->get_nb_dof()};
    this->r_k.resize(nb_dof);
    this->p_k.resize(nb_dof);
    this->Ap_k.resize(nb_dof);
    this->x_k.resize(nb_dof);
    if (this->reset_iter_count == muGrid::Unknown) {
      this->reset_iter_count = nb_dof / 4;
    }
    if (this->reset == ResetCG::iter_count and this->reset_iter_count <= 0) {
      throw SolverError(
          "Positive valued reset_iter_count is needed to perform user "
          "defined iteration count restart for the CG solver");
    }
  }

  /* ---------------------------------------------------------------------- */
  auto KrylovSolverTrustRegionCG::solve(const ConstVector_ref rhs)
      -> Vector_map {
    if (this->matrix_ptr.expired()) {
      std::stringstream error_message{};
      error_message << "The system matrix has been destroyed. Did you set the "
                       "matrix using a weak_ptr instead of a shared_ptr?";
      throw SolverError{error_message.str()};
    }
    this->is_on_bound = false;
    this->x_k.setZero();
    Real trust_region2{this->trust_region * this->trust_region};

    // Following implementation of Steihaug's CG, algorithm 7.2 in Nocedal's
    // Numerical Optimization (p. 171)

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
      std::cout << " , Norm of rhs in (trust region) Steihaug CG = "
                << rhs_norm2 << std::endl;
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
      count_width = static_cast<size_t>(std::log10(this->maxiter)) + 1;
    }
    Uint iter_counter{0};
    Uint current_counter{0};
    for (; current_counter < this->maxiter && rdr > rel_tol2;
         ++current_counter, ++this->counter) {
      this->Ap_k = this->matrix * this->p_k;

      Real pdAp{this->dot(this->p_k, this->Ap_k)};
      if (pdAp <= 0) {
        // Hessian is not positive definite, the minimizer is on the trust
        // region bound
        if (this->verbose > Verbosity::Silent && this->comm.rank() == 0) {
          std::cout << "  CG finished, reason: Hessian is not positive "
                       "definite (pdAp:"
                    << pdAp << ")" << std::endl;
        }
        ++this->counter;
        this->convergence = Convergence::HessianNotPositiveDefinite;
        return this->bound(rhs);
      }
      /*                    rᵀₖyₖ
       *             αₖ ← ————––
       *                    pᵀₖApₖ                                  */
      Real alpha{rdr / pdAp};

      //             xₖ₊₁ ← xₖ + αₖpₖ
      this->x_k += alpha * this->p_k;
      if (this->squared_norm(x_k) >= trust_region2) {
        // we are exceeding the trust region, the minimizer is on the trust
        // region bound
        if (this->verbose > Verbosity::Silent && this->comm.rank() == 0) {
          std::cout << "  CG finished, reason: step exceeded trust region "
                       "bounds"
                    << std::endl;
        }
        ++this->counter;
        this->convergence = Convergence::ExceededTrustRegionBound;
        return this->bound(rhs);
      }

      if (this->reset == ResetCG::gradient_orthogonality) {
        this->r_k_previous = this->r_k;
      }
      //             rₖ₊₁ ← rₖ + αₖApₖ
      this->r_k += alpha * this->Ap_k;

      Real new_rdr{this->squared_norm(this->r_k)};

      /*                     rᵀₖ₊₁rₖ₊₁
       *             βₖ₊₁ ← ————————–
       *                      rᵀₖrₖ                                */
      Real beta{new_rdr / rdr};

      //! CG reset worker
      auto && reset_cg{[&beta, this, &rhs, &iter_counter, &current_counter]() {
        this->r_k = this->matrix * this->x_k - rhs;
        beta = 0.0;
        iter_counter = 0;
        if (this->verbose > Verbosity::Silent && this->comm.rank() == 0) {
          std::cout << "Reset CG at step: " << current_counter << "\n";
        }
      }};

      if (current_counter > 1) {
        switch (this->reset) {
        case ResetCG::no_reset: {
          break;
        }
        case ResetCG::iter_count: {
          if (iter_counter++ > this->reset_iter_count) {
            reset_cg();
          }
          break;
        }
        case ResetCG::gradient_orthogonality: {
          /* Based on
            RESTART PROCEDURES FOR THE CONJUGATE GRADIENT METHOD by:
            M.J.D. POWELL
            Mathematical Programming 12 (1977) 241-254.
            North-Holland Publishing Company

            The threshold of 0.2 is a recommended heuristic there*/
          if (abs(this->dot(this->r_k, this->r_k_previous)) > 0.2 * new_rdr) {
            reset_cg();
          }
          break;
        }
        case ResetCG::valid_direction: {
          if (this->dot(this->r_k, this->p_k) > 0) {
            reset_cg();
          }
          break;
        }
        default: {
          throw SolverError("Unknown CG reset strategy ");
          break;
        }
        }
      }

      rdr = new_rdr;

      if (this->verbose > Verbosity::Silent && this->comm.rank() == 0) {
        std::cout << "  at CG step " << std::setw(count_width)
                  << current_counter << ": |r|/|b| = " << std::setw(15)
                  << std::sqrt(rdr / rhs_norm2) << ", cg_tol = " << tol
                  << std::endl;
      }

      //             pₖ₊₁ ← -rₖ₊₁ + βₖ₊₁pₖ
      this->p_k = -this->r_k + beta * this->p_k;
    }

    if (rdr < rel_tol2) {
      if (this->verbose > Verbosity::Silent && this->comm.rank() == 0) {
        std::cout << "  CG finished, reason: reached tolerance" << std::endl;
      }
      this->convergence = Convergence::ReachedTolerance;
    } else {
      std::stringstream err{};
      err << " After " << current_counter << " steps, the solver "
          << " FAILED with  |r|/|b| = " << std::setw(15)
          << std::sqrt(rdr / rhs_norm2) << ", cg_tol = " << tol << std::endl;
      throw ConvergenceError("Conjugate gradient has not converged." +
                             err.str());
    }
    return Vector_map(this->x_k.data(), this->x_k.size());
  }

  /* ---------------------------------------------------------------------- */
  auto KrylovSolverTrustRegionCG::bound(const ConstVector_ref rhs)
      -> Vector_map {
    this->is_on_bound = true;
    Real trust_region2{this->trust_region * this->trust_region};

    Real pdp{this->squared_norm(this->p_k)};
    Real xdx{this->squared_norm(this->x_k)};
    Real pdx{this->dot(this->p_k, this->x_k)};
    Real tmp{sqrt(pdx * pdx - pdp * (xdx - trust_region2))};
    Real tau1{-(pdx + tmp) / pdp};
    Real tau2{-(pdx - tmp) / pdp};

    this->x_k += tau1 * this->p_k;
    Real m1{this->dot(-rhs, this->x_k) +
            0.5 * this->dot(x_k, this->matrix * this->x_k)};
    this->x_k += (tau2 - tau1) * this->p_k;
    Real m2{this->dot(-rhs, this->x_k) +
            0.5 * this->dot(this->x_k, this->matrix * this->x_k)};

    // check which direction is the minimizer
    if (m2 < m1) {
      return Vector_map(this->x_k.data(), this->x_k.size());
    } else {
      this->x_k += (tau1 - tau2) * this->p_k;
      return Vector_map(this->x_k.data(), this->x_k.size());
    }
  }

  /* ---------------------------------------------------------------------- */
}  // namespace muSpectre
