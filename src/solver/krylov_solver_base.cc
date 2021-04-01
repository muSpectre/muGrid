/**
 * @file   krylov_solver_base.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   31 Jul 2020
 *
 * @brief  Implementation for KrylovSolverBase
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

#include "krylov_solver_base.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  KrylovSolverBase::KrylovSolverBase(
      std::shared_ptr<MatrixAdaptable> matrix_adaptable, const Real & tol,
      const Uint & maxiter, const Verbosity & verbose)
      : matrix_holder{matrix_adaptable},
        matrix_ptr{matrix_adaptable}, matrix{matrix_adaptable->get_adaptor()},
        tol{tol}, maxiter{maxiter}, verbose{verbose} {}

  /* ---------------------------------------------------------------------- */
  KrylovSolverBase::KrylovSolverBase(
      std::weak_ptr<MatrixAdaptable> matrix_adaptable, const Real & tol,
      const Uint & maxiter, const Verbosity & verbose)
      : matrix_ptr{matrix_adaptable},
        matrix{matrix_adaptable.lock()->get_adaptor()}, tol{tol},
        maxiter{maxiter}, verbose{verbose} {}

  /* ---------------------------------------------------------------------- */
  KrylovSolverBase::KrylovSolverBase(const Real & tol, const Uint & maxiter,
                                     const Verbosity & verbose)
      : tol{tol}, maxiter{maxiter}, verbose{verbose} {}

  /* ---------------------------------------------------------------------- */
  void KrylovSolverBase::set_matrix(
      std::shared_ptr<MatrixAdaptable> matrix_adaptable) {
    // just keeping a copy of the pointer keeps the matrix from destruction
    this->matrix_holder = matrix_adaptable;
    KrylovSolverBase::set_matrix(
        std::weak_ptr<MatrixAdaptable>{matrix_adaptable});
  }

  /* ---------------------------------------------------------------------- */
  void KrylovSolverBase::set_matrix(
      std::weak_ptr<MatrixAdaptable> matrix_adaptable) {
    this->matrix_ptr = matrix_adaptable;
    this->matrix = this->matrix_holder
                       ? this->matrix_holder->get_adaptor()
                       : this->matrix_ptr.lock()->get_weak_adaptor();
  }

  /* ---------------------------------------------------------------------- */
  KrylovSolverBase::Convergence KrylovSolverBase::get_convergence() const {
    return this->convergence;
  }

  /* ---------------------------------------------------------------------- */
  void KrylovSolverBase::reset_counter() {
    this->counter = 0;
    this->convergence = Convergence::DidNotConverge;
  }

  /* ---------------------------------------------------------------------- */
  Uint KrylovSolverBase::get_counter() const { return this->counter; }

  /* ---------------------------------------------------------------------- */
  Uint KrylovSolverBase::get_maxiter() const { return this->maxiter; }

  /* ---------------------------------------------------------------------- */
  Real KrylovSolverBase::get_tol() const { return this->tol; }

  /* ---------------------------------------------------------------------- */
  std::shared_ptr<MatrixAdaptable> KrylovSolverBase::get_matrix_holder() const {
    return this->matrix_holder;
  }

  /* ---------------------------------------------------------------------- */
  std::weak_ptr<MatrixAdaptable> KrylovSolverBase::get_matrix_ptr() const {
    return this->matrix_ptr;
  }

  /* ---------------------------------------------------------------------- */
  Index_t KrylovSolverBase::get_nb_dof() const {
    return this->matrix_ptr.lock()->get_nb_dof();
  }

  /* ---------------------------------------------------------------------- */
  const bool & KrylovSolverBase::get_is_on_bound() { return this->is_on_bound; }

}  // namespace muSpectre
