/**
 * file   solver_cg.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2017
 *
 * @brief  Implementation of cg solver
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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

#include "solver_cg.hh"


namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  SolverCG<DimS, DimM>::SolverCG(Ccoord resolutions, Real tol, Uint maxiter,
                                 bool verbose)
    :Parent(resolutions),
     r_k{make_field<Field_t>("residual r_k", this->collection)},
     p_k{make_field<Field_t>("search direction r_k", this->collection)},
     Ap_k{make_field<Field_t>("Effect of tangent A*p_k", this->collection)},
     tol{tol}, maxiter{maxiter}, verbose{verbose}
  {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void SolverCG<DimS, DimM>::solve(Fun_t& tangent_effect, const Field_t & rhs,
                              Field_t & x_f) {
    // Following implementation of algorithm 5.2 in Nocedal's Numerical Optimization (p. 112)

    auto r = this->r_k.eigen();
    auto p = this->p_k.eigen();
    auto Ap = this->Ap_k.eigen();
    auto x = x_f.eigen();

    // initialisation of algo
    tangent_effect(this->r_k, x_f);
    r = rhs.eigen();
    p = -r;

    this->converged = false;
    Real rdr = (r*r).sum();

    for (Uint i = 0; i < this->maxiter && rdr > this->tol; ++i) {
      tangent_effect(this->Ap_k, this->p_k);

      Real alpha = rdr/(r*Ap).sum();

      x += alpha * p;
      r += alpha * Ap;

      Real new_rdr = (r*r).sum();
      Real beta = new_rdr/rdr;
      rdr = new_rdr;

      if (this->verbose) {
        std::cout << "at CG step " << i << ": |r| = " << std::sqrt(rdr)
                  << ", cg_tol = " << this->tol << std::endl;
      }

      p = -r+beta*p;
    }
    if (rdr < this->tol) {
      this->converged=true;
    }
  }


  template class SolverCG<twoD, twoD>;
  template class SolverCG<twoD, threeD>;
  template class SolverCG<threeD, threeD>;
}  // muSpectre