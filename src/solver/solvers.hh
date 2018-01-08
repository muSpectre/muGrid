/**
 * file   solvers.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2017
 *
 * @brief  Free functions for solving
 *
 * @section LICENCE
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

#ifndef SOLVERS_H
#define SOLVERS_H

#include "solver/solver_cg.hh"

namespace muSpectre {

  template <Dim_t Dim>
  using Grad_t = Matrices::Tens2_t<Dim>;
  template <Dim_t Dim>
  using GradIncrements = std::vector<Grad_t<Dim>,
                                     Eigen::aligned_allocator<Grad_t<Dim>>>;

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM=DimS>
  typename SystemBase<DimS, DimM>::StrainField_t &
  newton_cg (SystemBase<DimS, DimM> & sys,
             const GradIncrements<DimM> & delF0,
             Formulation form,
             const Real cg_tol, const Real newton_tol, const Uint maxiter=0,
             Dim_t verbose = 0);

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM=DimS>
  inline typename SystemBase<DimS, DimM>::StrainField_t &
  newton_cg (SystemBase<DimS, DimM> & sys, const Grad_t<DimM> & delF0,
             Formulation form,
             const Real cg_tol, const Real newton_tol, const Uint maxiter=0,
             Dim_t verbose = 0){
    return newton_cg(sys, GradIncrements<DimM>{delF0}, form,
                    cg_tol, newton_tol, maxiter, verbose);
  }

    /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM=DimS>
  typename SystemBase<DimS, DimM>::StrainField_t &
  de_geus (SystemBase<DimS, DimM> & sys,
           const GradIncrements<DimM> & delF0,
           Formulation form,
           const Real cg_tol, const Real newton_tol, const Uint maxiter=0,
           Dim_t verbose = 0);

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM=DimS>
  inline typename SystemBase<DimS, DimM>::StrainField_t &
  de_geus (SystemBase<DimS, DimM> & sys, const Grad_t<DimM> & delF0,
           Formulation form,
           const Real cg_tol, const Real newton_tol, const Uint maxiter=0,
           Dim_t verbose = 0){
    return de_geus(sys, GradIncrements<DimM>{delF0}, form,
                   cg_tol, newton_tol, maxiter, verbose);
  }

}  // muSpectre

#endif /* SOLVERS_H */
