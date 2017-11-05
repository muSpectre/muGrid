/**
 * file   tensor_algebra.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   05 Nov 2017
 *
 * @brief  collection of compile-time quantities and algrebraic functions for
 *         tensor operations
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

#ifndef TENSOR_ALGEBRA_H
#define TENSOR_ALGEBRA_H

#include "common/common.hh"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


namespace muSpectre {

  namespace Tensors {

    template<Dim_t dim>
    using Tens2_t = Eigen::TensorFixedSize<Real, Eigen::Sizes<dim, dim>>;
    template<Dim_t dim>
    using Tens4_t = Eigen::TensorFixedSize<Real, Eigen::Sizes<dim, dim, dim, dim>>;

    //----------------------------------------------------------------------------//
    //! compile-time second-order identity
    template<Dim_t dim>
    constexpr inline Tens2_t<dim> I2() {
      Tens2_t<dim> T;
      using Mat_t = Eigen::Matrix<Real, dim, dim>;
      Eigen::Map<Mat_t>(&T(0, 0)) = Mat_t::Identity();
      return T;
    }

    /* ---------------------------------------------------------------------- */
    /** compile-time outer tensor product as defined by Curnier
     *  R_ijkl = A_ij.B_klxx
     *    0123     01   23
     */
    template<Dim_t dim, typename T1, typename T2>
    constexpr inline decltype(auto) outer(T1 && A, T2 && B) {
      std::array<Eigen::IndexPair<Dim_t>, 0> dims{};
      return A.contract(B, dims);
    }

    /* ---------------------------------------------------------------------- */
    /** compile-time underlined outer tensor product as defined by Curnier
     *  R_ijkl = A_ik.B_jlxx
     *    0123     02   13
     */
    template<Dim_t dim, typename T1, typename T2>
    constexpr inline decltype(auto) outer_under(T1 && A, T2 && B) {
      constexpr size_t order{4};
      return outer<dim>(A, B).shuffle(std::array<Dim_t, order>{0, 2, 1, 3});
    }

    /* ---------------------------------------------------------------------- */
    /** compile-time overlined outer tensor product as defined by Curnier
     *  R_ijkl = A_il.B_jkxx
     *    0123     03   12
     */
    template<Dim_t dim, typename T1, typename T2>
    constexpr inline decltype(auto) outer_over(T1 && A, T2 && B) {
      constexpr size_t order{4};
      return outer<dim>(A, B).shuffle(std::array<Dim_t, order>{0, 3, 1, 2});
    }


}  // muSpectre


#endif /* TENSOR_ALGEBRA_H */
