/**
 * @file   voigt_conversion.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   02 May 2017
 *
 * @brief  utilities to transform vector notation arrays into voigt notation
 *         arrays and vice-versa
 *
 * Copyright © 2017 Till Junge
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

#ifndef SRC_COMMON_VOIGT_CONVERSION_HH_
#define SRC_COMMON_VOIGT_CONVERSION_HH_

#include "common/muSpectre_common.hh"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>

namespace muSpectre {

  /**
   * implements a bunch of static functions to convert between full
   * and Voigt notation of tensors
   */
  //----------------------------------------------------------------------------//

  template <Dim_t dim>
  class VoigtConversion {
   public:
    VoigtConversion();

    //! obtain a fourth order voigt matrix from a tensor
    template <class Tens4, class Voigt, bool sym = true>
    inline static void fourth_to_voigt(const Tens4 & t, Voigt & v);

    //! return a fourth order voigt matrix from a tensor
    template <class Tens4, bool sym = true>
    inline static Eigen::Matrix<Real, vsize<sym>(dim), vsize<sym>(dim)>
    fourth_to_voigt(const Tens4 & t);

    //! return a fourth order non-symmetric voigt matrix from a tensor
    template <class Tens4>
    inline static Eigen::Matrix<Real, vsize<false>(dim), vsize<false>(dim)>
    fourth_to_2d(const Tens4 & t) {
      return fourth_to_voigt<Tens4, false>(t);
    }

    //! probably obsolete
    template <class Tens2, class Voigt, bool sym = true>
    inline static void second_to_voigt(const Tens2 & t, Voigt & v);

    //! probably obsolete
    template <class Tens2, class Voigt>
    inline static void gradient_to_voigt_strain(const Tens2 & F, Voigt & v);

    //! probably obsolete
    template <class Tens2, class Voigt>
    inline static void gradient_to_voigt_GreenLagrange_strain(const Tens2 & F,
                                                              Voigt & v);

    //! probably obsolete
    template <class Tens2, class Voigt, bool sym = true>
    inline static void stress_from_voigt(const Voigt & v, Tens2 & sigma);

   private:
    //! matrix of vector index I as function of tensor indices i,j
    static const Eigen::Matrix<Dim_t, dim, dim> mat;

    //! matrix of vector index I as function of tensor indices i,j
    static const Eigen::Matrix<Dim_t, dim, dim> sym_mat;
    //! array of matrix indices ij as function of vector index I
    static const Eigen::Matrix<Dim_t, dim * dim, 2> vec;
    //! factors to multiply the strain by for voigt notation
    static const Eigen::Matrix<Real, vsize(dim), 1> factors;
    /**
     * reordering between a row/column in voigt vs col-major matrix
     * (e.g., stiffness tensor)
     */
    static const Eigen::Matrix<Dim_t, dim * dim, 1> vec_vec;

   public:
    inline static auto get_mat() -> decltype(auto);
    inline static auto get_sym_mat() -> decltype(auto);
    inline static auto get_vec() -> decltype(auto);
    inline static auto get_factors() -> decltype(auto);
    inline static auto get_vec_vec() -> decltype(auto);
  };

  //! voigt vector indices for symmetric tensors
  template <>
  auto inline VoigtConversion<1>::get_sym_mat() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, 1 * 1, 1> sym_mat{
        (Eigen::Matrix<Dim_t, 1 * 1, 1>() << 0).finished()};
    return sym_mat;
  }

  template <>
  auto inline VoigtConversion<2>::get_sym_mat() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, 2, 2> sym_mat{
        (Eigen::Matrix<Dim_t, 2, 2>() << 0, 2, 2, 1).finished()};
    return sym_mat;
  }

  template <>
  auto inline VoigtConversion<3>::get_sym_mat() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, 3, 3> sym_mat{
        (Eigen::Matrix<Dim_t, 3, 3>() << 0, 5, 4, 5, 1, 3, 4, 3, 2).finished()};
    return sym_mat;
  }
  //----------------------------------------------------------------------------//

  //! voigt vector indices for non_symmetric tensors
  template <>
  auto inline VoigtConversion<1>::get_mat() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, 1 * 1, 1> mat{
        (Eigen::Matrix<Dim_t, 1 * 1, 1>() << 0).finished()};
    return mat;
  }

  template <>
  auto inline VoigtConversion<2>::get_mat() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, 2, 2> mat{
        (Eigen::Matrix<Dim_t, 2, 2>() << 0, 2, 3, 1).finished()};
    return mat;
  }

  template <>
  auto inline VoigtConversion<3>::get_mat() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, 3, 3> mat{
        (Eigen::Matrix<Dim_t, 3, 3>() << 0, 5, 4, 8, 1, 3, 7, 6, 2).finished()};
    return mat;
  }
  //----------------------------------------------------------------------------//
  //! matrix indices from voigt vectors
  template <>
  auto inline VoigtConversion<1>::get_vec() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, 1 * 1, 2> vec{
        (Eigen::Matrix<Dim_t, 1 * 1, 2>() << 0, 0).finished()};
    return vec;
  }

  template <>
  auto inline VoigtConversion<2>::get_vec() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, 2 * 2, 2> vec{
        (Eigen::Matrix<Dim_t, 2 * 2, 2>() << 0, 0, 1, 1, 0, 1, 1, 0)
            .finished()};
    return vec;
  }
  template <>
  auto inline VoigtConversion<3>::get_vec() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, 3 * 3, 2> vec{
        (Eigen::Matrix<Dim_t, 3 * 3, 2>() << 0, 0, 1, 1, 2, 2, 1, 2, 0, 2, 0, 1,
         2, 1, 2, 0, 1, 0)
            .finished()};
    return vec;
  }
  //----------------------------------------------------------------------------//
  template <>
  auto inline VoigtConversion<1>::get_factors() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, vsize(1), 1> factors{
        (Eigen::Matrix<Dim_t, vsize(1), 1>() << 1).finished()};
    return factors;
  }

  template <>
  auto inline VoigtConversion<2>::get_factors() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, vsize(2), 1> factors{
        (Eigen::Matrix<Dim_t, vsize(2), 1>() << 1, 1, 2).finished()};
    return factors;
  }

  template <>
  auto inline VoigtConversion<3>::get_factors() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, vsize(3), 1> factors{
        (Eigen::Matrix<Dim_t, vsize(3), 1>() << 1, 1, 1, 2, 2, 2).finished()};
    return factors;
  }
  //----------------------------------------------------------------------------//
  /**
   * reordering between a row/column in voigt vs col-major matrix
   * (e.g., stiffness tensor)
   */
  template <>
  auto inline VoigtConversion<1>::get_vec_vec() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, 1 * 1, 1> vec_vec{
        (Eigen::Matrix<Dim_t, 1 * 1, 1>() << 0).finished()};
    return vec_vec;
  }

  template <>
  auto inline VoigtConversion<2>::get_vec_vec() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, 2 * 2, 1> vec_vec{
        (Eigen::Matrix<Dim_t, 2 * 2, 1>() << 0, 3, 2, 1).finished()};
    return vec_vec;
  }

  template <>
  auto inline VoigtConversion<3>::get_vec_vec() -> decltype(auto) {
    const Eigen::Matrix<Dim_t, 3 * 3, 1> vec_vec{
        (Eigen::Matrix<Dim_t, 3 * 3, 1>() << 0, 8, 7, 5, 1, 6, 4, 3, 2)
            .finished()};
    return vec_vec;
  }

  //----------------------------------------------------------------------------//
  template <Dim_t dim>
  template <class Tens4, class Voigt, bool sym>
  inline void VoigtConversion<dim>::fourth_to_voigt(const Tens4 & t,
                                                    Voigt & v) {
    // upper case indices for Voigt notation, lower case for standard tensorial
    for (Dim_t I = 0; I < vsize<sym>(dim); ++I) {
      auto && i = vec(I, 0);
      auto && j = vec(I, 1);
      for (Dim_t J = 0; J < vsize<sym>(dim); ++J) {
        auto && k = vec(J, 0);
        auto && l = vec(J, 1);
        v(I, J) = t(i, j, k, l);
      }
    }
  }

  //----------------------------------------------------------------------------//
  template <Dim_t dim>
  template <class Tens4, bool sym>
  inline Eigen::Matrix<Real, vsize<sym>(dim), vsize<sym>(dim)>
  VoigtConversion<dim>::fourth_to_voigt(const Tens4 & t) {
    using V_t = Eigen::Matrix<Real, vsize<sym>(dim), vsize<sym>(dim)>;
    V_t temp;
    fourth_to_voigt<decltype(t), V_t, sym>(t, temp);
    return temp;
  }

  //----------------------------------------------------------------------------//
  template <Dim_t dim>
  template <class Tens2, class Voigt, bool sym>
  inline void VoigtConversion<dim>::second_to_voigt(const Tens2 & F,
                                                    Voigt & v) {
    for (Dim_t I = 0; I < vsize(dim); ++I) {
      auto && i = vec(I, 0);
      auto && j = vec(I, 1);
      v(I) = F(i, j);
    }
  }

  //----------------------------------------------------------------------------//
  template <Dim_t dim>
  template <class Tens2, class Voigt>
  inline void VoigtConversion<dim>::gradient_to_voigt_strain(const Tens2 & F,
                                                             Voigt & v) {
    for (Dim_t I = 0; I < vsize(dim); ++I) {
      auto && i = vec(I, 0);
      auto && j = vec(I, 1);
      v(I) = (F(i, j) + F(j, i)) / 2 * factors(I);
    }
  }

  //----------------------------------------------------------------------------//
  template <Dim_t dim>
  template <class Tens2, class Voigt>
  inline void
  VoigtConversion<dim>::gradient_to_voigt_GreenLagrange_strain(const Tens2 & F,
                                                               Voigt & v) {
    using mat = Eigen::Matrix<Real, dim, dim>;
    mat E = 0.5 * (F.transpose() * F - mat::Identity());
    for (Dim_t I = 0; I < vsize(dim); ++I) {
      auto && i = vec(I, 0);
      auto && j = vec(I, 1);
      v(I) = E(i, j) * factors(I);
    }
  }
}  // namespace muSpectre

#endif  // SRC_COMMON_VOIGT_CONVERSION_HH_
