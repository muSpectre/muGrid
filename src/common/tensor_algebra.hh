/**
 * @file   tensor_algebra.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   05 Nov 2017
 *
 * @brief  collection of compile-time quantities and algrebraic functions for
 *         tensor operations
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
 * General Public License for more details.
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
 */

#ifndef SRC_COMMON_TENSOR_ALGEBRA_HH_
#define SRC_COMMON_TENSOR_ALGEBRA_HH_

#include "common/T4_map_proxy.hh"
#include "common/common.hh"
#include "common/eigen_tools.hh"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <type_traits>

namespace muSpectre {

  namespace Tensors {

    //! second-order tensor representation
    template <Dim_t dim>
    using Tens2_t = Eigen::TensorFixedSize<Real, Eigen::Sizes<dim, dim>>;
    //! fourth-order tensor representation
    template <Dim_t dim>
    using Tens4_t =
        Eigen::TensorFixedSize<Real, Eigen::Sizes<dim, dim, dim, dim>>;

    //----------------------------------------------------------------------------//
    //! compile-time second-order identity
    template <Dim_t dim> constexpr inline Tens2_t<dim> I2() {
      Tens2_t<dim> T;
      using Mat_t = Eigen::Matrix<Real, dim, dim>;
      Eigen::Map<Mat_t>(&T(0, 0)) = Mat_t::Identity();
      return T;
    }

    /* ---------------------------------------------------------------------- */
    //! Check whether a given expression represents a Tensor specified order
    template <class T, Dim_t order> struct is_tensor {
      //! evaluated test
      constexpr static bool value =
          (std::is_convertible<T, Eigen::Tensor<Real, order>>::value ||
           std::is_convertible<T, Eigen::Tensor<Int, order>>::value ||
           std::is_convertible<T, Eigen::Tensor<Complex, order>>::value);
    };
    /* ---------------------------------------------------------------------- */
    /** compile-time outer tensor product as defined by Curnier
     *  R_ijkl = A_ij.B_klxx
     *    0123     01   23
     */
    template <Dim_t dim, typename T1, typename T2>
    constexpr inline decltype(auto) outer(T1 &&A, T2 &&B) {
      // Just make sure that the right type of parameters have been given
      constexpr Dim_t order{2};
      static_assert(is_tensor<T1, order>::value,
                    "T1 needs to be convertible to a second order Tensor");
      static_assert(is_tensor<T2, order>::value,
                    "T2 needs to be convertible to a second order Tensor");

      // actual function
      std::array<Eigen::IndexPair<Dim_t>, 0> dims{};
      return A.contract(B, dims);
    }

    /* ---------------------------------------------------------------------- */
    /** compile-time underlined outer tensor product as defined by Curnier
     *  R_ijkl = A_ik.B_jlxx
     *    0123     02   13
     *    0213     01   23 <- this defines the shuffle order
     */
    template <Dim_t dim, typename T1, typename T2>
    constexpr inline decltype(auto) outer_under(T1 &&A, T2 &&B) {
      constexpr size_t order{4};
      return outer<dim>(A, B).shuffle(std::array<Dim_t, order>{{0, 2, 1, 3}});
    }

    /* ---------------------------------------------------------------------- */
    /** compile-time overlined outer tensor product as defined by Curnier
     *  R_ijkl = A_il.B_jkxx
     *    0123     03   12
     *    0231     01   23 <- this defines the shuffle order
     */
    template <Dim_t dim, typename T1, typename T2>
    constexpr inline decltype(auto) outer_over(T1 &&A, T2 &&B) {
      constexpr size_t order{4};
      return outer<dim>(A, B).shuffle(std::array<Dim_t, order>{{0, 2, 3, 1}});
    }

    //! compile-time fourth-order symmetrising identity
    template <Dim_t dim> constexpr inline Tens4_t<dim> I4S() {
      auto I = I2<dim>();
      return 0.5 * (outer_under<dim>(I, I) + outer_over<dim>(I, I));
    }

  }  // namespace Tensors

  namespace Matrices {

    //! second-order tensor representation
    template <Dim_t dim> using Tens2_t = Eigen::Matrix<Real, dim, dim>;
    //! fourth-order tensor representation
    template <Dim_t dim> using Tens4_t = T4Mat<Real, dim>;

    //----------------------------------------------------------------------------//
    //! compile-time second-order identity
    template <Dim_t dim> constexpr inline Tens2_t<dim> I2() {
      return Tens2_t<dim>::Identity();
    }

    /* ---------------------------------------------------------------------- */
    /** compile-time outer tensor product as defined by Curnier
     *  R_ijkl = A_ij.B_klxx
     *    0123     01   23
     */
    template <typename T1, typename T2>
    constexpr inline decltype(auto) outer(T1 &&A, T2 &&B) {
      // Just make sure that the right type of parameters have been given
      constexpr Dim_t dim{EigenCheck::tensor_dim<T1>::value};
      static_assert((dim == EigenCheck::tensor_dim<T2>::value),
                    "A and B do not have the same dimension");

      Tens4_t<dim> product;

      for (Dim_t i = 0; i < dim; ++i) {
        for (Dim_t j = 0; j < dim; ++j) {
          for (Dim_t k = 0; k < dim; ++k) {
            for (Dim_t l = 0; l < dim; ++l) {
              get(product, i, j, k, l) = A(i, j) * B(k, l);
            }
          }
        }
      }
      return product;
    }

    /* ---------------------------------------------------------------------- */
    /** compile-time underlined outer tensor product as defined by Curnier
     *  R_ijkl = A_ik.B_jlxx
     *    0123     02   13
     *    0213     01   23 <- this defines the shuffle order
     */
    template <typename T1, typename T2>
    constexpr inline decltype(auto) outer_under(T1 &&A, T2 &&B) {
      // Just make sure that the right type of parameters have been given
      constexpr Dim_t dim{EigenCheck::tensor_dim<T1>::value};
      static_assert((dim == EigenCheck::tensor_dim<T2>::value),
                    "A and B do not have the same dimension");

      Tens4_t<dim> product;

      for (Dim_t i = 0; i < dim; ++i) {
        for (Dim_t j = 0; j < dim; ++j) {
          for (Dim_t k = 0; k < dim; ++k) {
            for (Dim_t l = 0; l < dim; ++l) {
              get(product, i, j, k, l) = A(i, k) * B(j, l);
            }
          }
        }
      }
      return product;
    }

    /* ---------------------------------------------------------------------- */
    /** compile-time overlined outer tensor product as defined by Curnier
     *  R_ijkl = A_il.B_jkxx
     *    0123     03   12
     *    0231     01   23 <- this defines the shuffle order
     */
    template <typename T1, typename T2>
    constexpr inline decltype(auto) outer_over(T1 &&A, T2 &&B) {
      // Just make sure that the right type of parameters have been given
      constexpr Dim_t dim{EigenCheck::tensor_dim<T1>::value};
      static_assert((dim == EigenCheck::tensor_dim<T2>::value),
                    "A and B do not have the same dimension");

      Tens4_t<dim> product;

      for (Dim_t i = 0; i < dim; ++i) {
        for (Dim_t j = 0; j < dim; ++j) {
          for (Dim_t k = 0; k < dim; ++k) {
            for (Dim_t l = 0; l < dim; ++l) {
              get(product, i, j, k, l) = A(i, l) * B(j, k);
            }
          }
        }
      }
      return product;
    }

    /**
     * Standart tensor multiplication
     */
    template <typename T4, typename T2>
    constexpr inline decltype(auto) tensmult(const Eigen::MatrixBase<T4> &A,
                                             const Eigen::MatrixBase<T2> &B) {
      constexpr Dim_t dim{T2::RowsAtCompileTime};
      static_assert(dim == T2::ColsAtCompileTime, "B is not square");
      static_assert(dim != Eigen::Dynamic, "B not statically sized");
      static_assert(dim * dim == T4::RowsAtCompileTime,
                    "A and B not compatible");
      static_assert(T4::RowsAtCompileTime == T4::ColsAtCompileTime,
                    "A is not square");
      Tens2_t<dim> result;
      result.setZero();

      for (Dim_t i = 0; i < dim; ++i) {
        for (Dim_t j = 0; j < dim; ++j) {
          for (Dim_t k = 0; k < dim; ++k) {
            for (Dim_t l = 0; l < dim; ++l) {
              result(i, j) += get(A, i, j, k, l) * B(k, l);
            }
          }
        }
      }
      return result;
    }

    //! compile-time fourth-order tracer
    template <Dim_t dim> constexpr inline Tens4_t<dim> Itrac() {
      auto I = I2<dim>();
      return outer(I, I);
    }

    //! compile-time fourth-order identity
    template <Dim_t dim> constexpr inline Tens4_t<dim> Iiden() {
      auto I = I2<dim>();
      return outer_under(I, I);
    }

    //! compile-time fourth-order transposer
    template <Dim_t dim> constexpr inline Tens4_t<dim> Itrns() {
      auto I = I2<dim>();
      return outer_over(I, I);
    }

    //! compile-time fourth-order symmetriser
    template <Dim_t dim> constexpr inline Tens4_t<dim> Isymm() {
      auto I = I2<dim>();
      return 0.5 * (outer_under(I, I) + outer_over(I, I));
    }

  }  // namespace Matrices
}  // namespace muSpectre

#endif  // SRC_COMMON_TENSOR_ALGEBRA_HH_
