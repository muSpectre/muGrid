/**
 * @file   util/eigen_checks.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Sep 2017
 *
 * @brief  Type traits for Eigen matrices
 *
 * Copyright © 2017 Till Junge
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
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

#ifndef SRC_LIBMUGRID_UTIL_EIGEN_CHECKS_HH_
#define SRC_LIBMUGRID_UTIL_EIGEN_CHECKS_HH_

#include "../core/types.hh"
#include "math.hh"

#include "Eigen/Dense"

#include <type_traits>

namespace muGrid {

  namespace EigenCheck {
    /**
     * Structure to determine whether an expression can be evaluated
     * into a `Eigen::Matrix`, `Eigen::Array`, etc. and which helps
     * determine compile-time size
     */
    template <class TestClass>
    struct is_matrix {
      using T = std::remove_cv_t<std::remove_reference_t<TestClass>>;
      constexpr static bool value{
          std::is_base_of<Eigen::MatrixBase<T>, T>::value};
    };

    template <class Derived>
    struct is_matrix<Eigen::Map<Derived>> {
      constexpr static bool value{is_matrix<Derived>::value};
    };

    template <class Derived>
    struct is_matrix<Eigen::MatrixBase<Eigen::Map<Derived>>> {
      constexpr static bool value{is_matrix<Derived>::value};
    };

    template <class Derived>
    struct is_matrix<Eigen::Ref<Derived>> {
      constexpr static bool value{is_matrix<Derived>::value};
    };

    /**
     * Helper class to check whether an `Eigen::Array` or
     * `Eigen::Matrix` is statically sized
     */
    template <class Derived>
    struct is_fixed {
      //! raw type for testing
      using T = std::remove_cv_t<std::remove_reference_t<Derived>>;
      //! evaluated test
      constexpr static bool value{T::SizeAtCompileTime != Eigen::Dynamic};
    };

    /**
     * Helper class to check whether an `Eigen::Array` or `Eigen::Matrix` is a
     * static-size and square.
     */
    template <class Derived>
    struct is_square {
      //! raw type for testing
      using T = std::remove_cv_t<std::remove_reference_t<Derived>>;
      //! true if the object is square and statically sized
      constexpr static bool value{
          (T::RowsAtCompileTime == T::ColsAtCompileTime) && is_fixed<T>::value};
    };

    /**
     * computes the dimension from a second order tensor represented
     * square matrix or array
     */
    template <class Derived>
    struct tensor_dim {
      //! raw type for testing
      using T = std::remove_cv_t<std::remove_reference_t<Derived>>;
      static_assert(is_matrix<T>::value,
                    "The type of T is not understood as an Eigen::Matrix");
      static_assert(is_square<T>::value, "T's matrix isn't square");
      //! evaluated dimension
      constexpr static Index_t value{T::RowsAtCompileTime};
    };

    //! computes the dimension from a fourth order tensor represented
    //! by a square matrix
    template <class Derived>
    struct tensor_4_dim {
      //! raw type for testing
      using T = std::remove_reference_t<Derived>;
      static_assert(is_matrix<T>::value,
                    "The type of t is not understood as an Eigen::Matrix");
      static_assert(is_square<T>::value, "t's matrix isn't square");
      //! evaluated dimension
      constexpr static Index_t value{ct_sqrt(T::RowsAtCompileTime)};
      static_assert(value * value == T::RowsAtCompileTime,
                    "This is not a fourth-order tensor mapped on a square "
                    "matrix");
    };

    namespace internal {
      /**
       * determine the rank of a Dim-dimensional tensor represented by an
       * `Eigen::Matrix` of shape NbRow × NbCol
       *
       * @tparam Dim spatial dimension
       * @tparam NbRow number of rows
       * @tparam NbCol number of columns
       */
      template <Dim_t Dim, Dim_t NbRow, Dim_t NbCol>
      constexpr inline Dim_t get_rank() {
        constexpr bool IsVec{(NbRow == Dim) and (NbCol == 1)};
        constexpr bool IsMat{(NbRow == Dim) and (NbCol == NbRow)};
        constexpr bool IsTen{(NbRow == Dim * Dim) and (NbCol == Dim * Dim)};
        static_assert(IsVec or IsMat or IsTen,
                      "can't understand the data type as a first-, second-, or "
                      "fourth-order tensor");
        if (IsVec) {
          return firstOrder;
        } else if (IsMat) {
          return secondOrder;
        } else if (IsTen) {
          return fourthOrder;
        }
        return zerothOrder;
      }

    }  // namespace internal

    /**
     * computes the rank of a tensor given the spatial dimension
     */
    template <class Derived, Dim_t Dim>
    struct tensor_rank {
      using T = std::remove_reference_t<Derived>;
      static_assert(is_matrix<T>::value,
                    "The type of t is not understood as an Eigen::Matrix");
      static constexpr Dim_t value{internal::get_rank<Dim, T::RowsAtCompileTime,
                                                      T::ColsAtCompileTime>()};
    };

  }  // namespace EigenCheck

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_UTIL_EIGEN_CHECKS_HH_
