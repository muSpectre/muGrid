/**
 * file   eigen_tools.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Sep 2017
 *
 * @brief  small tools to be used with Eigen
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

#ifndef EIGEN_TOOLS_H
#define EIGEN_TOOLS_H

#include <utility>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>
#include "common/common.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  //! Creates a Eigen::Sizes type for a Tensor defined by an order and dim
  namespace internal {

    template <Dim_t order, Dim_t dim, Dim_t... dims>
    struct SizesByOrderHelper {
      using Sizes = typename SizesByOrderHelper<order-1, dim, dim, dims...>::Sizes;
    };
    template <Dim_t dim, Dim_t... dims>
    struct SizesByOrderHelper<0, dim, dims...> {
      using Sizes = Eigen::Sizes<dims...>;
    };

  }  // internal

  template <Dim_t order, Dim_t dim>
  struct SizesByOrder {
    static_assert(order > 0, "works only for order greater than zero");
    using Sizes = typename
      internal::SizesByOrderHelper<order-1, dim, dim>::Sizes;
  };

  /* ---------------------------------------------------------------------- */
  //! Call a passed lambda with the unpacked sizes as arguments
  namespace internal {

    /* ---------------------------------------------------------------------- */
    template<Dim_t order, typename Fun_t, Dim_t dim, Dim_t ... args>
    struct CallSizesHelper {
      static decltype(auto) call(Fun_t && fun) {
        static_assert(order > 0, "can't handle empty sizes b)");
        return CallSizesHelper<order-1, Fun_t, dim, dim, args...>::call
          (fun);
      }
    };

    /* ---------------------------------------------------------------------- */
    template<typename Fun_t, Dim_t dim, Dim_t ... args>
    struct CallSizesHelper<0, Fun_t, dim, args...> {
      static decltype(auto) call(Fun_t && fun) {
        return fun(args...);
      }
    };

  }  // internal
  template<Dim_t order, Dim_t dim, typename Fun_t>
  inline decltype(auto) call_sizes(Fun_t && fun) {
    static_assert(order > 1, "can't handle empty sizes");
    return internal::CallSizesHelper<order-1, Fun_t, dim, dim>::
      call(std::forward<Fun_t>(fun));
  }


  //compile-time square root
  static constexpr Dim_t ct_sqrt(Dim_t res, Dim_t l, Dim_t r){
    if(l == r){
      return r;
    } else {
      const auto mid = (r + l) / 2;

      if(mid * mid >= res){
        return ct_sqrt(res, l, mid);
      } else {
        return ct_sqrt(res, mid + 1, r);
      }
    }
  }

  static constexpr Dim_t ct_sqrt(Dim_t res){
    return ct_sqrt(res, 1, res);
  }

  /**
   * Structure to determine whether an expression can be evaluated into a Matrix, Array, etc. and which helps determine compile-time size
   */
  namespace EigenCheck {
    template <class Derived>
    struct is_matrix {
      using T = std::remove_reference_t<Derived>;
      constexpr static bool value{std::is_same<typename Eigen::internal::traits<T>::XprKind,
                                               Eigen::MatrixXpr>::value};
    };

    template <class Derived>
    struct is_fixed {
      using T = std::remove_reference_t<Derived>;
      constexpr static bool value{T::SizeAtCompileTime != Eigen::Dynamic};
    };

    template <class Derived>
    struct is_square {
      using T = std::remove_reference_t<Derived>;
      constexpr static bool value{
        (T::RowsAtCompileTime == T::ColsAtCompileTime) &&
          is_fixed<T>::value};

    };

    template <class Derived>
    struct tensor_dim {
      using T = std::remove_reference_t<Derived>;
      static_assert(is_matrix<T>::value, "The type of t is not understood as an Eigen::Matrix");
      static_assert(is_square<T>::value, "t's matrix isn't square");
      constexpr static Dim_t value{T::RowsAtCompileTime};
    };

    template <class Derived>
    struct tensor_4_dim {
      using T = std::remove_reference_t<Derived>;
      static_assert(is_matrix<T>::value, "The type of t is not understood as an Eigen::Matrix");
      static_assert(is_square<T>::value, "t's matrix isn't square");
      constexpr static Dim_t value{ct_sqrt(T::RowsAtCompileTime)};
      static_assert(value*value == T::RowsAtCompileTime,
                    "This is not a fourth-order tensor mapped on a square "
                    "matrix");
    };

  };



}  // muSpectre


#endif /* EIGEN_TOOLS_H */
