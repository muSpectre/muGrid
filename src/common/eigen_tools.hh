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

}  // muSpectre


#endif /* EIGEN_TOOLS_H */
