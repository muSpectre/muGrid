/**
 * @file   eigen_tools.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Sep 2017
 *
 * @brief  small tools to be used with Eigen
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

#ifndef EIGEN_TOOLS_H
#define EIGEN_TOOLS_H

#include "common/common.hh"

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include <utility>
#include <type_traits>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  namespace internal {

    //! Creates a Eigen::Sizes type for a Tensor defined by an order and dim
    template <Dim_t order, Dim_t dim, Dim_t... dims>
    struct SizesByOrderHelper {
      //! type to use
      using Sizes = typename SizesByOrderHelper<order-1, dim, dim, dims...>::Sizes;
    };
    //! Creates a Eigen::Sizes type for a Tensor defined by an order and dim
    template <Dim_t dim, Dim_t... dims>
    struct SizesByOrderHelper<0, dim, dims...> {
      //! type to use
      using Sizes = Eigen::Sizes<dims...>;
    };

  }  // internal

  //! Creates a Eigen::Sizes type for a Tensor defined by an order and dim
  template <Dim_t order, Dim_t dim>
  struct SizesByOrder {
    static_assert(order > 0, "works only for order greater than zero");
    //! `Eigen::Sizes`
    using Sizes = typename
      internal::SizesByOrderHelper<order-1, dim, dim>::Sizes;
  };

  /* ---------------------------------------------------------------------- */
  namespace internal {

    /* ---------------------------------------------------------------------- */
    //! Call a passed lambda with the unpacked sizes as arguments
    template<Dim_t order, typename Fun_t, Dim_t dim, Dim_t ... args>
    struct CallSizesHelper {
      //! applies the call
      static decltype(auto) call(Fun_t && fun) {
        static_assert(order > 0, "can't handle empty sizes b)");
        return CallSizesHelper<order-1, Fun_t, dim, dim, args...>::call
          (fun);
      }
    };

    /* ---------------------------------------------------------------------- */
    template<typename Fun_t, Dim_t dim, Dim_t ... args>
    //! Call a passed lambda with the unpacked sizes as arguments
    struct CallSizesHelper<0, Fun_t, dim, args...> {
      //! applies the call
      static decltype(auto) call(Fun_t && fun) {
        return fun(args...);
      }
    };

  }  // internal

  /**
   * takes a lambda and calls it with the proper `Eigen::Sizes`
   * unpacked as arguments. Is used to call constructors of a
   * `Eigen::Tensor` or map thereof in a context where the spatial
   * dimension is templated
   */
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

  namespace EigenCheck {
    /**
     * Structure to determine whether an expression can be evaluated
     * into a `Eigen::Matrix`, `Eigen::Array`, etc. and which helps
     * determine compile-time size
     */
    template <class Derived>
    struct is_matrix {
      //! raw type for testing
      using T = std::remove_reference_t<Derived>;
      //! evaluated test
      constexpr static bool value{std::is_same<typename Eigen::internal::traits<T>::XprKind,
                                               Eigen::MatrixXpr>::value};
    };

    /**
     * Helper class to check whether an `Eigen::Array` or
     * `Eigen::Matrix` is statically sized
     */
    template <class Derived>
    struct is_fixed {
      //! raw type for testing
      using T = std::remove_reference_t<Derived>;
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
      using T = std::remove_reference_t<Derived>;
      //! true if the object is square and statically sized
      constexpr static bool value{
        (T::RowsAtCompileTime == T::ColsAtCompileTime) &&
          is_fixed<T>::value};

    };

    /**
     * computes the dimension from a second order tensor represented
     * square matrix or array
     */
    template <class Derived>
    struct tensor_dim {
      //! raw type for testing
      using T = std::remove_reference_t<Derived>;
      static_assert(is_matrix<T>::value, "The type of t is not understood as an Eigen::Matrix");
      static_assert(is_square<T>::value, "t's matrix isn't square");
      //! evaluated dimension
      constexpr static Dim_t value{T::RowsAtCompileTime};
    };

    //! computes the dimension from a fourth order tensor represented
    //! by a square matrix
    template <class Derived>
    struct tensor_4_dim {
      //! raw type for testing
      using T = std::remove_reference_t<Derived>;
      static_assert(is_matrix<T>::value, "The type of t is not understood as an Eigen::Matrix");
      static_assert(is_square<T>::value, "t's matrix isn't square");
      //! evaluated dimension
      constexpr static Dim_t value{ct_sqrt(T::RowsAtCompileTime)};
      static_assert(value*value == T::RowsAtCompileTime,
                    "This is not a fourth-order tensor mapped on a square "
                    "matrix");
    };

  };


  namespace log_comp {
    template <Dim_t dim>
    using Mat_t = Eigen::Matrix<Real, dim, dim>;
    template <Dim_t dim>
    using Vec_t = Eigen::Matrix<Real, dim, 1>;

    //! This is a static implementation of the explicit determination
    //! of log(Tensor) following Jog, C.S. J Elasticity (2008) 93:
    //! 141. https://doi.org/10.1007/s10659-008-9169-x
    /* ---------------------------------------------------------------------- */
    template <Dim_t dim, Dim_t i, Dim_t j = dim-1>
    struct Proj {
      static inline decltype(auto) compute(const Vec_t<dim> & eigs, const Mat_t<dim> & T) {
        static_assert(dim > 0, "only works for positive dimensions");
        return 1./(eigs(i) -eigs(j))*(T-eigs(j)*Mat_t<dim>::Identity()) * Proj<dim, i, j-1>::compute(eigs, T);
      }
    };

    // catch the case when there's nothing to do
    template <Dim_t dim, Dim_t other>
    struct Proj<dim,other,other> {
      static inline decltype(auto) compute(const Vec_t<dim> & eigs, const Mat_t<dim> & T) {
        static_assert(dim > 0, "only works for positive dimensions");
        return Proj<dim, other, other-1>::compute(eigs, T);
      }
    };

    // catch the normal tail case
    template <Dim_t dim, Dim_t i>
    struct Proj<dim,i,0> {
      static constexpr Dim_t j{0};
      static inline decltype(auto) compute(const Vec_t<dim> & eigs, const Mat_t<dim> & T) {
        static_assert(dim > 0, "only works for positive dimensions");
        return 1./(eigs(i) -eigs(j))*(T-eigs(j)*Mat_t<dim>::Identity());
      }
    };

    // catch the tail case when the last dimension is i
    template <Dim_t dim>
    struct Proj<dim,0,1> {
      static constexpr Dim_t i{0};
      static constexpr Dim_t j{1};
      static inline decltype(auto) compute(const Vec_t<dim> & eigs, const Mat_t<dim> & T) {
        static_assert(dim > 0, "only works for positive dimensions");
        return 1./(eigs(i) -eigs(j))*(T-eigs(j)*Mat_t<dim>::Identity());
      }
    };

    template <>
    struct Proj<1, 0, 0> {
      static constexpr Dim_t dim{1};
      static constexpr Dim_t i{0};
      static constexpr Dim_t j{0};

      static inline decltype(auto) compute(const Vec_t<dim> & /*eigs*/, const Mat_t<dim> & /*T*/) {
        return Mat_t<dim>::Identity();
      }
    };


    template <Dim_t dim, Dim_t i>
    inline decltype(auto) P(const Vec_t<dim> & eigs, const Mat_t<dim> & T) {
      return Proj<dim, i>::compute(eigs, T);
    }

    /* ---------------------------------------------------------------------- */
    template  <Dim_t dim, Dim_t i = dim-1>
    struct Summand {
      static inline  decltype(auto) compute(const Vec_t<dim> & eigs, const Mat_t<dim> & T) {
        return std::log(eigs(i))*P<dim, i>(eigs, T) + Summand<dim, i-1>::compute(eigs, T);
      }
    };

    template  <Dim_t dim>
    struct Summand <dim, 0>{
      static constexpr Dim_t i{0};
      static inline  decltype(auto) compute(const Vec_t<dim> & eigs, const Mat_t<dim> & T) {
        return std::log(eigs(i))*P<dim, i>(eigs, T);
      }
    };

    template <Dim_t dim>
    inline decltype(auto) Sum(const Vec_t<dim> & eigs, const Mat_t<dim> & T) {
      return Summand<dim>::compute(eigs, T);
    }


  }  // log_comp

  /**
   * computes the matrix logarithm efficiently for dim=1, 2, or 3 for
   * a diagonizable tensor. For larger tensors, better use the direct
   * eigenvalue/vector computation
   */
  template <Dim_t dim>
  inline decltype(auto) logm(const log_comp::Mat_t<dim>& mat) {
    using Mat = log_comp::Mat_t<dim>;
    Eigen::SelfAdjointEigenSolver<Mat> Solver{};
    Solver.computeDirect(mat, Eigen::EigenvaluesOnly);
    return Mat{log_comp::Sum(Solver.eigenvalues(), mat)};
  }


  /**
   * compute the matrix exponential. This may not be the most
   * efficient way to do this
   */
  template <Dim_t dim>
  inline decltype(auto) expm(const log_comp::Mat_t<dim>& mat) {
    using Mat = log_comp::Mat_t<dim>;
    Eigen::SelfAdjointEigenSolver<Mat> Solver{};
    Solver.computeDirect(mat, Eigen::ComputeEigenvectors);
    Mat retval{Mat::Zero()};
    for (Dim_t i = 0; i < dim; ++i) {
      auto && val = Solver.eigenvalues()(i);
      auto && vec = Solver.eigenvectors().col(i);
      retval += std::exp(val) * vec * vec.transpose();
    }
    return retval;
  }

}  // muSpectre


#endif /* EIGEN_TOOLS_H */
