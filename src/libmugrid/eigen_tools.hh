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

#ifndef SRC_LIBMUGRID_EIGEN_TOOLS_HH_
#define SRC_LIBMUGRID_EIGEN_TOOLS_HH_

#include "grid_common.hh"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <type_traits>
#include <utility>

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  namespace internal {

    //! Creates a Eigen::Sizes type for a Tensor defined by an order and dim
    template <Dim_t order, Dim_t dim, Dim_t... dims>
    struct SizesByOrderHelper {
      //! type to use
      using Sizes =
          typename SizesByOrderHelper<order - 1, dim, dim, dims...>::Sizes;
    };
    //! Creates a Eigen::Sizes type for a Tensor defined by an order and dim
    template <Dim_t dim, Dim_t... dims>
    struct SizesByOrderHelper<0, dim, dims...> {
      //! type to use
      using Sizes = Eigen::Sizes<dims...>;
    };

  }  // namespace internal

  //! Creates a Eigen::Sizes type for a Tensor defined by an order and dim
  template <Dim_t order, Dim_t dim>
  struct SizesByOrder {
    static_assert(order > 0, "works only for order greater than zero");
    //! `Eigen::Sizes`
    using Sizes =
        typename internal::SizesByOrderHelper<order - 1, dim, dim>::Sizes;
  };

  /* ---------------------------------------------------------------------- */
  namespace internal {

    /* ---------------------------------------------------------------------- */
    //! Call a passed lambda with the unpacked sizes as arguments
    template <Dim_t order, typename Fun_t, Dim_t dim, Dim_t... args>
    struct CallSizesHelper {
      //! applies the call
      static decltype(auto) call(Fun_t && fun) {
        static_assert(order > 0, "can't handle empty sizes b)");
        return CallSizesHelper<order - 1, Fun_t, dim, dim, args...>::call(fun);
      }
    };

    /* ---------------------------------------------------------------------- */
    template <typename Fun_t, Dim_t dim, Dim_t... args>
    //! Call a passed lambda with the unpacked sizes as arguments
    struct CallSizesHelper<0, Fun_t, dim, args...> {
      //! applies the call
      static decltype(auto) call(Fun_t && fun) { return fun(args...); }
    };

  }  // namespace internal

  /**
   * takes a lambda and calls it with the proper `Eigen::Sizes`
   * unpacked as arguments. Is used to call constructors of a
   * `Eigen::Tensor` or map thereof in a context where the spatial
   * dimension is templated
   */
  template <Dim_t order, Dim_t dim, typename Fun_t>
  inline decltype(auto) call_sizes(Fun_t && fun) {
    static_assert(order > 1, "can't handle empty sizes");
    return internal::CallSizesHelper<order - 1, Fun_t, dim, dim>::call(
        std::forward<Fun_t>(fun));
  }

  // compile-time square root
  static constexpr Dim_t ct_sqrt(Dim_t res, Dim_t l, Dim_t r) {
    if (l == r) {
      return r;
    } else {
      const auto mid = (r + l) / 2;

      if (mid * mid >= res) {
        return ct_sqrt(res, l, mid);
      } else {
        return ct_sqrt(res, mid + 1, r);
      }
    }
  }

  static constexpr Dim_t ct_sqrt(Dim_t res) { return ct_sqrt(res, 1, res); }

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
      using T = std::remove_reference_t<Derived>;
      static_assert(is_matrix<T>::value,
                    "The type of t is not understood as an Eigen::Matrix");
      static_assert(is_square<T>::value, "t's matrix isn't square");
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

  namespace log_comp {
    //! Matrix type used for logarithm evaluation
    template <Dim_t dim>
    using Mat_t = Eigen::Matrix<Real, dim, dim>;
    //! Vector type used for logarithm evaluation
    template <Dim_t dim>
    using Vec_t = Eigen::Matrix<Real, dim, 1>;

    //! This is a static implementation of the explicit determination
    //! of log(Tensor) following Jog, C.S. J Elasticity (2008) 93:
    //! 141. https://doi.org/10.1007/s10659-008-9169-x
    /* ---------------------------------------------------------------------- */
    template <Dim_t Dim, Dim_t I, Dim_t J = Dim - 1>
    struct Proj {
      //! wrapped function (raison d'être)
      static inline decltype(auto) compute(const Vec_t<Dim> & eigs,
                                           const Mat_t<Dim> & T) {
        static_assert(Dim > 0, "only works for positive dimensions");
        return 1. / (eigs(I) - eigs(J)) *
               (T - eigs(J) * Mat_t<Dim>::Identity()) *
               Proj<Dim, I, J - 1>::compute(eigs, T);
      }
    };

    //! catch the case when there's nothing to do
    template <Dim_t Dim, Dim_t Other>
    struct Proj<Dim, Other, Other> {
      //! wrapped function (raison d'être)
      static inline decltype(auto) compute(const Vec_t<Dim> & eigs,
                                           const Mat_t<Dim> & T) {
        static_assert(Dim > 0, "only works for positive dimensions");
        return Proj<Dim, Other, Other - 1>::compute(eigs, T);
      }
    };

    //! catch the normal tail case
    template <Dim_t Dim, Dim_t I>
    struct Proj<Dim, I, 0> {
      static constexpr Dim_t j{0};  //!< short-hand
      //! wrapped function (raison d'être)
      static inline decltype(auto) compute(const Vec_t<Dim> & eigs,
                                           const Mat_t<Dim> & T) {
        static_assert(Dim > 0, "only works for positive dimensions");
        return 1. / (eigs(I) - eigs(j)) *
               (T - eigs(j) * Mat_t<Dim>::Identity());
      }
    };

    //! catch the tail case when the last dimension is i
    template <Dim_t Dim>
    struct Proj<Dim, 0, 1> {
      static constexpr Dim_t I{0};  //!< short-hand
      static constexpr Dim_t J{1};  //!< short-hand

      //! wrapped function (raison d'être)
      static inline decltype(auto) compute(const Vec_t<Dim> & eigs,
                                           const Mat_t<Dim> & T) {
        static_assert(Dim > 0, "only works for positive dimensions");
        return 1. / (eigs(I) - eigs(J)) *
               (T - eigs(J) * Mat_t<Dim>::Identity());
      }
    };

    //! catch the general tail case
    template <>
    struct Proj<1, 0, 0> {
      static constexpr Dim_t Dim{1};  //!< short-hand
      static constexpr Dim_t I{0};    //!< short-hand
      static constexpr Dim_t J{0};    //!< short-hand

      //! wrapped function (raison d'être)
      static inline decltype(auto) compute(const Vec_t<Dim> & /*eigs*/,
                                           const Mat_t<Dim> & /*T*/) {
        return Mat_t<Dim>::Identity();
      }
    };

    //! Product term
    template <Dim_t Dim, Dim_t I>
    inline decltype(auto) P(const Vec_t<Dim> & eigs, const Mat_t<Dim> & T) {
      return Proj<Dim, I>::compute(eigs, T);
    }

    //! sum term
    template <Dim_t Dim, Dim_t I = Dim - 1>
    struct Summand {
      //! wrapped function (raison d'être)
      static inline decltype(auto) compute(const Vec_t<Dim> & eigs,
                                           const Mat_t<Dim> & T) {
        return std::log(eigs(I)) * P<Dim, I>(eigs, T) +
               Summand<Dim, I - 1>::compute(eigs, T);
      }
    };

    //! sum term
    template <Dim_t Dim>
    struct Summand<Dim, 0> {
      static constexpr Dim_t I{0};  //!< short-hand
      //! wrapped function (raison d'être)
      static inline decltype(auto) compute(const Vec_t<Dim> & eigs,
                                           const Mat_t<Dim> & T) {
        return std::log(eigs(I)) * P<Dim, I>(eigs, T);
      }
    };

    //! sum implementation
    template <Dim_t Dim>
    inline decltype(auto) Sum(const Vec_t<Dim> & eigs, const Mat_t<Dim> & T) {
      return Summand<Dim>::compute(eigs, T);
    }

  }  // namespace log_comp

  template <Dim_t Dim>
  using Matrix_t = Eigen::Matrix<Real, Dim, Dim>;

  template <Dim_t Dim>
  using SelfAdjointDecomp_t =
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, Dim, Dim>>;
  /**
   * computes the matrix logarithm efficiently for dim=1, 2, or 3 for
   * a diagonizable tensor. For larger tensors, better use the direct
   * eigenvalue/vector computation
   */
  template <Dim_t Dim>
  inline decltype(auto) logm(const log_comp::Mat_t<Dim> & mat) {
    using Mat_t = Eigen::Matrix<Real, Dim, Dim>;
    Eigen::SelfAdjointEigenSolver<Mat_t> Solver{};
    Solver.computeDirect(mat, Eigen::EigenvaluesOnly);
    return Mat_t{log_comp::Sum(Solver.eigenvalues(), mat)};
  }

  template <Dim_t Dim>
  using Matrix_t = Eigen::Matrix<Real, Dim, Dim>;
  /**
   * compute the spectral decomposition
   */
  template <class Derived, template <class Matrix_t>
                           class DecompType = Eigen::SelfAdjointEigenSolver>
  inline decltype(auto)
  spectral_decomposition(const Eigen::MatrixBase<Derived> & mat) {
    static_assert(Derived::SizeAtCompileTime != Eigen::Dynamic,
                  "works only for static matrices");
    static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime,
                  "works only for square matrices");
    constexpr Dim_t Dim{Derived::RowsAtCompileTime};
    using Mat_t = Eigen::Matrix<Real, Dim, Dim>;
    DecompType<Mat_t> Solver{};
    Solver.computeDirect(mat, Eigen::ComputeEigenvectors);
    return Solver;
  }

  /**
   * It seems we only need to take logs of self-adjoint matrices
   */

  template <Dim_t Dim, template <class Matrix_t>
                       class DecompType = Eigen::SelfAdjointEigenSolver>
  inline decltype(auto)
  logm_alt(const DecompType<Matrix_t<Dim>> & spectral_decomp) {
    using Mat_t = Eigen::Matrix<Real, Dim, Dim>;
    Mat_t retval{Mat_t::Zero()};
    for (Dim_t i{0}; i < Dim; ++i) {
      const Real & val{spectral_decomp.eigenvalues()(i)};
      auto & vec{spectral_decomp.eigenvectors().col(i)};
      retval += std::log(val) * vec * vec.transpose();
    }
    return retval;
  }

  /**
   * compute the matrix log with a spectral decomposition. This may not be the
   * most efficient way to do this
   */
  template <class Derived>
  inline decltype(auto) logm_alt(const Eigen::MatrixBase<Derived> & mat) {
    static_assert(Derived::SizeAtCompileTime != Eigen::Dynamic,
                  "works only for static matrices");
    static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime,
                  "works only for square matrices");
    constexpr Dim_t Dim{Derived::RowsAtCompileTime};
    using Mat_t = Eigen::Matrix<Real, Dim, Dim>;
    using Decomp_t = Eigen::SelfAdjointEigenSolver<Mat_t>;

    Decomp_t decomp{spectral_decomposition(mat)};

    return logm_alt(decomp);
  }

  /**
   * Uses a pre-existing spectral decomposition of a matrix to compute its
   * exponential
   *
   * @param spectral_decomp spectral decomposition of a matrix
   * @tparam Dim spatial dimension (i.e., number of rows and colums in the
   * matrix)
   */
  template <Dim_t Dim, template <class Matrix_t>
                       class DecompType = Eigen::SelfAdjointEigenSolver>
  inline decltype(auto)
  expm(const DecompType<Matrix_t<Dim>> & spectral_decomp) {
    using Mat_t = Matrix_t<Dim>;
    Mat_t retval{Mat_t::Zero()};
    for (Dim_t i{0}; i < Dim; ++i) {
      const Real & val{spectral_decomp.eigenvalues()(i)};
      auto & vec{spectral_decomp.eigenvectors().col(i)};
      retval += std::exp(val) * vec * vec.transpose();
    }
    return retval;
  }

  /**
   * compute the matrix exponential with a spectral decomposition. This may not
   * be the most efficient way to do this
   */
  template <class Derived>
  inline decltype(auto) expm(const Eigen::MatrixBase<Derived> & mat) {
    static_assert(Derived::SizeAtCompileTime != Eigen::Dynamic,
                  "works only for static matrices");
    static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime,
                  "works only for square matrices");
    constexpr Dim_t Dim{Derived::RowsAtCompileTime};
    using Mat_t = Eigen::Matrix<Real, Dim, Dim>;
    using Decomp_t = Eigen::SelfAdjointEigenSolver<Mat_t>;

    Decomp_t decomp{spectral_decomposition(mat)};

    return expm(decomp);
  }

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_EIGEN_TOOLS_HH_
