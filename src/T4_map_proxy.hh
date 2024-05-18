/**
 * @file   T4_map_proxy.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   19 Nov 2017
 *
 * @brief  Map type to allow fourth-order tensor-like maps on 2D matrices
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

#ifndef SRC_LIBMUGRID_T4_MAP_PROXY_HH_
#define SRC_LIBMUGRID_T4_MAP_PROXY_HH_

#include "eigen_tools.hh"

#include <Eigen/Dense>
#include <Eigen/src/Core/util/Constants.h>

#include <type_traits>

namespace muGrid {

  /**
   * simple adapter function to create a matrix that can be mapped as a tensor
   */
  template <typename T, Dim_t Dim>
  using T4Mat = Eigen::Matrix<T, Dim * Dim, Dim * Dim>;

  /**
   * Map onto `muGrid::T4Mat`
   */
  template <typename T, Dim_t Dim, bool ConstMap = false>
  using T4MatMap = std::conditional_t<ConstMap, Eigen::Map<const T4Mat<T, Dim>>,
                                      Eigen::Map<T4Mat<T, Dim>>>;

  template <class Derived>
  struct DimCounter {};

  /**
   * Convenience structure to determine the spatial dimension of a tensor
   * represented by a fixed-size `Eigen::Matrix`. used to derive spatial
   * dimension from input arguments of template functions thus avoiding the need
   * for redundant explicit specification.
   */
  template <class Derived>
  struct DimCounter<Eigen::MatrixBase<Derived>> {
   private:
    using Type = Eigen::MatrixBase<Derived>;
    constexpr static Dim_t Rows{Type::RowsAtCompileTime};

   public:
    static_assert(Rows != Eigen::Dynamic, "matrix type not statically sized");
    static_assert(Rows == Type::ColsAtCompileTime, "matrix type not square");
    //! storage for the dimension
    constexpr static Dim_t value{ct_sqrt(Rows)};
    static_assert(value * value == Rows,
                  "Only integer numbers of dimensions allowed");
  };

  /**
   * provides index-based access to fourth-order Tensors represented
   * by square matrices
   */
  template <typename T4>
  inline auto get(const Eigen::MatrixBase<T4> & t4, Dim_t i, Dim_t j, Dim_t k,
                  Dim_t l) -> decltype(auto) {
    constexpr Dim_t Dim{DimCounter<Eigen::MatrixBase<T4>>::value};
    const auto myColStride{(t4.colStride() == 1) ? t4.colStride()
                                                 : t4.colStride() / Dim};
    const auto myRowStride{(t4.rowStride() == 1) ? t4.rowStride()
                                                 : t4.rowStride() / Dim};
    return t4(i * myRowStride + j * myColStride,
              k * myRowStride + l * myColStride);
  }

  /**
   * provides constant index-based access to fourth-order Tensors represented
   * by square matrices
   */
  template <typename T4>
  inline auto get(Eigen::MatrixBase<T4> & t4, Dim_t i, Dim_t j, Dim_t k,
                  Dim_t l) -> decltype(t4.coeffRef(i, j)) {
    constexpr Dim_t Dim{DimCounter<Eigen::MatrixBase<T4>>::value};
    const auto myColStride{(t4.colStride() == 1) ? t4.colStride()
                                                 : t4.colStride() / Dim};
    const auto myRowStride{(t4.rowStride() == 1) ? t4.rowStride()
                                                 : t4.rowStride() / Dim};
    return t4.coeffRef(i * myRowStride + j * myColStride,
                       k * myRowStride + l * myColStride);
  }

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_T4_MAP_PROXY_HH_
