/**
 * @file   core/types.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Jan 2019
 *
 * @brief  Scalar type definitions for muGrid
 *
 * Copyright © 2019 Till Junge
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

#ifndef SRC_LIBMUGRID_CORE_TYPES_HH_
#define SRC_LIBMUGRID_CORE_TYPES_HH_

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

// Compiler compatibility macros
#if defined(_MSC_VER)
    #define MUGRID_RESTRICT __restrict
#else
    #define MUGRID_RESTRICT __restrict__
#endif

namespace muGrid {

    /**
     * \defgroup Scalars Scalar types
     * @{
     */

    /**
     * @typedef Dim_t
     * @brief A type alias for signed integers used for static dimensions.
     *
     * This type alias is used to represent signed integers for static
     * dimensions in the µGrid codebase. It is used for consistency throughout
     * the code. It is also capable of representing -1, which can be used to
     * denote unknown or undefined dimensions.
     */
    using Dim_t = int;

    /**
     * @typedef Index_t
     * @brief A type alias for std::ptrdiff_t used for size-related values.
     *
     * This type alias is used to represent size-related values in the µGrid
     * codebase. It ensures compatibility with Eigen's indexing system and
     * supports large arrays that exceed the range of Dim_t. For example,
     * arrays with dimensions 65536 × 65536 would overflow Dim_t, so Index_t
     * is used instead.
     */
    using Index_t = std::ptrdiff_t;

    /**
     * @typedef Size_t
     * @brief A type alias for std::size_t used for size-related values.
     *
     * This type alias is used to represent unsigned size-related values in
     * the µGrid codebase. It is typically used for indexing and size
     * calculations where negative values are not expected.
     */
    using Size_t = std::size_t;

    using Uint = unsigned int;  //!< type to use in math for unsigned integers
    using Int = int;            //!< type to use in math for signed integers
    using Real = double;        //!< type to use in math for real numbers
    using Complex =
        std::complex<Real>;  //!< type to use in math for complex numbers

    /**@}*/

    //! Dimension constants
    constexpr Dim_t oneD{1};    //!< constant for a one-dimensional problem
    constexpr Dim_t twoD{2};    //!< constant for a two-dimensional problem
    constexpr Dim_t threeD{3};  //!< constant for a three-dimensional problem
    constexpr Dim_t fourD{4};   //!< constant for a four-dimensional problem

    //! Tensor order constants
    constexpr Index_t zerothOrder{0};  //!< constant for scalars
    constexpr Index_t firstOrder{1};   //!< constant for vectors
    constexpr Index_t secondOrder{2};  //!< constant second-order tensors
    constexpr Index_t fourthOrder{4};  //!< constant fourth-order tensors

    //! Quadrature/node constants
    constexpr Index_t OneQuadPt{1};   //!< constant for 1 quadrature point/pixel
    constexpr Index_t TwoQuadPts{2};  //!< constant for 2 quadrature point/pixel
    constexpr Index_t FourQuadPts{
        4};                        //!< constant for 4 quadrature point/pixel
    constexpr Index_t OneNode{1};  //!< constant for 1 node per pixel

    //! constant used to explicitly denote unknown positive integers
    constexpr static Dim_t Unknown{-1};

    //! Type used for shapes and strides
    using Shape_t = std::vector<Index_t>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CORE_TYPES_HH_
