/**
 * @file   grid/strides.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   29 Sep 2017
 *
 * @brief  Stride calculation utilities for grid operations
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

#ifndef SRC_LIBMUGRID_GRID_STRIDES_HH_
#define SRC_LIBMUGRID_GRID_STRIDES_HH_

#include "core/types.hh"
#include "core/coordinates.hh"
#include "core/exception.hh"

#include <utility>
#include <numeric>
#include <algorithm>

namespace muGrid {
    namespace CcoordOps {
        namespace internal {
            //! simple helper returning the first argument and ignoring the
            //! second
            template <typename T>
            constexpr T ret(T val, size_t /*dummy*/) {
                return val;
            }

            //! helper to build cubes
            template <Dim_t Dim, typename T, size_t... I>
            constexpr std::array<T, Dim> cube_fun(T val,
                                                  std::index_sequence<I...>) {
                return std::array<T, Dim>{ret(val, I)...};
            }

            //! compute the stride in a direction of a column-major grid,
            //! s_k = prod_{i < k} N_i (first axis fastest)
            template <Dim_t Dim>
            constexpr Index_t
            col_major_stride(const GridIndex<Dim> & nb_grid_pts,
                             const size_t index) {
                static_assert(Dim > 0,
                              "only for positive numbers of dimensions");

                Index_t ret_val{1};
                for (size_t i{0}; i < index; ++i) {
                    ret_val *= nb_grid_pts[i];
                }
                return ret_val;
            }

            //! compute the stride in a direction of a row-major grid,
            //! s_k = prod_{i > k} N_i (last axis fastest)
            template <Dim_t Dim>
            constexpr Index_t
            row_major_stride(const GridIndex<Dim> & nb_grid_pts,
                             const size_t index) {
                static_assert(Dim > 0,
                              "only for positive numbers of dimensions");

                Index_t ret_val{1};
                for (size_t i{Dim - 1}; i > index; --i) {
                    ret_val *= nb_grid_pts[i];
                }
                return ret_val;
            }

            //! get all strides from a column-major grid (helper function)
            template <Dim_t Dim, size_t... I>
            constexpr GridIndex<Dim>
            compute_col_major_strides(const GridIndex<Dim> & nb_grid_pts,
                                      std::index_sequence<I...>) {
                return GridIndex<Dim>{col_major_stride<Dim>(nb_grid_pts, I)...};
            }

            //! get all strides from a row-major grid (helper function)
            template <Dim_t Dim, size_t... I>
            constexpr GridIndex<Dim>
            compute_row_major_strides(const GridIndex<Dim> & nb_grid_pts,
                                      std::index_sequence<I...>) {
                return GridIndex<Dim>{row_major_stride<Dim>(nb_grid_pts, I)...};
            }
        }  // namespace internal

        //! returns a grid of equal number of grid points in each direction
        template <size_t Dim, typename T>
        constexpr std::array<T, Dim> get_cube(T nb_grid_pts) {
            return internal::cube_fun<Dim>(nb_grid_pts,
                                           std::make_index_sequence<Dim>{});
        }

        //! returns a grid of equal number of grid points in each direction
        template <size_t MaxDim = fourD>  // 4 to ease alignment
        DynCoord<MaxDim> get_cube(const Dim_t & dim,
                                  const Index_t & nb_grid_pts) {
            switch (dim) {
            case oneD: {
                return DynCoord<MaxDim>{get_cube<oneD>(nb_grid_pts)};
            }
            case twoD: {
                return DynCoord<MaxDim>{get_cube<twoD>(nb_grid_pts)};
            }
            case threeD: {
                return DynCoord<MaxDim>{get_cube<threeD>(nb_grid_pts)};
            }
            default:
                throw RuntimeError("Unknown dimension");
            }
        }

        /* ----------------------------------------------------------------------
         */
        //! get all strides from a column-major grid
        template <size_t Dim>
        constexpr GridIndex<Dim>
        get_col_major_strides(const GridIndex<Dim> & nb_grid_pts) {
            return internal::compute_col_major_strides<Dim>(
                nb_grid_pts, std::make_index_sequence<Dim>{});
        }

        /* ----------------------------------------------------------------------
         */
        //! get all strides from a column-major grid
        template <size_t MaxDim>
        constexpr DynCoord<MaxDim>
        get_col_major_strides(const DynCoord<MaxDim> & nb_grid_pts) {
            switch (nb_grid_pts.get_dim()) {
            case oneD: {
                return DynCoord<MaxDim>{
                    internal::compute_col_major_strides<oneD>(
                        nb_grid_pts.template get<oneD>(),
                        std::make_index_sequence<oneD>{})};
            }
            case twoD: {
                return DynCoord<MaxDim>{
                    internal::compute_col_major_strides<twoD>(
                        nb_grid_pts.template get<twoD>(),
                        std::make_index_sequence<twoD>{})};
            }
            case threeD: {
                return DynCoord<MaxDim>{
                    internal::compute_col_major_strides<threeD>(
                        nb_grid_pts.template get<threeD>(),
                        std::make_index_sequence<threeD>{})};
            }
            default:
                throw RuntimeError("unforeseen dimensionality, is it really "
                                   "necessary to have other "
                                   "dimensions than 1, 2, and 3?");
            }
        }

        /* ----------------------------------------------------------------------
         */
        //! get all strides from a row-major grid
        template <size_t Dim>
        constexpr GridIndex<Dim>
        get_row_major_strides(const GridIndex<Dim> & nb_grid_pts) {
            return internal::compute_row_major_strides<Dim>(
                nb_grid_pts, std::make_index_sequence<Dim>{});
        }

        /* ----------------------------------------------------------------------
         */
        //! get all strides from a row-major grid
        template <size_t MaxDim>
        constexpr DynCoord<MaxDim>
        get_row_major_strides(const DynCoord<MaxDim> & nb_grid_pts) {
            switch (nb_grid_pts.get_dim()) {
            case oneD: {
                return DynCoord<MaxDim>{
                    internal::compute_row_major_strides<oneD>(
                        nb_grid_pts.template get<oneD>(),
                        std::make_index_sequence<oneD>{})};
            }
            case twoD: {
                return DynCoord<MaxDim>{
                    internal::compute_row_major_strides<twoD>(
                        nb_grid_pts.template get<twoD>(),
                        std::make_index_sequence<twoD>{})};
            }
            case threeD: {
                return DynCoord<MaxDim>{
                    internal::compute_row_major_strides<threeD>(
                        nb_grid_pts.template get<threeD>(),
                        std::make_index_sequence<threeD>{})};
            }
            default:
                throw RuntimeError("unforeseen dimensionality, is it really "
                                   "necessary to have other "
                                   "dimensions than 1, 2, and 3?");
            }
        }

        //! compute the order of the axes given strides, fastest first, i.e.
        //! the permutation pi with s_{pi(0)} <= s_{pi(1)} <= ...
        template <class T>
        T compute_axes_order(const T & shape, const T & strides) {
            T axes_order(shape.size());
            std::iota(axes_order.begin(), axes_order.end(), 0);
            std::sort(axes_order.begin(), axes_order.end(),
                      [&shape, &strides](const Dim_t & a, const Dim_t & b) {
                          // A strict weak ordering: by stride, ties (which
                          // a size-1 axis can create at any stride) by size,
                          // then by index. All ties must be broken
                          // deterministically, and a size-1 axis must sort
                          // before a longer axis of equal stride: for shape
                          // (4, 1, 3), strides (1, 4, 4) the order must be
                          // (0, 1, 2): with (0, 2, 1),
                          // get_coord_from_axes_order would peel off the
                          // size-1 axis as the slowest one and assign it the
                          // coordinate of axis 2.
                          if (strides[a] != strides[b]) {
                              return strides[a] < strides[b];
                          }
                          if (shape[a] != shape[b]) {
                              return shape[a] < shape[b];
                          }
                          return a < b;
                      });
            return axes_order;
        }

        //! compute the order of the axes given strides, fastest first
        template <size_t dim>
        GridIndex<dim> compute_axes_order(const GridIndex<dim> & shape,
                                          const GridIndex<dim> & strides) {
            GridIndex<dim> axes_order;
            std::iota(axes_order.begin(), axes_order.end(), 0);
            std::sort(axes_order.begin(), axes_order.end(),
                      [&shape, &strides](const Dim_t & a, const Dim_t & b) {
                          // A strict weak ordering: by stride, ties (which
                          // a size-1 axis can create at any stride) by size,
                          // then by index. All ties must be broken
                          // deterministically, and a size-1 axis must sort
                          // before a longer axis of equal stride: for shape
                          // (4, 1, 3), strides (1, 4, 4) the order must be
                          // (0, 1, 2): with (0, 2, 1),
                          // get_coord_from_axes_order would peel off the
                          // size-1 axis as the slowest one and assign it the
                          // coordinate of axis 2.
                          if (strides[a] != strides[b]) {
                              return strides[a] < strides[b];
                          }
                          if (shape[a] != shape[b]) {
                              return shape[a] < shape[b];
                          }
                          return a < b;
                      });
            return axes_order;
        }

        //! check whether strides represent a contiguous buffer: with the axes
        //! sorted fastest first (pi = compute_axes_order) and size-1 axes
        //! skipped, s_{pi(0)} = 1 and s_{pi(k+1)} = s_{pi(k)} N_{pi(k)}.
        //! Empty grids (zero volume) and 0-d grids count as contiguous.
        template <class T>
        bool is_buffer_contiguous(const T & nb_grid_pts, const T & strides) {
            Index_t dim{static_cast<Index_t>(nb_grid_pts.size())};
            if (dim == 0) {
                return true;
            }
            // Local copy of CcoordOps::compute_volume, which lives in
            // index_ops.hh (that header includes this one)
            auto compute_volume = [](const T & lengths) {
                typename std::remove_cv<typename std::remove_reference<
                    decltype(lengths[0])>::type>::type vol{};
                vol++;
                for (auto && length : lengths) {
                    vol *= length;
                }
                return vol;
            };
            if (compute_volume(nb_grid_pts) == 0) {
                return true;
            }
            if (static_cast<Index_t>(strides.size()) != dim) {
                throw RuntimeError(
                    "Mismatch between dimensions of nb_grid_pts and "
                    "strides");
            }
            auto axes{compute_axes_order(nb_grid_pts, strides)};
            // Accumulate the running stride in Index_t (not int): a contiguous
            // buffer for a grid > 2^31 points would otherwise overflow.
            Index_t stride{1};
            bool is_contiguous{true};
            for (Index_t i{0}; i < dim; ++i) {
                // A size-1 axis is never stepped along, so its stride does not
                // matter for contiguity.
                if (nb_grid_pts[axes[i]] == 1) {
                    continue;
                }
                is_contiguous &= strides[axes[i]] == stride;
                stride *= nb_grid_pts[axes[i]];
            }
            return is_contiguous;
        }

        //-----------------------------------------------------------------------//
        //! get the number of pixels in a grid
        template <size_t Dim>
        constexpr size_t get_size(const GridIndex<Dim> & nb_grid_pts) {
            size_t retval{1};
            for (size_t i{0}; i < Dim; ++i) {
                retval *= nb_grid_pts[i];
            }
            return retval;
        }

        //-----------------------------------------------------------------------//
        //! get the number of pixels in a grid
        template <size_t MaxDim>
        size_t get_size(const DynCoord<MaxDim> & nb_grid_pts) {
            size_t retval{1};
            Dim_t dim{nb_grid_pts.get_dim()};
            for (Dim_t i{0}; i < dim; ++i) {
                retval *= nb_grid_pts[i];
            }
            return retval;
        }

        //-----------------------------------------------------------------------//
        //! get the buffer size required to store a grid given its strides,
        //! max_k N_k s_k (the slowest axis spans the whole buffer)
        template <size_t dim>
        constexpr size_t get_buffer_size(const GridIndex<dim> & nb_grid_pts,
                                         const GridIndex<dim> & strides) {
            size_t buffer_size{0};
            // We need to loop over the dimensions because the largest stride
            // can occur anywhere. (It depends on the storage order.)
            for (size_t i{0}; i < dim; ++i) {
                buffer_size =
                    std::max(buffer_size,
                             static_cast<size_t>(nb_grid_pts[i] * strides[i]));
            }
            return buffer_size;
        }

        //-----------------------------------------------------------------------//
        //! get the buffer size required to store a grid given its strides
        size_t get_buffer_size(const DynGridIndex & nb_grid_pts,
                               const DynGridIndex & strides);

        //-----------------------------------------------------------------------//
        //! get the buffer size required to store a grid given its strides
        size_t get_buffer_size(const Shape_t & nb_grid_pts,
                               const Shape_t & strides);

    }  // namespace CcoordOps
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_GRID_STRIDES_HH_
