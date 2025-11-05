/**
 * @file   ccoord_operations.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   29 Sep 2017
 *
 * @brief  common operations on pixel addressing
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

#include <functional>
#include <numeric>
#include <utility>

#include "Eigen/Dense"

#include "exception.hh"
#include "grid_common.hh"

#ifndef SRC_LIBMUGRID_CCOORD_OPERATIONS_HH_
#define SRC_LIBMUGRID_CCOORD_OPERATIONS_HH_

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

            //! computes hermitian size according to FFTW
            template <Dim_t Dim, size_t... I>
            constexpr Ccoord_t<Dim> herm(const Ccoord_t<Dim> & nb_grid_pts,
                                         std::index_sequence<I...>) {
                return Ccoord_t<Dim>{nb_grid_pts.front() / 2 + 1,
                                     nb_grid_pts[I + 1]...};
            }

            //! compute the stride in a direction of a column-major grid
            template <Dim_t Dim>
            constexpr Index_t
            col_major_stride(const Ccoord_t<Dim> & nb_grid_pts,
                             const size_t index) {
                static_assert(Dim > 0,
                              "only for positive numbers of dimensions");

                Index_t ret_val{1};
                for (size_t i{0}; i < index; ++i) {
                    ret_val *= nb_grid_pts[i];
                }
                return ret_val;
            }

            //! compute the stride in a direction of a row-major grid
            template <Dim_t Dim>
            constexpr Index_t
            row_major_stride(const Ccoord_t<Dim> & nb_grid_pts,
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
            constexpr Ccoord_t<Dim>
            compute_col_major_strides(const Ccoord_t<Dim> & nb_grid_pts,
                                      std::index_sequence<I...>) {
                return Ccoord_t<Dim>{col_major_stride<Dim>(nb_grid_pts, I)...};
            }

            //! get all strides from a row-major grid (helper function)
            template <Dim_t Dim, size_t... I>
            constexpr Ccoord_t<Dim>
            compute_row_major_strides(const Ccoord_t<Dim> & nb_grid_pts,
                                      std::index_sequence<I...>) {
                return Ccoord_t<Dim>{row_major_stride<Dim>(nb_grid_pts, I)...};
            }
        }  // namespace internal

        //! modulo operator that can handle negative values
        template <typename T>
        T modulo(const T & a, const T & b) {
            return (b + (a % b)) % b;
        }

        //! returns a grid of equal number of grid points in each direction
        template <size_t Dim, typename T>
        constexpr std::array<T, Dim> get_cube(T nb_grid_pts) {
            return internal::cube_fun<Dim>(nb_grid_pts,
                                           std::make_index_sequence<Dim>{});
        }

        //! returns a grid of equal number of grid points in each direction
        template <size_t MaxDim = fourD>  // 4 to ease alignment
        DynCcoord<MaxDim> get_cube(const Dim_t & dim,
                                   const Index_t & nb_grid_pts) {
            switch (dim) {
            case oneD: {
                return DynCcoord<MaxDim>{get_cube<oneD>(nb_grid_pts)};
            }
            case twoD: {
                return DynCcoord<MaxDim>{get_cube<twoD>(nb_grid_pts)};
            }
            case threeD: {
                return DynCcoord<MaxDim>{get_cube<threeD>(nb_grid_pts)};
            }
            default:
                throw RuntimeError("Unknown dimension");
            }
        }

        //! return physical vector of a cell of cubic pixels
        template <size_t Dim>
        Eigen::Matrix<Real, Dim, 1> get_vector(const Ccoord_t<Dim> & ccoord,
                                               Real pix_size = 1.) {
            Eigen::Matrix<Real, Dim, 1> retval;
            for (size_t i{0}; i < Dim; ++i) {
                retval[i] = pix_size * ccoord[i];
            }
            return retval;
        }

        //! return physical vector of a cell of general pixels
        template <size_t Dim, typename T>
        Eigen::Matrix<T, Dim, 1>
        get_vector(const Ccoord_t<Dim> & ccoord,
                   Eigen::Matrix<T, Dim_t(Dim), 1> pix_size) {
            Eigen::Matrix<T, Dim, 1> retval{pix_size};
            for (size_t i{0}; i < Dim; ++i) {
                retval[i] *= ccoord[i];
            }
            return retval;
        }

        //! return physical vector of a cell of general pixels
        template <size_t Dim, typename T>
        Eigen::Matrix<T, Dim, 1>
        get_vector(const Ccoord_t<Dim> & ccoord,
                   const std::array<T, Dim> & pix_size) {
            Eigen::Matrix<T, Dim, 1> retval{};
            for (size_t i{0}; i < Dim; ++i) {
                retval[i] = pix_size[i] * ccoord[i];
            }
            return retval;
        }

        //! return physical vector of a cell of general pixels
        template <size_t Dim, size_t MaxDim, typename T>
        Eigen::Matrix<T, Dim, 1>
        get_vector(const Ccoord_t<Dim> & ccoord,
                   const DynCcoord<MaxDim, T> & pix_size) {
            assert(Dim == pix_size.get_dim());
            Eigen::Matrix<T, Dim, 1> retval{};
            for (size_t i{0}; i < Dim; ++i) {
                retval[i] = pix_size[i] * ccoord[i];
            }
            return retval;
        }

        /* ----------------------------------------------------------------------
         */
        //! return physical vector of a cell of cubic pixels
        template <size_t Dim>
        Eigen::Matrix<Real, Dim, 1> get_vector(const IntCoord_t & ccoord,
                                               Real pix_size = 1.) {
            assert(Dim == ccoord.get_dim());
            Eigen::Matrix<Real, Dim, 1> retval;
            for (size_t i{0}; i < Dim; ++i) {
                retval[i] = pix_size * ccoord[i];
            }
            return retval;
        }

        /* ----------------------------------------------------------------------
         */
        //! return physical vector of a cell of general pixels
        template <size_t Dim, typename T>
        Eigen::Matrix<T, Dim, 1>
        get_vector(const IntCoord_t ccoord,
                   Eigen::Matrix<T, Dim_t(Dim), 1> pix_size) {
            assert(Dim == ccoord.get_dim());
            Eigen::Matrix<T, Dim, 1> retval = pix_size;
            for (size_t i{0}; i < Dim; ++i) {
                retval[i] *= ccoord[i];
            }
            return retval;
        }

        /* ----------------------------------------------------------------------
         */
        //! return physical vector of a cell of general pixels
        template <size_t Dim, typename T>
        Eigen::Matrix<T, Dim, 1>
        get_vector(const IntCoord_t ccoord,
                   const std::array<T, Dim> & pix_size) {
            assert(Dim == ccoord.get_dim());
            Eigen::Matrix<T, Dim, 1> retval{};
            for (size_t i{0}; i < Dim; ++i) {
                retval[i] = pix_size[i] * ccoord[i];
            }
            return retval;
        }

        /* ----------------------------------------------------------------------
         */
        //! return physical vector of a cell of general pixels
        template <size_t Dim, size_t MaxDim, typename T>
        Eigen::Matrix<T, Dim, 1>
        get_vector(const IntCoord_t ccoord,
                   const DynCcoord<MaxDim, T> & pix_size) {
            assert(Dim == ccoord.get_dim());
            assert(Dim == pix_size.get_dim());
            Eigen::Matrix<T, Dim, 1> retval{};
            for (size_t i{0}; i < Dim; ++i) {
                retval[i] = pix_size[i] * ccoord[i];
            }
            return retval;
        }

        /* ----------------------------------------------------------------------
         */
        //! get all strides from a column-major grid
        template <size_t Dim>
        constexpr Ccoord_t<Dim>
        get_col_major_strides(const Ccoord_t<Dim> & nb_grid_pts) {
            return internal::compute_col_major_strides<Dim>(
                nb_grid_pts, std::make_index_sequence<Dim>{});
        }

        /* ----------------------------------------------------------------------
         */
        //! get all strides from a column-major grid
        template <size_t MaxDim>
        constexpr DynCcoord<MaxDim>
        get_col_major_strides(const DynCcoord<MaxDim> & nb_grid_pts) {
            switch (nb_grid_pts.get_dim()) {
            case oneD: {
                return DynCcoord<MaxDim>{
                    internal::compute_col_major_strides<oneD>(
                        nb_grid_pts.template get<oneD>(),
                        std::make_index_sequence<oneD>{})};
                break;
            }
            case twoD: {
                return DynCcoord<MaxDim>{
                    internal::compute_col_major_strides<twoD>(
                        nb_grid_pts.template get<twoD>(),
                        std::make_index_sequence<twoD>{})};
                break;
            }
            case threeD: {
                return DynCcoord<MaxDim>{
                    internal::compute_col_major_strides<threeD>(
                        nb_grid_pts.template get<threeD>(),
                        std::make_index_sequence<threeD>{})};
                break;
            }
            default:
                throw RuntimeError("unforeseen dimensionality, is it really "
                                   "necessary to have other "
                                   "dimensions than 1, 2, and 3?");
                break;
            }
        }

        /* ----------------------------------------------------------------------
         */
        //! get all strides from a row-major grid
        template <size_t Dim>
        constexpr Ccoord_t<Dim>
        get_row_major_strides(const Ccoord_t<Dim> & nb_grid_pts) {
            return internal::compute_row_major_strides<Dim>(
                nb_grid_pts, std::make_index_sequence<Dim>{});
        }

        /* ----------------------------------------------------------------------
         */
        //! get all strides from a row-major grid
        template <size_t MaxDim>
        constexpr DynCcoord<MaxDim>
        get_row_major_strides(const DynCcoord<MaxDim> & nb_grid_pts) {
            switch (nb_grid_pts.get_dim()) {
            case oneD: {
                return DynCcoord<MaxDim>{
                    internal::compute_row_major_strides<oneD>(
                        nb_grid_pts.template get<oneD>(),
                        std::make_index_sequence<oneD>{})};
                break;
            }
            case twoD: {
                return DynCcoord<MaxDim>{
                    internal::compute_row_major_strides<twoD>(
                        nb_grid_pts.template get<twoD>(),
                        std::make_index_sequence<twoD>{})};
                break;
            }
            case threeD: {
                return DynCcoord<MaxDim>{
                    internal::compute_row_major_strides<threeD>(
                        nb_grid_pts.template get<threeD>(),
                        std::make_index_sequence<threeD>{})};
                break;
            }
            default:
                throw RuntimeError("unforeseen dimensionality, is it really "
                                   "necessary to have other "
                                   "dimensions than 1, 2, and 3?");
                break;
            }
        }

        //------------------------------------------------------------------------//
        //! get the i-th pixel in a grid of size nb_grid_pts
        template <size_t Dim>
        constexpr Ccoord_t<Dim> get_coord(const Ccoord_t<Dim> & nb_grid_pts,
                                          const Ccoord_t<Dim> & locations,
                                          Index_t index) {
            Ccoord_t<Dim> retval{{0}};
            Index_t factor{1};
            for (size_t i{0}; i < Dim; ++i) {
                retval[i] = index / factor % nb_grid_pts[i] + locations[i];
                if (i != Dim - 1) {
                    factor *= nb_grid_pts[i];
                }
            }
            return retval;
        }

        //------------------------------------------------------------------------//
        //! get the i-th pixel in a grid of size nb_grid_pts
        template <size_t Dim, size_t... I>
        constexpr Ccoord_t<Dim> get_coord(const Ccoord_t<Dim> & nb_grid_pts,
                                          const Ccoord_t<Dim> & locations,
                                          Index_t index,
                                          std::index_sequence<I...>) {
            Ccoord_t<Dim> ccoord{get_coord<Dim>(nb_grid_pts, locations, index)};
            return Ccoord_t<Dim>({ccoord[I]...});
        }

        //------------------------------------------------------------------------//
        //! get the i-th pixel in a grid of size nb_grid_pts - specialization
        //! for one dimension
        template <size_t... I>
        constexpr Ccoord_t<1> get_coord(const Ccoord_t<1> & nb_grid_pts,
                                        const Ccoord_t<1> & locations,
                                        Index_t index,
                                        std::index_sequence<I...>) {
            return Ccoord_t<1>({get_coord<1>(nb_grid_pts, locations, index)});
        }

        //! compute the order of the axes given strides, fastest first
        template <class T>
        T compute_axes_order(const T & shape, const T & strides) {
            T axes_order(shape.size());
            std::iota(axes_order.begin(), axes_order.end(), 0);
            std::sort(axes_order.begin(), axes_order.end(),
                      [&shape, &strides](const Dim_t & a, const Dim_t & b) {
                          return (strides[a] == 1 and strides[b] == 1 and
                                  shape[a] < shape[b]) or
                                 strides[a] < strides[b];
                      });
            return axes_order;
        }

        //! compute the order of the axes given strides, fastest first
        template <size_t dim>
        Ccoord_t<dim> compute_axes_order(const Ccoord_t<dim> & shape,
                                         const Ccoord_t<dim> & strides) {
            Ccoord_t<dim> axes_order;
            std::iota(axes_order.begin(), axes_order.end(), 0);
            std::sort(axes_order.begin(), axes_order.end(),
                      [&shape, &strides](const Dim_t & a, const Dim_t & b) {
                          return (strides[a] == 1 and strides[b] == 1 and
                                  shape[a] < shape[b]) or
                                 strides[a] < strides[b];
                      });
            return axes_order;
        }

        //! get the i-th pixel in a grid of size nb_grid_pts, with axes order
        template <size_t dim>
        Ccoord_t<dim> get_coord_from_axes_order(
            const Ccoord_t<dim> & nb_grid_pts, const Ccoord_t<dim> & locations,
            const Ccoord_t<dim> & strides, const Ccoord_t<dim> & axes_order,
            Index_t index) {
            Ccoord_t<dim> retval{{nb_grid_pts[0]}};
            for (Index_t i{dim - 1}; i >= 0; --i) {
                Index_t cur_coord{index / strides[axes_order[i]]};
                retval[axes_order[i]] = cur_coord;
                index -= cur_coord * strides[axes_order[i]];
            }
            for (size_t i{0}; i < dim; ++i) {
                retval[i] += locations[i];
            }
            return retval;
        }

        //! get the i-th pixel in a grid of size nb_grid_pts, with strides
        template <size_t dim>
        Ccoord_t<dim> get_coord_from_strides(const Ccoord_t<dim> & nb_grid_pts,
                                             const Ccoord_t<dim> & locations,
                                             const Ccoord_t<dim> & strides,
                                             Index_t index) {
            return get_coord_from_axes_order(
                nb_grid_pts, locations, strides,
                compute_axes_order(nb_grid_pts, strides), index);
        }

        //! get the i-th pixel in a grid of size nb_grid_pts, with axes order
        //! and location
        template <class T>
        T get_coord0_from_axes_order(const T & nb_grid_pts, const T & strides,
                                     const T & axes_order, Index_t index) {
            auto dim{nb_grid_pts.get_dim()};
            T retval(dim);
            for (Index_t i{dim - 1}; i >= 0; --i) {
                Index_t cur_coord{index / strides[axes_order[i]]};
                retval[axes_order[i]] = cur_coord;
                index -= cur_coord * strides[axes_order[i]];
            }
            return retval;
        }

        //! get the i-th pixel in a grid of size nb_grid_pts, with axes order
        template <class T>
        T get_coord_from_axes_order(const T & nb_grid_pts, const T & locations,
                                    const T & strides, const T & axes_order,
                                    Index_t index) {
            auto dim{nb_grid_pts.get_dim()};
            auto retval{get_coord0_from_axes_order(nb_grid_pts, strides,
                                                   axes_order, index)};
            for (Dim_t i{0}; i < dim; ++i) {
                retval[i] += locations[i];
            }
            return retval;
        }

        //! get the i-th pixel in a grid of size nb_grid_pts, with strides
        template <class T>
        T get_coord_from_strides(const T & nb_grid_pts, const T & locations,
                                 const T & strides, Index_t index) {
            return get_coord_from_axes_order(
                nb_grid_pts, locations, strides,
                compute_axes_order(nb_grid_pts, strides), index);
        }

        //------------------------------------------------------------------------//
        //! get the linear index of a pixel in a column-major grid
        template <size_t Dim>
        constexpr Dim_t get_index(const Ccoord_t<Dim> & nb_grid_pts,
                                  const Ccoord_t<Dim> & locations,
                                  const Ccoord_t<Dim> & ccoord) {
            Dim_t retval{0};
            Dim_t factor{1};
            for (size_t i{0}; i < Dim; ++i) {
                retval += (ccoord[i] - locations[i]) * factor;
                if (i != Dim - 1) {
                    factor *= nb_grid_pts[i];
                }
            }
            return retval;
        }

        //! get the linear index of a pixel in a column-major grid
        Dim_t get_index(const IntCoord_t & nb_grid_pts,
                        const IntCoord_t & locations,
                        const IntCoord_t & ccoord);

        //-----------------------------------------------------------------------//
        //! these functions can be used whenever it is necessary to calculate
        //! the volume of a cell or each pixels of the cell
        template <size_t MaxDim, typename T>
        T compute_volume(const DynCcoord<MaxDim, T> & lengths) {
            T vol{};
            vol++;
            for (auto && length : lengths) {
                vol *= length;
            }
            return vol;
        }

        //! these functions can be used whenever it is necessary to calculate
        //! the volume of a cell or each pixels of the cell
        template <typename T>
        T compute_volume(const std::vector<T> & lengths) {
            T vol{};
            vol++;
            for (auto && length : lengths) {
                vol *= length;
            }
            return vol;
        }

        Real compute_pixel_volume(const IntCoord_t & nb_grid_pts,
                                  const RealCoord_t & lengths);

        //! check whether strides represent a contiguous buffer
        template <class T>
        bool is_buffer_contiguous(const T & nb_grid_pts, const T & strides) {
            Index_t dim{static_cast<Index_t>(nb_grid_pts.size())};
            if (dim == 0) {
                return true;
            }
            if (compute_volume(nb_grid_pts) == 0) {
                return true;
            }
            if (static_cast<Index_t>(strides.size()) != dim) {
                throw RuntimeError(
                    "Mismatch between dimensions of nb_grid_pts and "
                    "strides");
            }
            auto axes{compute_axes_order(nb_grid_pts, strides)};
            Dim_t stride{1};
            bool is_contiguous{true};
            for (Index_t i{0}; i < dim; ++i) {
                is_contiguous &= strides[axes[i]] == stride;
                stride *= nb_grid_pts[axes[i]];
            }
            return is_contiguous;
        }

        //! get the linear index of a pixel given a set of strides
        template <size_t Dim>
        constexpr Index_t
        get_index_from_strides(const Ccoord_t<Dim> & strides,
                               const Ccoord_t<Dim> & locations,
                               const Ccoord_t<Dim> & ccoord) {
            Index_t retval{0};
            for (const auto & tup : akantu::zip(strides, locations, ccoord)) {
                const auto & stride{std::get<0>(tup)};
                const auto & location{std::get<1>(tup)};
                const auto & coord{std::get<2>(tup)};
                retval += stride * (coord - location);
            }
            return retval;
        }

        //! get the linear index of a pixel given a set of strides
        template <class T>
        Index_t get_index_from_strides(const T & strides, const T & locations,
                                       const T & ccoord) {
            const auto dim{strides.size()};
            if (locations.size() != dim) {
                std::stringstream error{};
                error << "Dimension mismatch between strides (dim = " << dim
                      << ") and locations (dim = " << locations.get_dim()
                      << ")";
                throw RuntimeError(error.str());
            }
            if (ccoord.size() != dim) {
                std::stringstream error{};
                error << "Dimension mismatch between strides (dim = " << dim
                      << ") and ccoord (dim = " << ccoord.get_dim() << ")";
                throw RuntimeError(error.str());
            }
            Index_t retval{0};
            for (const auto & tup : akantu::zip(strides, locations, ccoord)) {
                const auto & stride{std::get<0>(tup)};
                const auto & location{std::get<1>(tup)};
                const auto & coord{std::get<2>(tup)};
                retval += stride * (coord - location);
            }
            return retval;
        }

        //-----------------------------------------------------------------------//
        //! get the number of pixels in a grid
        template <size_t Dim>
        constexpr size_t get_size(const Ccoord_t<Dim> & nb_grid_pts) {
            size_t retval{1};
            for (size_t i{0}; i < Dim; ++i) {
                retval *= nb_grid_pts[i];
            }
            return retval;
        }

        //-----------------------------------------------------------------------//
        //! get the number of pixels in a grid
        template <size_t MaxDim>
        size_t get_size(const DynCcoord<MaxDim> & nb_grid_pts) {
            size_t retval{1};
            Dim_t dim{nb_grid_pts.get_dim()};
            for (Dim_t i{0}; i < dim; ++i) {
                retval *= nb_grid_pts[i];
            }
            return retval;
        }

        //-----------------------------------------------------------------------//
        //! get the buffer size required to store a grid given its strides
        template <size_t dim>
        constexpr size_t get_buffer_size(const Ccoord_t<dim> & nb_grid_pts,
                                         const Ccoord_t<dim> & strides) {
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
        size_t get_buffer_size(const IntCoord_t & nb_grid_pts,
                               const IntCoord_t & strides);

        //-----------------------------------------------------------------------//
        //! get the buffer size required to store a grid given its strides
        size_t get_buffer_size(const Shape_t & nb_grid_pts,
                               const Shape_t & strides);

        /**
         * Iteration over square (or cubic) discretisation grids. Duplicates
         * capabilities of `muGrid::CcoordOps::Pixels` without needing to be
         * templated with the spatial dimension. Iteration is slower, though.
         */
        class Pixels {
           public:
            Pixels();

            //! Constructor with default strides (column-major pixel storage
            //! order)
            explicit Pixels(
                const IntCoord_t & nb_subdomain_grid_pts,
                const IntCoord_t & subdomain_locations = IntCoord_t{});

            /**
             * Constructor with custom strides (any, including partially
             * transposed pixel storage order)
             */
            Pixels(const IntCoord_t & nb_subdomain_grid_pts,
                   const IntCoord_t & subdomain_locations,
                   const IntCoord_t & strides);

            //! Constructor with default strides from statically sized coords
            template <size_t Dim>
            explicit Pixels(
                const Ccoord_t<Dim> & nb_subdomain_grid_pts,
                const Ccoord_t<Dim> & subdomain_locations = Ccoord_t<Dim>{});

            //! Constructor with custom strides from statically sized coords
            template <size_t Dim>
            Pixels(const Ccoord_t<Dim> & nb_subdomain_grid_pts,
                   const Ccoord_t<Dim> & subdomain_locations,
                   const Ccoord_t<Dim> & strides);

            //! Copy constructor
            Pixels(const Pixels & other) = default;

            //! Move constructor
            Pixels(Pixels && other) = default;

            //! Destructor
            virtual ~Pixels() = default;

            //! Copy assignment operator
            Pixels & operator=(const Pixels & other) = default;

            //! Move assignment operator
            Pixels & operator=(Pixels && other) = default;

            //! evaluate and return the linear index corresponding to dynamic
            //! `ccoord`
            Index_t get_index(const IntCoord_t & ccoord) const {
                return get_index_from_strides(
                    this->strides, this->subdomain_locations, ccoord);
            }

            //! evaluate and return the linear index corresponding to `ccoord`
            template <size_t Dim>
            Index_t get_index(const Ccoord_t<Dim> & ccoord) const {
                if (this->dim != Dim) {
                    throw RuntimeError("dimension mismatch");
                }
                return get_index_from_strides(
                    this->strides.template get<Dim>(),
                    this->subdomain_locations.template get<Dim>(), ccoord);
            }

            //! return coordinates of the i-th pixel
            IntCoord_t get_coord(const Index_t & index) const {
                return get_coord_from_axes_order(
                    this->nb_subdomain_grid_pts, this->subdomain_locations,
                    this->strides, this->axes_order, index);
            }

            //! return coordinates of the i-th pixel, with zero as location
            IntCoord_t get_coord0(const Index_t & index) const {
                return get_coord0_from_axes_order(this->nb_subdomain_grid_pts,
                                                  this->strides,
                                                  this->axes_order, index);
            }

            IntCoord_t get_neighbour(const IntCoord_t & ccoord,
                                     const IntCoord_t & offset) const {
                return modulo(ccoord + offset - this->subdomain_locations,
                              this->nb_subdomain_grid_pts) +
                       this->subdomain_locations;
            }

            /**
             * Iterator class for `muSpectre::Pixels`
             */
            class iterator {
               public:
                //! stl
                using value_type = IntCoord_t;
                using const_value_type = const value_type;  //!< stl conformance
                using pointer = value_type *;               //!< stl conformance
                using difference_type = std::ptrdiff_t;     //!< stl conformance
                using iterator_category = std::forward_iterator_tag;
                //!< stl
                //!< conformance

                //! constructor
                iterator(const Pixels & pixels, Size_t index)
                    : pixels{pixels}, coord0{pixels.get_coord0(index)} {}

                //! constructor
                iterator(const Pixels & pixels, IntCoord_t coord0)
                    : pixels{pixels}, coord0{coord0} {}

                //! Default constructor
                iterator() = delete;

                //! Copy constructor
                iterator(const iterator & other) = default;

                //! Move constructor
                iterator(iterator && other) = default;

                //! Destructor
                ~iterator() = default;

                //! Copy assignment operator
                iterator & operator=(const iterator & other) = delete;

                //! Move assignment operator
                iterator & operator=(iterator && other) = delete;

                //! dereferencing
                value_type operator*() const {
                    return this->pixels.subdomain_locations + this->coord0;
                }

                //! pre-increment
                iterator & operator++() {
                    auto axis{this->pixels.axes_order[0]};
                    // Increase fastest index
                    ++this->coord0[axis];
                    // Check whether coordinate is out of bounds
                    Index_t aindex{0};
                    while (aindex < this->pixels.dim - 1 &&
                           this->coord0[axis] >=
                               this->pixels.nb_subdomain_grid_pts[axis]) {
                        this->coord0[axis] = 0;
                        // Get next fastest axis
                        axis = this->pixels.axes_order[++aindex];
                        ++this->coord0[axis];
                    }
                    return *this;
                }

                //! inequality
                bool operator!=(const iterator & other) const {
                    return this->coord0 != other.coord0;
                }

                //! equality
                bool operator==(const iterator & other) const {
                    return not(*this != other);
                }

               protected:
                const Pixels & pixels;  //!< ref to pixels in cell
                IntCoord_t coord0;      //!< coordinate of current pixel
            };

            //! stl conformance
            iterator begin() const { return iterator(*this, 0); }

            //! stl conformance
            iterator end() const {
                return ++iterator(*this, this->nb_subdomain_grid_pts - 1);
            }

            //! stl conformance
            size_t size() const {
                return get_size(this->nb_subdomain_grid_pts);
            }

            //! buffer size, including padding
            size_t buffer_size() const {
                return get_buffer_size(this->nb_subdomain_grid_pts,
                                       this->strides);
            }

            //! return spatial dimension
            Dim_t get_dim() const { return this->dim; }

            //! return the resolution of the discretisation grid in each spatial
            //! dim
            const IntCoord_t & get_nb_subdomain_grid_pts() const {
                return this->nb_subdomain_grid_pts;
            }

            /**
             * return the ccoordinates of the bottom, left, (front) pixel/voxel
             * of this processors partition of the discretisation grid. For
             * sequential calculations, this is alvays the origin
             */
            const IntCoord_t & get_subdomain_locations() const {
                return this->subdomain_locations;
            }

            //! return the strides used for iterating over the pixels
            const IntCoord_t & get_strides() const { return this->strides; }

            /**
             * enumerator class for `muSpectre::Pixels`
             */
            class Enumerator final {
               public:
                //! Default constructor
                Enumerator() = delete;

                //! Constructor
                explicit Enumerator(const Pixels & pixels) : pixels{pixels} {}

                //! Copy constructor
                Enumerator(const Enumerator & other) = default;

                //! Move constructor
                Enumerator(Enumerator && other) = default;

                //! Destructor
                virtual ~Enumerator() = default;

                //! Copy assignment operator
                Enumerator & operator=(const Enumerator & other) = delete;

                //! Move assignment operator
                Enumerator & operator=(Enumerator && other) = delete;

                /**
                 * @class iterator
                 * @brief A derived class from Pixels::iterator, used for
                 * iterating over Pixels.
                 *
                 * This class is a final class, meaning it cannot be further
                 * derived from. It provides a custom implementation of the
                 * dereference operator (*).
                 *
                 * @tparam Parent Alias for the base class Pixels::iterator.
                 *
                 * @note The using Parent::Parent; statement is a C++11 feature
                 * called "Inheriting Constructors" which means that this
                 * derived class will have the same constructors as the base
                 * class.
                 */
                class iterator final : public Pixels::iterator {
                   public:
                    using Parent = Pixels::iterator;
                    using Parent::Parent;

                    /**
                     * @brief Overloaded dereference operator (*).
                     *
                     * This function returns a tuple containing the index of the
                     * pixel and the pixel's coordinates.
                     *
                     * @return std::tuple<Index_t, Parent::value_type> A tuple
                     * containing the index of the pixel and the pixel's
                     * coordinates.
                     */
                    std::tuple<Index_t, Parent::value_type> operator*() const {
                        auto && pixel{this->Parent::operator*()};
                        return std::tuple<Index_t, Parent::value_type>{
                            this->pixels.get_index(pixel), pixel};
                    }
                };

                //! stl conformance
                iterator begin() const { return iterator{this->pixels, 0}; }

                //! stl conformance
                iterator end() const {
                    iterator it{this->pixels,
                                this->pixels.nb_subdomain_grid_pts - 1};
                    ++it;
                    return it;
                }

                //! stl conformance
                size_t size() const { return this->pixels.size(); }

                size_t buffer_size() const {
                    return this->pixels.buffer_size();
                }

               protected:
                const Pixels & pixels;
            };

            /**
             * iterates in tuples of pixel index ond coordinate. Useful in
             * parallel problems, where simple enumeration of the pixels would
             * be incorrect
             */
            Enumerator enumerate() const { return Enumerator(*this); }

           protected:
            Dim_t dim;                         //!< spatial dimension
            IntCoord_t nb_subdomain_grid_pts;  //!< nb_grid_pts of this domain
            IntCoord_t subdomain_locations;    //!< locations of this domain
            IntCoord_t strides;                //!< strides of memory layout
            IntCoord_t axes_order;             //!< order of axes
            bool contiguous;                   //!< is this a contiguous buffer?
        };
    }  // namespace CcoordOps
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CCOORD_OPERATIONS_HH_
