/**
 * @file   grid/index_ops.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   29 Sep 2017
 *
 * @brief  Index and coordinate conversion operations for grid addressing
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

#ifndef SRC_LIBMUGRID_GRID_INDEX_OPS_HH_
#define SRC_LIBMUGRID_GRID_INDEX_OPS_HH_

#include "core/types.hh"
#include "core/coordinates.hh"
#include "core/exception.hh"
#include "grid/iterators.hh"
#include "strides.hh"

#include "Eigen/Dense"

#include <cassert>
#include <sstream>
#include <vector>

namespace muGrid {
    namespace CcoordOps {
        //! modulo operator that can handle negative values
        template <typename T>
        T modulo(const T & a, const T & b) {
            return (b + (a % b)) % b;
        }

        //------------------------------------------------------------------------//
        //! get the i-th pixel in a grid of size nb_grid_pts
        template <size_t Dim>
        constexpr GridIndex<Dim> get_coord(const GridIndex<Dim> & nb_grid_pts,
                                          const GridIndex<Dim> & locations,
                                          Index_t index) {
            GridIndex<Dim> retval{{0}};
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
        constexpr GridIndex<Dim> get_coord(const GridIndex<Dim> & nb_grid_pts,
                                          const GridIndex<Dim> & locations,
                                          Index_t index,
                                          std::index_sequence<I...>) {
            GridIndex<Dim> ccoord{get_coord<Dim>(nb_grid_pts, locations, index)};
            return GridIndex<Dim>({ccoord[I]...});
        }

        //------------------------------------------------------------------------//
        //! get the i-th pixel in a grid of size nb_grid_pts - specialization
        //! for one dimension
        template <size_t... I>
        constexpr GridIndex<1> get_coord(const GridIndex<1> & nb_grid_pts,
                                        const GridIndex<1> & locations,
                                        Index_t index,
                                        std::index_sequence<I...>) {
            return GridIndex<1>({get_coord<1>(nb_grid_pts, locations, index)});
        }

        //! get the i-th pixel in a grid of size nb_grid_pts, with axes order
        template <size_t dim>
        GridIndex<dim> get_coord_from_axes_order(
            const GridIndex<dim> & nb_grid_pts, const GridIndex<dim> & locations,
            const GridIndex<dim> & strides, const GridIndex<dim> & axes_order,
            Index_t index) {
            GridIndex<dim> retval{{nb_grid_pts[0]}};
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
        GridIndex<dim> get_coord_from_strides(const GridIndex<dim> & nb_grid_pts,
                                             const GridIndex<dim> & locations,
                                             const GridIndex<dim> & strides,
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
        constexpr Dim_t get_index(const GridIndex<Dim> & nb_grid_pts,
                                  const GridIndex<Dim> & locations,
                                  const GridIndex<Dim> & ccoord) {
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
        Dim_t get_index(const DynGridIndex & nb_grid_pts,
                        const DynGridIndex & locations,
                        const DynGridIndex & ccoord);

        //-----------------------------------------------------------------------//
        //! these functions can be used whenever it is necessary to calculate
        //! the volume of a cell or each pixels of the cell
        template <size_t MaxDim, typename T>
        T compute_volume(const DynCoord<MaxDim, T> & lengths) {
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

        Real compute_pixel_volume(const DynGridIndex & nb_grid_pts,
                                  const DynCoord<fourD, Real> & lengths);

        //! get the linear index of a pixel given a set of strides
        template <size_t Dim>
        constexpr Index_t
        get_index_from_strides(const GridIndex<Dim> & strides,
                               const GridIndex<Dim> & locations,
                               const GridIndex<Dim> & ccoord) {
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

    }  // namespace CcoordOps
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_GRID_INDEX_OPS_HH_
