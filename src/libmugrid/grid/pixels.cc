/**
 * @file   pixels.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Oct 2019
 *
 * @brief  pre-compilable pixel operations
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
#include <iostream>

#include "core/exception.hh"
#include "grid/pixels.hh"

namespace muGrid {

    namespace CcoordOps {
        size_t get_buffer_size(const DynGridIndex & nb_grid_pts,
                               const DynGridIndex & strides) {
            const Dim_t & dim{nb_grid_pts.get_dim()};
            if (strides.get_dim() != dim) {
                std::stringstream error{};
                error << "Dimension mismatch between nb_grid_pts (= "
                      << nb_grid_pts << ") and strides (= " << strides << ")";
                throw RuntimeError(error.str());
            }
            size_t buffer_size{0};
            // We need to loop over the dimensions because the largest stride
            // can occur anywhere. (It depends on the storage order.)
            for (Dim_t i{0}; i < dim; ++i) {
                buffer_size =
                    std::max(buffer_size,
                             static_cast<size_t>(nb_grid_pts[i] * strides[i]));
            }
            return buffer_size;
        }

        size_t get_buffer_size(const Shape_t & nb_grid_pts,
                               const Shape_t & strides) {
            const size_t & dim{nb_grid_pts.size()};
            if (strides.size() != dim) {
                std::stringstream error{};
                error << "Dimension mismatch between nb_grid_pts (= "
                      << nb_grid_pts << ") and strides (= " << strides << ")";
                throw RuntimeError(error.str());
            }
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

        /* ----------------------------------------------------------------------
         */
        Pixels::Pixels()
            : dim{}, nb_subdomain_grid_pts{}, subdomain_locations{}, strides{},
              axes_order{}, contiguous{false} {}

        /* ----------------------------------------------------------------------
         */
        Pixels::Pixels(const DynGridIndex & nb_subdomain_grid_pts,
                       const DynGridIndex & subdomain_locations)
            : dim(nb_subdomain_grid_pts.get_dim()),
              nb_subdomain_grid_pts(nb_subdomain_grid_pts),
              subdomain_locations{
                  subdomain_locations.get_dim() == 0
                      ? DynGridIndex(nb_subdomain_grid_pts.get_dim())
                      : subdomain_locations},
              strides(get_col_major_strides(nb_subdomain_grid_pts)),
              axes_order{
                  compute_axes_order(nb_subdomain_grid_pts, this->strides)},
              contiguous{true} {
            if (this->dim != this->subdomain_locations.get_dim()) {
                std::stringstream error{};
                error << "Dimension mismatch between nb_subdomain_grid_pts (= "
                      << nb_subdomain_grid_pts
                      << ") and subdomain_locations (= " << subdomain_locations
                      << ")";
                throw RuntimeError(error.str());
            }
        }

        /* ----------------------------------------------------------------------
         */
        Pixels::Pixels(const DynGridIndex & nb_subdomain_grid_pts,
                       const DynGridIndex & subdomain_locations,
                       const DynGridIndex & strides)
            : dim(nb_subdomain_grid_pts.get_dim()),
              nb_subdomain_grid_pts(nb_subdomain_grid_pts),
              subdomain_locations{
                  subdomain_locations.get_dim() == 0
                      ? DynGridIndex(nb_subdomain_grid_pts.get_dim())
                      : subdomain_locations},
              strides{strides},
              axes_order{compute_axes_order(nb_subdomain_grid_pts, strides)},
              contiguous{is_buffer_contiguous(nb_subdomain_grid_pts, strides)} {
            if (this->dim != this->subdomain_locations.get_dim()) {
                std::stringstream error{};
                error << "Dimension mismatch between nb_subdomain_grid_pts (= "
                      << nb_subdomain_grid_pts
                      << ") and subdomain_locations (= " << subdomain_locations
                      << ")";
                throw RuntimeError(error.str());
            }
            if (this->dim != this->strides.get_dim()) {
                std::stringstream error{};
                error << "Dimension mismatch between subdomain_locations (= "
                      << subdomain_locations << ") and strides (= " << strides
                      << ")";
                throw RuntimeError(error.str());
            }
        }

        /* ----------------------------------------------------------------------
         */
        template <size_t Dim>
        Pixels::Pixels(const GridIndex<Dim> & nb_subdomain_grid_pts,
                       const GridIndex<Dim> & subdomain_locations)
            : dim(Dim), nb_subdomain_grid_pts(nb_subdomain_grid_pts),
              subdomain_locations(subdomain_locations),
              strides(get_col_major_strides(nb_subdomain_grid_pts)),
              axes_order{compute_axes_order(DynGridIndex{nb_subdomain_grid_pts},
                                            this->strides)},
              contiguous{true} {}

        /* ----------------------------------------------------------------------
         */
        template <size_t Dim>
        Pixels::Pixels(const GridIndex<Dim> & nb_subdomain_grid_pts,
                       const GridIndex<Dim> & subdomain_locations,
                       const GridIndex<Dim> & strides)
            : dim(Dim), nb_subdomain_grid_pts(nb_subdomain_grid_pts),
              subdomain_locations(subdomain_locations), strides{strides},
              axes_order{compute_axes_order(DynGridIndex{nb_subdomain_grid_pts},
                                            DynGridIndex{strides})},
              contiguous{is_buffer_contiguous(DynGridIndex{nb_subdomain_grid_pts},
                                              DynGridIndex{strides})} {}

        template Pixels::Pixels(const GridIndex<oneD> &, const GridIndex<oneD> &);
        template Pixels::Pixels(const GridIndex<twoD> &, const GridIndex<twoD> &);
        template Pixels::Pixels(const GridIndex<threeD> &,
                                const GridIndex<threeD> &);
        template Pixels::Pixels(const GridIndex<oneD> &, const GridIndex<oneD> &,
                                const GridIndex<oneD> &);
        template Pixels::Pixels(const GridIndex<twoD> &, const GridIndex<twoD> &,
                                const GridIndex<twoD> &);
        template Pixels::Pixels(const GridIndex<threeD> &,
                                const GridIndex<threeD> &,
                                const GridIndex<threeD> &);
    }  // namespace CcoordOps

}  // namespace muGrid
