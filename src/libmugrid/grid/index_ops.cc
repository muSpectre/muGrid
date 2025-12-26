/**
 * @file   ccoord_operations.cc
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
#include "grid/index_ops.hh"

namespace muGrid {

    namespace CcoordOps {

        //------------------------------------------------------------------------//
        Dim_t get_index(const DynGridIndex & nb_grid_pts,
                        const DynGridIndex & locations,
                        const DynGridIndex & ccoord) {
            const Dim_t dim{nb_grid_pts.get_dim()};
            if (locations.get_dim() != dim) {
                std::stringstream error{};
                error << "Dimension mismatch between nb_grid_pts (= "
                      << nb_grid_pts << ") and locations (= " << locations
                      << ")";
                throw RuntimeError(error.str());
            }
            if (ccoord.get_dim() != dim) {
                std::stringstream error{};
                error << "Dimension mismatch between nb_grid_pts (= "
                      << nb_grid_pts << ") and locations (= " << locations
                      << ")";
                throw RuntimeError(error.str());
            }
            Dim_t retval{0};
            Dim_t factor{1};
            for (Dim_t i = 0; i < dim; ++i) {
                retval += (ccoord[i] - locations[i]) * factor;
                if (i != dim - 1) {
                    factor *= nb_grid_pts[i];
                }
            }
            return retval;
        }

        //-----------------------------------------------------------------------//
        Real compute_pixel_volume(const DynGridIndex & nb_grid_pts,
                                  const DynCoord<fourD, Real> & lengths) {
            Real vol{1.0};
            for (auto && tup : akantu::zip(nb_grid_pts, lengths)) {
                auto && nb_grid_pt{std::get<0>(tup)};
                auto && length{std::get<1>(tup)};
                vol *= (length / nb_grid_pt);
            }
            return vol;
        }
    }  // namespace CcoordOps

}  // namespace muGrid
