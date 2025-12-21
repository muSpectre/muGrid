/**
* @file   python_helpers.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   10 Apr 2025
 *
 * @brief  Helper classes for Python bindings
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

#ifndef SRC_LIBMUGRID_PYTHON_HELPERS_HH_
#define SRC_LIBMUGRID_PYTHON_HELPERS_HH_

#include <pybind11/eigen.h>

#include "grid/index_ops.hh"
#include "core/coordinates.hh"

namespace py = pybind11;

namespace muGrid {
    template<typename T>
    T normalize_coord(Int coord, Int length) {
        return static_cast<T>(CcoordOps::modulo(coord, length)) / length;
    }

    template<>
    inline Int normalize_coord<Int>(Int coord, Int length) {
        return CcoordOps::modulo(coord, length);
    }

    template<typename T, bool with_ghosts, class C>
    auto py_coords(const C &field_like) {
        auto &fc{field_like.get_collection()};
        if (fc.get_domain() != FieldCollection::ValidityDomain::Global) {
            throw RuntimeError("Coordinates can only be computed for global fields");
        }

        auto &gfc{dynamic_cast<const GlobalFieldCollection &>(fc)};

        std::vector<Index_t> shape{};
        const Index_t dim{field_like.get_spatial_dim()};
        shape.push_back(dim);
        auto nb_subdomain_grid_pts{gfc.get_nb_subdomain_grid_pts_without_ghosts()};
        if (with_ghosts) {
            nb_subdomain_grid_pts = gfc.get_nb_subdomain_grid_pts_with_ghosts();
        }
        for (auto &&n: nb_subdomain_grid_pts) {
            shape.push_back(n);
        }
        py::array_t<T, py::array::f_style> coords(shape);
        const auto &nb_domain_grid_pts{gfc.get_nb_domain_grid_pts()};
        auto subdomain_locations{gfc.get_subdomain_locations_without_ghosts()};
        if (with_ghosts) {
            subdomain_locations = gfc.get_subdomain_locations_with_ghosts();
        }
        const auto nb_subdomain_pixels{
            CcoordOps::get_size(nb_subdomain_grid_pts)
        };
        T *ptr{static_cast<T *>(coords.request().ptr)};
        for (size_t k{0}; k < nb_subdomain_pixels; ++k) {
            *ptr = normalize_coord<T>(k % nb_subdomain_grid_pts[0] +
                                      subdomain_locations[0],
                                      nb_domain_grid_pts[0]);
            ptr++;
            size_t yz{k};
            for (int i = 1; i < dim; ++i) {
                yz /= nb_subdomain_grid_pts[i - 1];
                *ptr = normalize_coord<T>(yz % nb_subdomain_grid_pts[i] +
                                          subdomain_locations[i],
                                          nb_domain_grid_pts[i]);
                ptr++;
            }
        }
        return coords;
    }
} // namespace muGrid

#endif  // SRC_LIBMUGRID_PYTHON_HELPERS_HH_
