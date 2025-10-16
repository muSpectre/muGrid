/**
 * @file   bind_py_file_io.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   20 Oct 2024
 *
 * @brief  Utility functions for testing the numpy interface.
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

#include "libmugrid/field_collection_global.hh"
#include "libmugrid/numpy_tools.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pybind11/eigen.h>

using muGrid::Real;
using muGrid::Index_t;
using muGrid::IterUnit;
using muGrid::GlobalFieldCollection;
using muGrid::NumpyProxy;
using pybind11::literals::operator""_a;

using muGrid::operator<<;

namespace py = pybind11;

void add_testing(py::module & mod) {
  mod.def(
      "test_numpy_copy",
      [](GlobalFieldCollection & fc, py::array_t<Real> & input_array) {
        const py::buffer_info & info = input_array.request();
        auto & dim{fc.get_spatial_dim()};
        if (info.shape.size() < static_cast<size_t>(dim)) {
          std::stringstream s;
          s << "Input array has " << info.shape.size() << " dimensions "
            << "but the field collection was set up for " << dim
            << " dimensions.";
          throw std::runtime_error(s.str());
        }
        auto nb_dof_per_pixel{std::accumulate(info.shape.begin(),
                                              info.shape.end() - dim, 1,
                                              std::multiplies<Index_t>())};
        NumpyProxy<Real> input_proxy(
            fc.get_nb_domain_grid_pts(), fc.get_nb_subdomain_grid_pts_with_ghosts(),
            fc.get_subdomain_locations_with_ghosts(), nb_dof_per_pixel, input_array);
        return numpy_copy(input_proxy.get_field(), input_proxy.get_iter_type());
      },
      "fc"_a, "input_array"_a,
      "Test numpy wrapper.");
}