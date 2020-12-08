/**
 * @file   bind_py_cell_data.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   04 Sep 2020
 *
 * @brief  Python bindings for the cell data class
 *
 * Copyright © 2020 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
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

#include "cell/cell_data.hh"
#include "solver/matrix_adaptor.hh"

#include <libmufft/fft_engine_base.hh>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

using muSpectre::CellData;
using CellData_ptr = std::shared_ptr<CellData>;
using muSpectre::Real;
using pybind11::literals::operator""_a;
namespace py = pybind11;

/* ---------------------------------------------------------------------- */
void add_cell_data_helper(py::module & mod) {
  py::class_<CellData, std::shared_ptr<CellData>>(mod, "CellData")
      .def_static(
          "make",
          [](const muSpectre::DynCcoord_t & nb_domain_grid_pts,
             const muGrid::DynRcoord_t & domain_lenghts) -> CellData_ptr {
            return CellData::make(nb_domain_grid_pts, domain_lenghts);
          },
          "nb_domain_grid_pts"_a, "domain_lenghts"_a)
      .def_property_readonly(
          "fields",
          [](CellData & cell_data) -> muGrid::GlobalFieldCollection & {
            return cell_data.get_fields();
          },
          py::return_value_policy::reference_internal)
      .def("add_material", &CellData::add_material, "material"_a,
           py::return_value_policy::reference_internal)
      .def_property_readonly("spatial_dim", &CellData::get_spatial_dim)
      .def_property_readonly("material_dim", &CellData::get_material_dim)
      .def_property("nb_quad_pts", &CellData::get_nb_quad_pts,
                    &CellData::set_nb_quad_pts)
      .def_property("nb_nodal_pts", &CellData::get_nb_nodal_pts,
                    &CellData::set_nb_nodal_pts)
      .def_property_readonly("has_nb_quad_pts", &CellData::has_nb_quad_pts)
      .def_property_readonly("has_nb_nodal_pts", &CellData::has_nb_nodal_pts)
      .def_property_readonly(
          "pixels",
          [](const CellData & cell_data)
              -> const muGrid::CcoordOps::DynamicPixels & {
            return cell_data.get_pixels();
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly("quad_pt_indices", &CellData::get_quad_pt_indices,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("pixel_indices", &CellData::get_pixel_indices,
                             py::return_value_policy::reference_internal)
      .def_property_readonly(
          "nb_domain_grid_pts",
          [](const CellData & cell_data) -> muGrid::DynCcoord_t {
            return cell_data.get_nb_domain_grid_pts();
          })
      .def_property_readonly(
          "nb_subdomain_grid_pts",
          [](const CellData & cell_data) -> muGrid::DynCcoord_t {
            return cell_data.get_nb_subdomain_grid_pts();
          })
      .def_property_readonly(
          "subdomain_locations",
          [](const CellData & cell_data) -> muGrid::DynCcoord_t {
            return cell_data.get_subdomain_locations();
          })
      .def_property_readonly(
          "domain_lengths",
          [](const CellData & cell_data) -> muGrid::DynRcoord_t {
            return cell_data.get_domain_lengths();
          })
      .def_property_readonly("FFT_engine", &CellData::get_FFT_engine)
      .def("save_history_variables", &CellData::save_history_variables);
}

void add_cell_data(py::module & mod) {
  add_cell_data_helper(mod);
}
