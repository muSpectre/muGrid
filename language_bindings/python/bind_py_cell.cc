/**
 * @file   bind_py_cell.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   09 Jan 2018
 *
 * @brief  Python bindings for the cell factory function
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#include "common/muSpectre_common.hh"
#include <libmugrid/ccoord_operations.hh>
#include "cell/cell_factory.hh"
#include "cell/cell_base.hh"

#ifdef WITH_FFTWMPI
#include "projection/fftwmpi_engine.hh"
#endif
#ifdef WITH_PFFT
#include "projection/pfft_engine.hh"
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "pybind11/eigen.h"

#include <sstream>
#include <memory>

using muSpectre::Ccoord_t;
using muSpectre::Dim_t;
using muSpectre::Formulation;
using muSpectre::Rcoord_t;
using pybind11::literals::operator""_a;
namespace py = pybind11;

/**
 * cell factory for specific FFT engine
 */
#ifdef WITH_MPI
template <Dim_t dim, class FFTEngine>
void add_parallel_cell_factory_helper(py::module & mod, const char * name) {
  using Ccoord = Ccoord_t<dim>;
  using Rcoord = Rcoord_t<dim>;

  mod.def(name,
          [](Ccoord res, Rcoord lens, Formulation form, size_t comm) {
            return make_parallel_cell<dim, dim, CellBase<dim, dim>, FFTEngine>(
                std::move(res), std::move(lens), std::move(form),
                std::move(Communicator(MPI_Comm(comm))));
          },
          "resolutions"_a, "lengths"_a = CcoordOps::get_cube<dim>(1.),
          "formulation"_a = Formulation::finite_strain,
          "communicator"_a = size_t(MPI_COMM_SELF));
}
#endif

/**
 * the cell factory is only bound for default template parameters
 */
template <Dim_t dim>
void add_cell_factory_helper(py::module & mod) {
  using Ccoord = Ccoord_t<dim>;
  using Rcoord = Rcoord_t<dim>;

  mod.def("CellFactory",
          [](Ccoord res, Rcoord lens, Formulation form) {
            return make_cell(std::move(res), std::move(lens), std::move(form));
          },
          "resolutions"_a,
          "lengths"_a = muSpectre::CcoordOps::get_cube<dim>(1.),
          "formulation"_a = Formulation::finite_strain);

#ifdef WITH_FFTWMPI
  add_parallel_cell_factory_helper<dim, FFTWMPIEngine<dim>>(
      mod, "FFTWMPICellFactory");
#endif

#ifdef WITH_PFFT
  add_parallel_cell_factory_helper<dim, PFFTEngine<dim>>(mod,
                                                         "PFFTCellFactory");
#endif
}

void add_cell_factory(py::module & mod) {
  add_cell_factory_helper<muSpectre::twoD>(mod);
  add_cell_factory_helper<muSpectre::threeD>(mod);
}

/**
 * CellBase for which the material and spatial dimension are identical
 */
template <Dim_t dim>
void add_cell_base_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "CellBase" << dim << 'd';
  const std::string name = name_stream.str();
  using sys_t = muSpectre::CellBase<dim, dim>;
  py::class_<sys_t, muSpectre::Cell>(mod, name.c_str())
      .def("__len__", &sys_t::size)
      .def("__iter__",
           [](sys_t & s) { return py::make_iterator(s.begin(), s.end()); })
      .def("initialise", &sys_t::initialise,
           "flags"_a = muSpectre::FFT_PlanFlags::estimate)
      .def(
          "directional_stiffness",
          [](sys_t & cell, py::EigenDRef<Eigen::ArrayXXd> & v) {
            if ((size_t(v.cols()) != cell.size() ||
                 size_t(v.rows()) != dim * dim)) {
              std::stringstream err{};
              err << "need array of shape (" << dim * dim << ", " << cell.size()
                  << ") but got (" << v.rows() << ", " << v.cols() << ").";
              throw std::runtime_error(err.str());
            }
            if (!cell.is_initialised()) {
              cell.initialise();
            }
            const std::string out_name{"temp output for directional stiffness"};
            const std::string in_name{"temp input for directional stiffness"};
            constexpr bool create_tangent{true};
            auto & K = cell.get_tangent(create_tangent);
            auto & input = cell.get_managed_T2_field(in_name);
            auto & output = cell.get_managed_T2_field(out_name);
            input.eigen() = v;
            cell.directional_stiffness(K, input, output);
            return output.eigen();
          },
          "δF"_a)
      .def("project",
           [](sys_t & cell, py::EigenDRef<Eigen::ArrayXXd> & v) {
             if ((size_t(v.cols()) != cell.size() ||
                  size_t(v.rows()) != dim * dim)) {
               std::stringstream err{};
               err << "need array of shape (" << dim * dim << ", "
                   << cell.size() << ") but got (" << v.rows() << ", "
                   << v.cols() << ").";
               throw std::runtime_error(err.str());
             }
             if (!cell.is_initialised()) {
               cell.initialise();
             }
             const std::string in_name{"temp input for projection"};
             auto & input = cell.get_managed_T2_field(in_name);
             input.eigen() = v;
             cell.project(input);
             return input.eigen();
           },
           "field"_a)
      .def("get_strain", [](sys_t & s) { return s.get_strain().eigen(); },
           py::return_value_policy::reference_internal)
      .def("get_stress",
           [](sys_t & s) { return Eigen::ArrayXXd(s.get_stress().eigen()); })
      .def_property_readonly("size", &sys_t::size)
      .def("evaluate_stress_tangent",
           [](sys_t & cell, py::EigenDRef<Eigen::ArrayXXd> & v) {
             if ((size_t(v.cols()) != cell.size() ||
                  size_t(v.rows()) != dim * dim)) {
               std::stringstream err{};
               err << "need array of shape (" << dim * dim << ", "
                   << cell.size() << ") but got (" << v.rows() << ", "
                   << v.cols() << ").";
               throw std::runtime_error(err.str());
             }
             auto & strain{cell.get_strain()};
             strain.eigen() = v;
             auto stress_tgt{cell.evaluate_stress_tangent(strain)};
             return std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd>(
                 std::get<0>(stress_tgt).eigen(),
                 std::get<1>(stress_tgt).eigen());
           },
           "strain"_a)
      .def("evaluate_stress",
           [](sys_t & cell, py::EigenDRef<Eigen::ArrayXXd> & v) {
             if ((size_t(v.cols()) != cell.size() ||
                  size_t(v.rows()) != dim * dim)) {
               std::stringstream err{};
               err << "need array of shape (" << dim * dim << ", "
                   << cell.size() << ") but got (" << v.rows() << ", "
                   << v.cols() << ").";
               throw std::runtime_error(err.str());
             }
             auto & strain{cell.get_strain()};
             strain.eigen() = v;
             return cell.evaluate_stress();
           },
           "strain"_a, py::return_value_policy::reference_internal)
      .def("get_projection", &sys_t::get_projection)
      .def("get_subdomain_resolutions", &sys_t::get_subdomain_resolutions)
      .def("get_subdomain_locations", &sys_t::get_subdomain_locations)
      .def("get_domain_resolutions", &sys_t::get_domain_resolutions)
      .def("get_domain_lengths", &sys_t::get_domain_resolutions)
      .def("set_uniform_strain",
           [](sys_t & cell, py::EigenDRef<Eigen::ArrayXXd> & v) -> void {
             cell.set_uniform_strain(v);
           },
           "strain"_a)
      .def("save_history_variables", &sys_t::save_history_variables);
}

void add_cell_base(py::module & mod) {
  py::class_<muSpectre::Cell>(mod, "Cell")
      .def("get_globalised_internal_real_array",
           &muSpectre::Cell::get_globalised_internal_real_array,
           "unique_name"_a,
           "Convenience function to copy local (internal) fields of "
           "materials into a global field. At least one of the materials in "
           "the cell needs to contain an internal field named "
           "`unique_name`. If multiple materials contain such a field, they "
           "all need to be of same scalar type and same number of "
           "components. This does not work for split pixel cells or "
           "laminate pixel cells, as they can have multiple entries for the "
           "same pixel. Pixels for which no field named `unique_name` "
           "exists get an array of zeros."
           "\n"
           "Parameters:\n"
           "unique_name: fieldname to fill the global field with. At "
           "least one material must have such a field, or an "
           "Exception is raised.")
      .def("get_globalised_current_real_array",
           &muSpectre::Cell::get_globalised_current_real_array, "unique_name"_a)
      .def("get_globalised_old_real_array",
           &muSpectre::Cell::get_globalised_old_real_array, "unique_name"_a,
           "nb_steps_ago"_a = 1)
      .def("get_managed_real_array", &muSpectre::Cell::get_managed_real_array,
           "unique_name"_a, "nb_components"_a,
           "returns a field or nb_components real numbers per pixel");
  add_cell_base_helper<muSpectre::twoD>(mod);
  add_cell_base_helper<muSpectre::threeD>(mod);
}

void add_cell(py::module & mod) {
  add_cell_factory(mod);

  auto cell{mod.def_submodule("cell")};
  cell.doc() = "bindings for cells and cell factories";
  add_cell_base(cell);
}
