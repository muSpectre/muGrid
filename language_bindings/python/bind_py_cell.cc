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
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "cell/cell_factory.hh"
#include "cell/cell_base.hh"

#ifdef WITH_FFTWMPI
#include "fft/fftwmpi_engine.hh"
#endif
#ifdef WITH_PFFT
#include "fft/pfft_engine.hh"
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/eigen.h"

#include <sstream>
#include <memory>

using namespace muSpectre;
namespace py=pybind11;
using namespace pybind11::literals;

/**
 * cell factory for specific FFT engine
 */
#ifdef WITH_MPI
template <Dim_t dim, class FFTEngine>
void add_parallel_cell_factory_helper(py::module & mod,
                                      const char *name) {
  using Ccoord = Ccoord_t<dim>;
  using Rcoord = Rcoord_t<dim>;

  mod.def
    (name,
     [](Ccoord res, Rcoord lens, Formulation form, size_t comm) {
       return make_parallel_cell
         <dim, dim, CellBase<dim, dim>, FFTEngine>
         (std::move(res), std::move(lens), std::move(form),
          std::move(Communicator(MPI_Comm(comm))));
     },
     "resolutions"_a,
     "lengths"_a=CcoordOps::get_cube<dim>(1.),
     "formulation"_a=Formulation::finite_strain,
     "communicator"_a=size_t(MPI_COMM_SELF));
}
#endif

/**
 * the cell factory is only bound for default template parameters
 */
template <Dim_t dim>
void add_cell_factory_helper(py::module & mod) {
  using Ccoord = Ccoord_t<dim>;
  using Rcoord = Rcoord_t<dim>;

  mod.def
    ("CellFactory",
     [](Ccoord res, Rcoord lens, Formulation form) {
       return make_cell(std::move(res), std::move(lens), std::move(form));
     },
     "resolutions"_a,
     "lengths"_a=CcoordOps::get_cube<dim>(1.),
     "formulation"_a=Formulation::finite_strain);

#ifdef WITH_FFTWMPI
  add_parallel_cell_factory_helper<dim, FFTWMPIEngine<dim, dim>>(
    mod, "FFTWMPICellFactory");
#endif

#ifdef WITH_PFFT
  add_parallel_cell_factory_helper<dim, PFFTEngine<dim, dim>>(
    mod, "PFFTCellFactory");
#endif
}

void add_cell_factory(py::module & mod) {
  add_cell_factory_helper<twoD  >(mod);
  add_cell_factory_helper<threeD>(mod);
}

/**
 * CellBase for which the material and spatial dimension are identical
 */
template <Dim_t dim>
void add_cell_base_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "CellBase" << dim << 'd';
  const std::string name = name_stream.str();
  using sys_t = CellBase<dim, dim>;
  py::class_<sys_t>(mod, name.c_str())
    .def("__len__", &sys_t::size)
    .def("__iter__", [](sys_t & s) {
        return py::make_iterator(s.begin(), s.end());
      })
    .def("initialise", &sys_t::initialise, "flags"_a=FFT_PlanFlags::estimate)
    .def("directional_stiffness",
         [](sys_t& cell, py::EigenDRef<Eigen::ArrayXXd>& v) {
           if ((size_t(v.cols()) != cell.size() ||
                size_t(v.rows()) != dim*dim)) {
             std::stringstream err{};
             err << "need array of shape (" << dim*dim << ", "
                 << cell.size() << ") but got (" << v.rows() << ", "
                 << v.cols() << ").";
             throw std::runtime_error(err.str());
           }
           if (!cell.is_initialised()) {
             cell.initialise();
           }
           const std::string out_name{"temp output for directional stiffness"};
           const std::string in_name{"temp input for directional stiffness"};
           constexpr bool create_tangent{true};
           auto & K = cell.get_tangent(create_tangent);
           auto & input = cell.get_managed_field(in_name);
           auto & output = cell.get_managed_field(out_name);
           input.eigen() = v;
           cell.directional_stiffness(K, input, output);
           return output.eigen();
         },
         "δF"_a)
    .def("project",
         [](sys_t& cell, py::EigenDRef<Eigen::ArrayXXd>& v) {
           if ((size_t(v.cols()) != cell.size() ||
                size_t(v.rows()) != dim*dim)) {
             std::stringstream err{};
             err << "need array of shape (" << dim*dim << ", "
                 << cell.size() << ") but got (" << v.rows() << ", "
                 << v.cols() << ").";
             throw std::runtime_error(err.str());
           }
           if (!cell.is_initialised()) {
             cell.initialise();
           }
           const std::string in_name{"temp input for projection"};
           auto & input = cell.get_managed_field(in_name);
           input.eigen() = v;
           cell.project(input);
           return input.eigen();
         },
         "field"_a)
    .def("get_strain",[](sys_t & s) {
        return Eigen::ArrayXXd(s.get_strain().eigen());
      })
    .def("get_stress",[](sys_t & s) {
        return Eigen::ArrayXXd(s.get_stress().eigen());
      })
    .def("size", &sys_t::size)
    .def("evaluate_stress_tangent",
         [](sys_t& cell, py::EigenDRef<Eigen::ArrayXXd>& v ) {
           if ((size_t(v.cols()) != cell.size() ||
                size_t(v.rows()) != dim*dim)) {
             std::stringstream err{};
             err << "need array of shape (" << dim*dim << ", "
                 << cell.size() << ") but got (" << v.rows() << ", "
                 << v.cols() << ").";
             throw std::runtime_error(err.str());
           }
           auto & strain{cell.get_strain()};
           strain.eigen() = v;
           cell.evaluate_stress_tangent(strain);
         },
         "strain"_a)
    .def("get_projection",
         &sys_t::get_projection)
    .def("get_subdomain_resolutions", &sys_t::get_subdomain_resolutions)
    .def("get_subdomain_locations", &sys_t::get_subdomain_locations)
    .def("get_domain_resolutions", &sys_t::get_domain_resolutions)
    .def("get_domain_lengths", &sys_t::get_domain_resolutions);
}

void add_cell_base(py::module & mod) {
  add_cell_base_helper<twoD>  (mod);
  add_cell_base_helper<threeD>(mod);
}

void add_cell(py::module & mod) {
  add_cell_factory(mod);

  auto cell{mod.def_submodule("cell")};
  cell.doc() = "bindings for cells and cell factories";
 
  cell.def("scale_by_2", [](py::EigenDRef<Eigen::ArrayXXd>& v) {
      v *= 2;
    });
  add_cell_base(cell);
}
