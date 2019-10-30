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
 * Lesser General Public License for more details.
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
 *
 */

#include "common/muSpectre_common.hh"
#include "cell/cell_factory.hh"
#include "cell/cell_base.hh"

#ifdef WITH_SPLIT
#include "cell/cell_split_factory.hh"
#include "cell/cell_split.hh"
#endif

#include <libmugrid/ccoord_operations.hh>
#include <libmufft/communicator.hh>

#ifdef WITH_FFTWMPI
#include "libmufft/fftwmpi_engine.hh"
#endif
#ifdef WITH_PFFT
#include "libmufft/pfft_engine.hh"
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <memory>

using muSpectre::Ccoord_t;
using muSpectre::Dim_t;
using muSpectre::Formulation;
using muSpectre::Gradient_t;
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
  using Gradient = Gradient_t<dim>;

  mod.def(name,
          [](Ccoord nb_grid_pts, Rcoord lens, Formulation form,
             Gradient gradient, muFFT::Communicator & comm) {
            // Initialize with muFFT Communicator object
            return muSpectre::make_cell<dim, dim, muSpectre::CellBase<dim, dim>,
                                        FFTEngine>(
                std::move(nb_grid_pts), std::move(lens), std::move(form),
                std::move(gradient), std::move(comm));
          },
          "nb_grid_pts"_a, "lengths"_a = muGrid::CcoordOps::get_cube<dim>(1.),
          "formulation"_a = Formulation::finite_strain,
          "gradient"_a = muSpectre::make_fourier_gradient<dim>(),
          "communicator"_a = muFFT::Communicator(MPI_COMM_SELF));
}
#endif

#ifdef WITH_SPLIT
template <Dim_t dim>
void add_split_cell_factory_helper(py::module & mod) {
  using Ccoord = Ccoord_t<dim>;
  using Rcoord = Rcoord_t<dim>;
  using Gradient = Gradient_t<dim>;

  mod.def("CellFactorySplit",
          [](Ccoord res, Rcoord lens, Formulation form, Gradient gradient) {
            return make_cell_split(std::move(res), std::move(lens),
                                   std::move(form), std::move(gradient));
          },
          "resolutions"_a, "lengths"_a = muGrid::CcoordOps::get_cube<dim>(1.),
          "formulation"_a = Formulation::finite_strain,
          "gradient"_a = muSpectre::make_fourier_gradient<dim>());
}
#endif

/**
 * the cell factory is only bound for default template parameters
 */
template <Dim_t dim>
void add_cell_factory_helper(py::module & mod) {
  using Ccoord = Ccoord_t<dim>;
  using Rcoord = Rcoord_t<dim>;
  using Gradient = Gradient_t<dim>;

  mod.def("CellFactory",
          [](Ccoord res, Rcoord lens, Formulation form, Gradient gradient) {
            return muSpectre::make_cell(std::move(res), std::move(lens),
                                        std::move(form), std::move(gradient));
          },
          "nb_grid_pts"_a, "lengths"_a = muGrid::CcoordOps::get_cube<dim>(1.),
          "formulation"_a = Formulation::finite_strain,
          "gradient"_a = muSpectre::make_fourier_gradient<dim>());

#ifdef WITH_SPLIT
  add_split_cell_factory_helper<dim>(mod);
#endif

#ifdef WITH_FFTWMPI
  add_parallel_cell_factory_helper<dim, muFFT::FFTWMPIEngine<dim>>(
      mod, "FFTWMPICellFactory");
#endif

#ifdef WITH_PFFT
  add_parallel_cell_factory_helper<dim, muFFT::PFFTEngine<dim>>(
      mod, "PFFTCellFactory");
#endif
}

void add_cell_factory(py::module & mod) {
  add_cell_factory_helper<muGrid::twoD>(mod);
  add_cell_factory_helper<muGrid::threeD>(mod);
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
#ifdef WITH_SPLIT
  using Mat_t = muSpectre::MaterialBase<dim, dim>;
#endif
  py::class_<sys_t, muSpectre::Cell>(mod, name.c_str())
      .def("__len__", &sys_t::size)
      .def("__iter__",
           [](sys_t & s) { return py::make_iterator(s.begin(), s.end()); })
      .def("initialise", &sys_t::initialise,
           "flags"_a = muFFT::FFT_PlanFlags::estimate)
      .def("is_initialised", [](sys_t & s) { return s.is_initialised(); },
           py::return_value_policy::reference_internal)
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
      .def_property_readonly("strain",
                             [](sys_t & s) { return s.get_strain().eigen(); })
      .def_property_readonly("stress",
                             [](sys_t & s) { return s.get_stress().eigen(); })
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
      .def_property_readonly("projection", &sys_t::get_projection)
      .def_property_readonly("communicator", &sys_t::get_communicator)
      .def_property_readonly("nb_subdomain_grid_pts",
                             &sys_t::get_nb_subdomain_grid_pts)
      .def_property_readonly("subdomain_locations",
                             &sys_t::get_subdomain_locations)
      .def_property_readonly("nb_domain_grid_pts",
                             &sys_t::get_nb_domain_grid_pts)
      .def_property_readonly("domain_lengths", &sys_t::get_domain_lengths)
      .def("set_uniform_strain",
           [](sys_t & cell, py::EigenDRef<Eigen::ArrayXXd> & v) -> void {
             cell.set_uniform_strain(v);
           },
           "strain"_a)
      .def("save_history_variables", &sys_t::save_history_variables)

#ifdef WITH_SPLIT
      .def("make_precipitate_laminate",
           [](sys_t & cell, Mat_t & mat_lam, Mat_t & mat_precipitate_cell,
              std::shared_ptr<Mat_t> mat_precipitate,
              std::shared_ptr<Mat_t> mat_matrix,
              std::vector<Rcoord_t<dim>> precipitate_vertices) {
             cell.make_pixels_precipitate_for_laminate_material(
                 precipitate_vertices, mat_lam, mat_precipitate_cell,
                 mat_precipitate, mat_matrix);
           },
           "material_laminate"_a, "mat_precipitate_cell"_a,
           "material_precipitate"_a, "material_matrix"_a, "vertices"_a)
      .def("complete_material_assignemnt_simple",
           [](sys_t & cell, Mat_t & mat_matrix_cell) {
             cell.complete_material_assignment_simple(mat_matrix_cell);
           },
           "material_matrix_cell"_a)
#endif
      ;  // NOLINT
}

#ifdef WITH_SPLIT
template <Dim_t dim>
void add_cell_split_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "CellSplit" << dim << 'd';
  const std::string name = name_stream.str();
  using sys_split_t = muSpectre::CellSplit<dim, dim>;
  using sys_t = muSpectre::CellBase<dim, dim>;
  using Mat_t = muSpectre::MaterialBase<dim, dim>;
  py::class_<sys_split_t, sys_t>(mod, name.c_str())
      .def("make_precipitate",
           [](sys_split_t & cell, Mat_t & mat,
              std::vector<Rcoord_t<dim>> precipitate_vertices) {
             cell.make_automatic_precipitate_split_pixels(precipitate_vertices,
                                                          mat);
           },
           "material"_a, "vertices"_a)

      .def("complete_material_assignment",
           [](sys_split_t & cell, Mat_t & mat) {
             cell.complete_material_assignment(mat);
           },
           "material"_a)
      .def("get_splitness", [](sys_t & cell) { return cell.get_splitness(); });
}
#endif

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
#ifdef WITH_SPLIT
void add_cell_split(py::module & mod) {
  add_cell_base(mod);
  add_cell_split_helper<muSpectre::twoD>(mod);
  add_cell_split_helper<muSpectre::threeD>(mod);
}

#endif
void add_cell(py::module & mod) {
  add_cell_factory(mod);

  auto cell{mod.def_submodule("cell")};
  cell.doc() = "bindings for cells and cell factories";

#ifdef WITH_SPLIT
  add_cell_split(cell);
#else
  add_cell_base(cell);
#endif
}
