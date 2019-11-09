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

#include <libmugrid/state_nfield.hh>

#include "common/muSpectre_common.hh"
#include "cell/cell_factory.hh"
#include "cell/ncell.hh"

#ifdef WITH_SPLIT
#include "cell/cell_split_factory.hh"
#include "cell/cell_split.hh"
#endif

#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/numpy.hh>
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

using muFFT::Gradient_t;
using muGrid::numpy_wrap;
using muGrid::NumpyProxy;
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
  using Gradient = Gradient_t<dim>;

  mod.def(
      name,
      [](Ccoord nb_grid_pts, Rcoord lens, Formulation form, Gradient gradient,
         muFFT::Communicator & comm) {
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

  mod.def(
      "CellFactorySplit",
      [](Ccoord res, Rcoord lens, Formulation form, Gradient gradient) {
        return make_cell_split(std::move(res), std::move(lens), std::move(form),
                               std::move(gradient));
      },
      "resolutions"_a, "lengths"_a = muGrid::CcoordOps::get_cube<dim>(1.),
      "formulation"_a = Formulation::finite_strain,
      "gradient"_a = muSpectre::make_fourier_gradient<dim>());
}
#endif

/**
 * the cell factory is only bound for default template parameters
 */
void add_cell_factory(py::module & mod) {
  using DynCcoord_t = muGrid::DynCcoord_t;
  using DynRcoord_t = muGrid::DynRcoord_t;
  using Gradient = Gradient_t;

  mod.def(
      "CellFactory",
      [](DynCcoord_t res, DynRcoord_t lens, Formulation form,
         Gradient gradient) {
        return muSpectre::make_cell(std::move(res), std::move(lens),
                                    std::move(form), std::move(gradient));
      },
      "nb_grid_pts"_a, "lengths"_a, "formulation"_a, "gradient"_a);

  mod.def(
      "CellFactory",
      [](DynCcoord_t res, DynRcoord_t lens, Formulation form) {
        return muSpectre::make_cell(std::move(res), std::move(lens),
                                    std::move(form));
      },
      "nb_grid_pts"_a, "lengths"_a, "formulation"_a);

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

/**
 * CellBase for which the material and spatial dimension are identical
 */
void add_cell_helper(py::module & mod) {
  using muSpectre::NCell;
  using muSpectre::Real;
#ifdef WITH_SPLIT
  using Mat_t = muSpectre::MaterialBase<dim, dim>;
#endif
  auto NumpyT2Proxy{
      [](NCell & cell,
         py::array_t<Real, py::array::f_style> & tensor2) -> NumpyProxy<Real> {
        auto && strain_shape{cell.get_strain_shape()};
        auto & proj{cell.get_projection()};
        return NumpyProxy<Real>{proj.get_nb_subdomain_grid_pts(),
                                proj.get_subdomain_locations(),
                                proj.get_nb_quad(),
                                {strain_shape[0], strain_shape[1]},
                                tensor2};
      }};
  py::class_<NCell>(mod, "Cell")
      .def("initialise", &NCell::initialise,
           "flags"_a = muFFT::FFT_PlanFlags::estimate)
      .def(
          "is_initialised", [](NCell & s) { return s.is_initialised(); },
          py::return_value_policy::reference_internal)
      .def(
          "directional_stiffness",
          [&NumpyT2Proxy](
              NCell & cell,
              py::array_t<Real, py::array::f_style> & delta_strain) {
            if (!cell.is_initialised()) {
              cell.initialise();
            }
            auto & fields{cell.get_fields()};
            const std::string out_name{"temp output for directional stiffness"};
            if (not fields.field_exists(out_name)) {
              auto && strain_shape{cell.get_strain_shape()};
              auto && nb_components{strain_shape[0] * strain_shape[1]};
              fields.register_real_field(out_name, nb_components);
            }
            auto & delta_stress{
                dynamic_cast<muGrid::RealNField &>(fields.get_field(out_name))};
            auto delta_strain_array{NumpyT2Proxy(cell, delta_strain)};
            cell.evaluate_projected_directional_stiffness(
                delta_strain_array.get_field(), delta_stress);
            return numpy_wrap(delta_stress);
          },
          "delta_strain"_a, py::return_value_policy::reference_internal)
      .def("project", &NCell::apply_projection, "field"_a)
      .def_property("strain", &NCell::get_strain,
                    [](NCell & cell, muGrid::TypedNFieldBase<Real> & strain) {
                      cell.get_strain() = strain;
                    })
      .def_property_readonly("stress", &NCell::get_stress)
      .def_property_readonly("nb_dof", &NCell::get_nb_dof)
      .def_property_readonly("nb_pixels", &NCell::get_nb_pixels)
      .def(
          "evaluate_stress_tangent",
          [&NumpyT2Proxy](NCell & cell,
                          py::array_t<Real, py::array::f_style> & strain) {
            auto strain_array{NumpyT2Proxy(cell, strain)};

            cell.get_strain() = strain_array.get_field();
            auto stress_tgt{cell.evaluate_stress_tangent()};
            return py::make_tuple(numpy_wrap(std::get<0>(stress_tgt)),
                                  numpy_wrap(std::get<1>(stress_tgt)));
          },
          "strain"_a, py::return_value_policy::reference_internal)
      .def(
          "evaluate_stress",
          [&NumpyT2Proxy](NCell & cell,
                          py::array_t<Real, py::array::f_style> & strain) {
            auto strain_array{NumpyT2Proxy(cell, strain)};

            cell.get_strain() = strain_array.get_field();
            return numpy_wrap(cell.evaluate_stress());
          },
          "strain"_a, py::return_value_policy::reference_internal)
      .def_property_readonly("projection", &NCell::get_projection)
      .def_property_readonly("communicator", &NCell::get_communicator)
      .def_property_readonly(
          "nb_subdomain_grid_pts",
          [](NCell & cell) {
            return cell.get_projection().get_nb_subdomain_grid_pts();
          })
      .def_property_readonly(
          "subdomain_locations",

          [](NCell & cell) {
            return cell.get_projection().get_subdomain_locations();
          })
      .def_property_readonly(
          "nb_domain_grid_pts",
          [](NCell & cell) {
            return cell.get_projection().get_nb_domain_grid_pts();
          })
      .def_property_readonly(
          "domain_lengths",
          [](NCell & cell) {
            return cell.get_projection().get_domain_lengths();
          })
      .def(
          "set_uniform_strain",
          [](NCell & cell, py::EigenDRef<Eigen::ArrayXXd> & strain) -> void {
            cell.set_uniform_strain(strain);
          },
          "strain"_a)
      .def("save_history_variables", &NCell::save_history_variables)
      .def("get_globalised_internal_real_field",
           &NCell::globalise_real_internal_field, "unique_name"_a,
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
           "Exception is raised.",
           py::return_value_policy::reference_internal)
      .def(
          "get_globalised_current_real_field",
          [](NCell & cell, const std::string & unique_prefix)
              -> muGrid::TypedNFieldBase<Real> & {
            auto current_name{cell.get_fields()
                                  .get_state_field(unique_prefix)
                                  .current()
                                  .get_name()};
            return cell.globalise_real_internal_field(current_name);
          },
          "unique_prefix"_a, py::return_value_policy::reference_internal)
      .def(
          "get_globalised_old_real_field",
          [](NCell & cell, const std::string & unique_prefix,
             const size_t & nb_steps_ago) -> muGrid::TypedNFieldBase<Real> & {
            auto old_name{cell.get_fields()
                              .get_state_field(unique_prefix)
                              .old(nb_steps_ago)
                              .get_name()};
            return cell.globalise_real_internal_field(old_name);
          },
          "unique_prefix"_a, "nb_steps_ago"_a = 1,
          py::return_value_policy::reference_internal)
    .def_property_readonly("pixels", &NCell::get_pixels)
    .def_property_readonly("pixel_indices", &NCell::get_pixel_indices)
    .def_property_readonly("quad_pt_indices", &NCell::get_quad_pt_indices)

#ifdef WITH_SPLIT
      .def(
          "make_precipitate_laminate",
          [](Cell_t & cell, Mat_t & mat_lam, Mat_t & mat_precipitate_cell,
             std::shared_ptr<Mat_t> mat_precipitate,
             std::shared_ptr<Mat_t> mat_matrix,
             std::vector<DynRcoord_t> precipitate_vertices) {
            cell.make_pixels_precipitate_for_laminate_material(
                precipitate_vertices, mat_lam, mat_precipitate_cell,
                mat_precipitate, mat_matrix);
          },
          "material_laminate"_a, "mat_precipitate_cell"_a,
          "material_precipitate"_a, "material_matrix"_a, "vertices"_a)
      .def(
          "complete_material_assignemnt_simple",
          [](Cell_t & cell, Mat_t & mat_matrix_cell) {
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
  using Cell_t = muSpectre::CellBase<dim, dim>;
  using Mat_t = muSpectre::MaterialBase<dim, dim>;
  py::class_<sys_split_t, Cell_t>(mod, name.c_str())
      .def(
          "make_precipitate",
          [](sys_split_t & cell, Mat_t & mat,
             std::vector<DynRcoord_t> precipitate_vertices) {
            cell.make_automatic_precipitate_split_pixels(precipitate_vertices,
                                                         mat);
          },
          "material"_a, "vertices"_a)

      .def(
          "complete_material_assignment",
          [](sys_split_t & cell, Mat_t & mat) {
            cell.complete_material_assignment(mat);
          },
          "material"_a)
      .def("get_splitness", [](Cell_t & cell) { return cell.get_splitness(); });
}
#endif

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
  add_cell_helper(cell);
#endif
}
