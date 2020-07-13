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

#include <libmugrid/state_field.hh>

#include "common/muSpectre_common.hh"
#include "cell/cell_factory.hh"
#include "cell/cell.hh"
#include "projection/projection_base.hh"

#ifdef WITH_SPLIT
#include "cell/cell_split_factory.hh"
#include "cell/cell_split.hh"
#endif

#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/numpy_tools.hh>
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

using muFFT::Communicator;
using muFFT::Gradient_t;
using muGrid::numpy_wrap;
using muGrid::NumpyProxy;
using muSpectre::Ccoord_t;
using muSpectre::Cell;
using muSpectre::Formulation;
using muSpectre::Index_t;
using muSpectre::Rcoord_t;
using pybind11::literals::operator""_a;
namespace py = pybind11;

#ifdef WITH_FFTWMPI
using muFFT::FFTWMPIEngine;
#endif

/**
 * the cell factory is only bound for default template parameters
 */
void add_cell_factory(py::module & mod) {
  using DynCcoord_t = muGrid::DynCcoord_t;
  using DynRcoord_t = muGrid::DynRcoord_t;

  mod.def(
      "CellFactory",
      [](DynCcoord_t res, DynRcoord_t lens, Formulation form,
         Gradient_t gradient, Communicator comm,
         const muFFT::FFT_PlanFlags & flags) {
        return muSpectre::make_cell(std::move(res), std::move(lens),
                                    std::move(form), std::move(gradient),
                                    std::move(comm), std::move(flags));
      },
      "nb_grid_pts"_a, "lengths"_a, "formulation"_a, "gradient"_a,
      "communicator"_a, "flags"_a = muFFT::FFT_PlanFlags::estimate);

  mod.def(
      "CellFactory",
      [](DynCcoord_t res, DynRcoord_t lens, Formulation form,
         Gradient_t gradient) {
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
}

#ifdef WITH_FFTWMPI
void add_fftwmpi_cell_factory(py::module & mod) {
  using DynCcoord_t = muGrid::DynCcoord_t;
  using DynRcoord_t = muGrid::DynRcoord_t;

  mod.def(
      "FFTWMPICellFactory",
      [](DynCcoord_t res, DynRcoord_t lens, Formulation form,
         Gradient_t gradient, Communicator comm) {
        return muSpectre::make_cell<Cell, FFTWMPIEngine>(
            std::move(res), std::move(lens), std::move(form),
            std::move(gradient), std::move(comm));
      },
      "nb_grid_pts"_a, "lengths"_a, "formulation"_a, "gradient"_a,
      "communicator"_a);

  mod.def(
      "FFTWMPICellFactory",
      [](DynCcoord_t res, DynRcoord_t lens, Formulation form,
         Gradient_t gradient) {
        return muSpectre::make_cell<Cell, FFTWMPIEngine>(
            std::move(res), std::move(lens), std::move(form),
            std::move(gradient));
      },
      "nb_grid_pts"_a, "lengths"_a, "formulation"_a, "gradient"_a);

  mod.def(
      "FFTWMPICellFactory",
      [](DynCcoord_t res, DynRcoord_t lens, Formulation form) {
        return muSpectre::make_cell<Cell, FFTWMPIEngine>(
            std::move(res), std::move(lens), std::move(form));
      },
      "nb_grid_pts"_a, "lengths"_a, "formulation"_a);
}
#endif

#ifdef WITH_SPLIT
void add_split_cell_factory_helper(py::module & mod) {
  using DynCcoord_t = muGrid::DynCcoord_t;
  using DynRcoord_t = muGrid::DynRcoord_t;
  mod.def(
      "CellFactorySplit",
      [](DynCcoord_t res, DynRcoord_t lens, Formulation form,
         Gradient_t gradient) {
        return make_cell_split(std::move(res), std::move(lens), std::move(form),
                               std::move(gradient));
      },
      "resolutions"_a, "lengths"_a,
      "formulation"_a = Formulation::finite_strain, "gradient"_a);
}
#endif

/**
 * CellBase for which the material and spatial dimension are identical
 */
void add_cell_helper(py::module & mod) {
  using muSpectre::Cell;
  using muSpectre::Real;
#ifdef WITH_SPLIT
  using Mat_t = muSpectre::MaterialBase;
  using DynRcoord_t = muGrid::DynRcoord_t;
#endif
  auto NumpyT2Proxy{
      [](Cell & cell,
         py::array_t<Real, py::array::f_style> & tensor2)
          -> NumpyProxy<Real, py::array::f_style> {
        auto && strain_shape{cell.get_strain_shape()};
        auto & proj{cell.get_projection()};
        return NumpyProxy<Real, py::array::f_style>{
            proj.get_nb_subdomain_grid_pts(),
            proj.get_subdomain_locations(),
            proj.get_nb_quad_pts(),
            {strain_shape[0], strain_shape[1]},
            tensor2};
      }};
  py::class_<Cell>(mod, "Cell")
      .def(py::init([](const muSpectre::ProjectionBase & projection) {
        return Cell{projection.clone()};
      }))
      .def("initialise", &Cell::initialise)
      .def(
          "is_initialised", [](Cell & s) { return s.is_initialised(); },
          py::return_value_policy::reference_internal)
      .def(
          "directional_stiffness",
          [&NumpyT2Proxy](
              Cell & cell,
              py::array_t<Real, py::array::f_style> & delta_strain) {
            if (!cell.is_initialised()) {
              cell.initialise();
            }
            auto & fields{cell.get_fields()};
            const std::string out_name{"temp output for directional stiffness"};
            if (not fields.field_exists(out_name)) {
              fields.register_real_field(out_name, cell.get_strain_shape(),
                                         muSpectre::QuadPtTag);
            }
            auto & delta_stress{
                dynamic_cast<muGrid::RealField &>(fields.get_field(out_name))};
            auto delta_strain_array{
                NumpyT2Proxy(cell, delta_strain)};
            cell.evaluate_projected_directional_stiffness(
                delta_strain_array.get_field(), delta_stress);
            return numpy_wrap(delta_stress);
          },
          "delta_strain"_a, py::keep_alive<0, 1>())
      .def("project", &Cell::apply_projection, "strain"_a)
      .def(
          "project",
          [&NumpyT2Proxy](Cell & cell,
                          py::array_t<Real, py::array::f_style> & strain) {
            if (!cell.is_initialised()) {
              cell.initialise();
            }
            auto & fields{cell.get_fields()};
            const std::string out_name{"temp output for projection"};
            if (not fields.field_exists(out_name)) {
              fields.register_real_field(out_name, cell.get_strain_shape(),
                                         muSpectre::QuadPtTag);
            }
            auto & strain_field{
                dynamic_cast<muGrid::RealField &>(fields.get_field(out_name))};
            strain_field =
                NumpyT2Proxy(cell, strain)
                    .get_field();
            cell.apply_projection(strain_field);
            return numpy_wrap(strain_field);
          },
          "strain"_a)
      .def_property("strain", &Cell::get_strain,
                    [](Cell & cell, muGrid::TypedFieldBase<Real> & strain) {
                      cell.get_strain() = strain;
                    })
      .def_property_readonly("stress", &Cell::get_stress)
      .def_property_readonly("nb_dof", &Cell::get_nb_dof)
      .def_property_readonly("nb_pixels", &Cell::get_nb_pixels)
      .def(
          "evaluate_stress_tangent",
          [&NumpyT2Proxy](Cell & cell,
                          py::array_t<Real, py::array::f_style> & strain) {
            auto strain_array{
                NumpyT2Proxy(cell, strain)};

            cell.get_strain() = strain_array.get_field();
            auto && stress_tgt{cell.evaluate_stress_tangent()};
            auto && numpy_stress{numpy_wrap(std::get<0>(stress_tgt))};
            auto && numpy_tangent{numpy_wrap(std::get<1>(stress_tgt))};
            return py::make_tuple(numpy_stress, numpy_tangent);
          },
          "strain"_a, py::return_value_policy::reference_internal)
      .def(
          "evaluate_stress_tangent",
          [](Cell & cell, muGrid::TypedFieldBase<Real> & strain) {
            cell.get_strain() = strain;
            auto && stress_tgt{cell.evaluate_stress_tangent()};
            auto && numpy_stress{numpy_wrap(std::get<0>(stress_tgt))};
            auto && numpy_tangent{numpy_wrap(std::get<1>(stress_tgt))};
            return py::make_tuple(numpy_stress, numpy_tangent);
          },
          "strain"_a, py::return_value_policy::reference_internal)
      .def(
          "evaluate_stress",
          [&NumpyT2Proxy](Cell & cell,
                          py::array_t<Real, py::array::f_style> & strain) {
            auto strain_array{
                NumpyT2Proxy(cell, strain)};

            cell.get_strain() = strain_array.get_field();
            return numpy_wrap(cell.evaluate_stress());
          },
          "strain"_a, py::keep_alive<0, 1>())
      .def_property_readonly("projection", &Cell::get_projection)
      .def_property_readonly("communicator", &Cell::get_communicator)
      .def_property_readonly(
          "nb_subdomain_grid_pts",
          [](Cell & cell) {
            return cell.get_projection().get_nb_subdomain_grid_pts();
          })
      .def_property_readonly(
          "subdomain_locations",

          [](Cell & cell) {
            return cell.get_projection().get_subdomain_locations();
          })
      .def_property_readonly(
          "nb_domain_grid_pts",
          [](Cell & cell) {
            return cell.get_projection().get_nb_domain_grid_pts();
          })
      .def_property_readonly(
          "domain_lengths",
          [](Cell & cell) {
            return cell.get_projection().get_domain_lengths();
          })
      .def(
          "set_uniform_strain",
          [](Cell & cell, py::EigenDRef<Eigen::ArrayXXd> & strain) -> void {
            cell.set_uniform_strain(strain);
          },
          "strain"_a)
      .def("save_history_variables", &Cell::save_history_variables)
      .def("get_globalised_internal_real_field",
           &Cell::globalise_real_internal_field, "unique_name"_a,
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
      .def("get_globalised_current_real_field",
           &Cell::globalise_real_current_field, "unique_name"_a,
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
      .def("get_globalised_old_real_field", &Cell::globalise_real_old_field,
           "unique_name"_a, "nb_steps_ago"_a = 1,
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
      .def_property_readonly("pixels", &Cell::get_pixels)
      .def_property_readonly("pixel_indices", &Cell::get_pixel_indices)
      .def_property_readonly("quad_pt_indices", &Cell::get_quad_pt_indices)
      .def_property_readonly("fields", &Cell::get_fields)

#ifdef WITH_SPLIT
      .def(
          "make_precipitate_laminate",
          [](Cell & cell, Mat_t & mat_lam, Mat_t & mat_precipitate_cell,
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
          [](Cell & cell, Mat_t & mat_matrix_cell) {
            cell.complete_material_assignment_simple(mat_matrix_cell);
          },
          "material_matrix_cell"_a)
#endif
      ;  // NOLINT
}

#ifdef WITH_SPLIT
void add_cell_split_helper(py::module & mod) {
  using DynRcoord_t = muGrid::DynRcoord_t;
  using CellSplit_t = muSpectre::CellSplit;
  using Cell_t = muSpectre::Cell;
  using Mat_t = muSpectre::MaterialBase;
  py::class_<CellSplit_t, Cell_t>(mod, "CellSplit")
      .def(
          "make_precipitate",
          [](CellSplit_t & cell, Mat_t & mat,
             std::vector<DynRcoord_t> precipitate_vertices) {
            cell.make_automatic_precipitate_split_pixels(precipitate_vertices,
                                                         mat);
          },
          "vertices"_a, "material"_a)

      .def(
          "complete_material_assignment",
          [](CellSplit_t & cell, Mat_t & mat) {
            cell.complete_material_assignment(mat);
          },
          "material"_a)
      .def("get_splitness", [](Cell_t & cell) { return cell.get_splitness(); });
}
#endif

void add_cell(py::module & mod) {
  add_cell_factory(mod);

  auto cell{mod.def_submodule("cell")};
  cell.doc() = "bindings for cells and cell factories";

#ifdef WITH_SPLIT
  add_split_cell_factory_helper(mod);
  add_cell_helper(cell);
  add_cell_split_helper(cell);
#else
  add_cell_helper(cell);
#endif

#ifdef WITH_FFTWMPI
  add_fftwmpi_cell_factory(mod);
#endif
}
