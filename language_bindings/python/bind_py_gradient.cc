/**
* @file   bind_py_field.cc
*
* @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
*
* @date   16 Oct 2019
*
* @brief  Python bindings for fields
*
* Copyright © 2018 Till Junge
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

#include "libmugrid/grid_common.hh"
#include "libmugrid/field_typed.hh"
#include "libmugrid/gradient_operator_base.hh"
#include "libmugrid/gradient_operator_default.hh"
#include "libmugrid/numpy_tools.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <sstream>

using muGrid::GradientOperatorBase;
using muGrid::GradientOperatorDefault;
using muGrid::TypedFieldBase;
using muGrid::Real;
using muGrid::Index_t;
using pybind11::literals::operator""_a;

namespace py = pybind11;


// A helper class that bounces calls to virtual methods back to Python 
class PyGradientOperator : public GradientOperatorBase {
public:
  // Inherit the constructors 
  using GradientOperatorBase::GradientOperatorBase;

  // Trampoline (one for each virtual function) 

  void apply_gradient(const TypedFieldBase<Real> & nodal_field,
                      TypedFieldBase<Real> & quadrature_point_field) const override {
    PYBIND11_OVERRIDE_PURE(
      void,
      GradientOperatorBase,
      apply_gradient,
      nodal_field, quadrature_point_field
    );
  }

  void apply_gradient_increment(
        const TypedFieldBase<Real> & nodal_field, const Real & alpha,
        TypedFieldBase<Real> & quadrature_point_field) const override {
    PYBIND11_OVERRIDE_PURE(
      void,
      GradientOperatorBase,
      apply_gradient_increment,
      nodal_field, alpha, quadrature_point_field
    );
  }

  void 
  apply_transpose(const TypedFieldBase<Real> & quadrature_point_field,
                  TypedFieldBase<Real> & nodal_field,
                  const std::vector<Real> & weights = {}) const override {
    PYBIND11_OVERRIDE_PURE(
      void,
      GradientOperatorBase,
      apply_transpose,
      quadrature_point_field, nodal_field, weights
    );
  }

  void apply_transpose_increment(
      const TypedFieldBase<Real> & quadrature_point_field, const Real & alpha,
      TypedFieldBase<Real> & nodal_field,
      const std::vector<Real> & weights = {}) const override {
    PYBIND11_OVERRIDE_PURE(
      void,
      GradientOperatorBase,
      apply_transpose,
      quadrature_point_field, alpha, nodal_field, weights
    );
  }

  Index_t get_nb_pixel_quad_pts() const override {
    PYBIND11_OVERRIDE_PURE(
      Index_t,
      GradientOperatorBase,
      get_nb_pixel_quad_pts,
    );
  }

  Index_t get_nb_pixel_nodal_pts() const override {
    PYBIND11_OVERRIDE_PURE(
      Index_t,
      GradientOperatorBase,
      get_nb_pixel_nodal_pts,
    );
  }

  Index_t get_spatial_dim() const override {
    PYBIND11_OVERRIDE_PURE(
      Index_t,
      GradientOperatorBase,
      get_spatial_dim,
    );
  }
};


// Bind class GraidentOperatorBase 
void add_gradient_operator_base(py::module & mod) {
  py::class_<GradientOperatorBase, PyGradientOperator>(mod, "GradientOperatorBase")
    .def(py::init<>())
    .def("apply_gradient", &GradientOperatorBase::apply_gradient,
         "nodal_field"_a, "quadrature_point_field"_a)
    .def("get_nb_pixel_quad_pts", &GradientOperatorBase::get_nb_pixel_quad_pts)
    .def("get_nb_pixel_nodal_pts", &GradientOperatorBase::get_nb_pixel_nodal_pts)
    .def("get_spatial_dim", &GradientOperatorBase::get_spatial_dim)
    ;
}


// Bind class GraidentOperatorDefault 
void add_gradient_operator_default(py::module & mod) {
  py::class_<GradientOperatorDefault, GradientOperatorBase>(mod, "GradientOperatorDefault")
    .def(py::init<const Index_t &, const Index_t &, 
         const Index_t &, const Index_t &, const Index_t &,
         const std::vector<std::vector<Eigen::MatrixXd>> &,
         const std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>> &>(),
         "spatial_dim"_a, "nb_quad_pts"_a, "nb_elements"_a, "nb_elemnodal_pts"_a,
         "nb_pixelnodal_pts"_a, "shape_fn_gradients"_a, "nodal_pts"_a)
    .def("apply_gradient", &GradientOperatorDefault::apply_gradient,
      "nodal_field"_a, "quadrature_point_field"_a)
    .def_property_readonly("pixel_gradient", &GradientOperatorDefault::get_pixel_gradient)
    .def_property_readonly("spatial_dim", &GradientOperatorDefault::get_spatial_dim)
    .def_property_readonly("nb_quad_pts", &GradientOperatorDefault::get_nb_quad_pts_per_element)
    .def_property_readonly("nb_elements", &GradientOperatorDefault::get_nb_elements)
    //  .def_property_readonly("nb_elemnodal_pts", &GradientOperatorDefault::???)
    .def_property_readonly("nb_pixelnodal_pts", &GradientOperatorDefault::get_nb_pixel_nodal_pts)
    ;
}


void add_gradient_classes(py::module & mod) {
  add_gradient_operator_base(mod);
  add_gradient_operator_default(mod);
}
