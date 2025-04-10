/**
* @file   bind_py_convolution_operator.cc
*
* @author Yizhen Wang <yizhen.wang@imtek.uni-freiburg.de>
*
* @date   28 Nov 2024
*
* @brief  Python bindings for the convolution operator
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
#include "libmugrid/convolution_operator_base.hh"
#include "libmugrid/convolution_operator.hh"
#include "libmugrid/numpy_tools.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <sstream>

using muGrid::ConvolutionOperatorBase;
using muGrid::ConvolutionOperator;
using muGrid::TypedFieldBase;
using muGrid::Real;
using muGrid::Index_t;
using muGrid::Shape_t;
using pybind11::literals::operator""_a;

namespace py = pybind11;


// A helper class that bounces calls to virtual methods back to Python 
class PyConvolutionOperator : public ConvolutionOperatorBase {
public:
  // Inherit the constructors 
  using ConvolutionOperatorBase::ConvolutionOperatorBase;

  // Trampoline (one for each virtual function) 

  void apply(const TypedFieldBase<Real> & nodal_field,
                      TypedFieldBase<Real> & quadrature_point_field) const override {
    PYBIND11_OVERRIDE_PURE(
      void,
      ConvolutionOperatorBase,
      apply,
      nodal_field, quadrature_point_field
    );
  }

  void apply_increment(
        const TypedFieldBase<Real> & nodal_field, const Real & alpha,
        TypedFieldBase<Real> & quadrature_point_field) const override {
    PYBIND11_OVERRIDE_PURE(
      void,
      ConvolutionOperatorBase,
      apply_increment,
      nodal_field, alpha, quadrature_point_field
    );
  }

  void 
  transpose(const TypedFieldBase<Real> & quadrature_point_field,
                  TypedFieldBase<Real> & nodal_field,
                  const std::vector<Real> & weights = {}) const override {
    PYBIND11_OVERRIDE_PURE(
      void,
      ConvolutionOperatorBase,
      transpose,
      quadrature_point_field, nodal_field, weights
    );
  }

  void transpose_increment(
      const TypedFieldBase<Real> & quadrature_point_field, const Real & alpha,
      TypedFieldBase<Real> & nodal_field,
      const std::vector<Real> & weights = {}) const override {
    PYBIND11_OVERRIDE_PURE(
      void,
      ConvolutionOperatorBase,
      transpose,
      quadrature_point_field, alpha, nodal_field, weights
    );
  }

  Index_t get_nb_quad_pts() const override {
    PYBIND11_OVERRIDE_PURE(
      Index_t,
      ConvolutionOperatorBase,
      get_nb_quad_pts,
    );
  }

  Index_t get_nb_nodal_pts() const override {
    PYBIND11_OVERRIDE_PURE(
      Index_t,
      ConvolutionOperatorBase,
      get_nb_nodal_pts,
    );
  }

  Index_t get_spatial_dim() const override {
    PYBIND11_OVERRIDE_PURE(
      Index_t,
      ConvolutionOperatorBase,
      get_spatial_dim,
    );
  }
};


// Bind class GraidentOperatorBase 
void add_convolution_operator_base(py::module & mod) {
  py::class_<ConvolutionOperatorBase, PyConvolutionOperator>(mod, "ConvolutionOperatorBase")
    .def(py::init<>())
    .def("apply", &ConvolutionOperatorBase::apply, "nodal_field"_a, "quadrature_point_field"_a)
    .def("get_nb_quad_pts", &ConvolutionOperatorBase::get_nb_quad_pts)
    .def("get_nb_nodal_pts", &ConvolutionOperatorBase::get_nb_nodal_pts)
    .def("get_spatial_dim", &ConvolutionOperatorBase::get_spatial_dim)
    ;
}


// Bind class ConvolutionOperator
void add_convolution_operator_default(py::module & mod) {
  py::class_<ConvolutionOperator, ConvolutionOperatorBase>(mod, "ConvolutionOperator")
    .def(py::init<Eigen::Ref<const Eigen::MatrixXd>, const Shape_t &,
        const Index_t &, const Index_t &, const Index_t &>(),
        "pixel_operator"_a, "conv_pts_shape"_a, "nb_pixelnodal_pts"_a,
        "nb_quad_pts"_a, "nb_operators"_a)
    .def("apply", &ConvolutionOperator::apply, "nodal_field"_a, "quadrature_point_field"_a)
    .def("transpose", &ConvolutionOperator::transpose, "quadrature_point_field"_a, 
        "nodal_field"_a, "weights"_a=std::vector<Real>{})
    .def_property_readonly("pixel_operator", &ConvolutionOperator::get_pixel_operator)
    .def_property_readonly("spatial_dim", &ConvolutionOperator::get_spatial_dim)
    .def_property_readonly("nb_quad_pts", &ConvolutionOperator::get_nb_quad_pts)
    .def_property_readonly("nb_nodal_pts", &ConvolutionOperator::get_nb_nodal_pts)
    ;
}


void add_convolution_operator_classes(py::module & mod) {
  add_convolution_operator_base(mod);
  add_convolution_operator_default(mod);
}
