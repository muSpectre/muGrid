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

#include "core/types.hh"
#include "field/field_typed.hh"
#include "operators/convolution_operator_base.hh"
#include "operators/convolution_operator.hh"
#include "benchmark/laplace_operator.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <sstream>

using muGrid::ConvolutionOperatorBase;
using muGrid::ConvolutionOperator;
using muGrid::LaplaceOperator;
using muGrid::TypedFieldBase;
using muGrid::Real;
using muGrid::Index_t;
using muGrid::Shape_t;
using muGrid::HostSpace;
using pybind11::literals::operator""_a;

namespace py = pybind11;

// Type aliases for host fields
using RealFieldHost = TypedFieldBase<Real, HostSpace>;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
using DeviceSpace = muGrid::DefaultDeviceSpace;
using RealFieldDevice = TypedFieldBase<Real, DeviceSpace>;
#endif


// A helper class that bounces calls to virtual methods back to Python 
class PyConvolutionOperator : public ConvolutionOperatorBase {
public:
    // Inherit the constructors
    using ConvolutionOperatorBase::ConvolutionOperatorBase;

    // Trampoline (one for each virtual function)

    void apply(const TypedFieldBase<Real> &nodal_field,
               TypedFieldBase<Real> &quadrature_point_field) const override {
        PYBIND11_OVERRIDE_PURE(
            void,
            ConvolutionOperatorBase,
            apply,
            nodal_field, quadrature_point_field
        );
    }

    void apply_increment(
        const TypedFieldBase<Real> &nodal_field, const Real &alpha,
        TypedFieldBase<Real> &quadrature_point_field) const override {
        PYBIND11_OVERRIDE_PURE(
            void,
            ConvolutionOperatorBase,
            apply_increment,
            nodal_field, alpha, quadrature_point_field
        );
    }

    void
    transpose(const TypedFieldBase<Real> &quadrature_point_field,
              TypedFieldBase<Real> &nodal_field,
              const std::vector<Real> &weights = {}) const override {
        PYBIND11_OVERRIDE_PURE(
            void,
            ConvolutionOperatorBase,
            transpose,
            quadrature_point_field, nodal_field, weights
        );
    }

    void transpose_increment(
        const TypedFieldBase<Real> &quadrature_point_field, const Real &alpha,
        TypedFieldBase<Real> &nodal_field,
        const std::vector<Real> &weights = {}) const override {
        PYBIND11_OVERRIDE_PURE(
            void,
            ConvolutionOperatorBase,
            transpose,
            quadrature_point_field, alpha, nodal_field, weights
        );
    }

    Index_t get_nb_operators() const override {
        PYBIND11_OVERRIDE_PURE(
            Index_t,
            ConvolutionOperatorBase,
            get_nb_operators,
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
void add_convolution_operator_base(py::module &mod) {
    py::class_<ConvolutionOperatorBase, PyConvolutionOperator>(mod, "ConvolutionOperatorBase")
            .def(py::init<>())
            .def("apply", &ConvolutionOperatorBase::apply, "nodal_field"_a, "quadrature_point_field"_a)
            .def_property_readonly("nb_quad_pts", &ConvolutionOperatorBase::get_nb_quad_pts)
            .def_property_readonly("nb_nodal_pts", &ConvolutionOperatorBase::get_nb_nodal_pts)
            .def_property_readonly("nb_operators", &ConvolutionOperatorBase::get_nb_operators)
            .def_property_readonly("spatial_dim", &ConvolutionOperatorBase::get_spatial_dim);
}


// Bind class ConvolutionOperator
void add_convolution_operator_default(py::module &mod) {
    // Function pointer types for explicit overload selection
    using ApplyHostFn = void (ConvolutionOperator::*)(
        const RealFieldHost&, RealFieldHost&) const;
    using TransposeHostFn = void (ConvolutionOperator::*)(
        const RealFieldHost&, RealFieldHost&, const std::vector<Real>&) const;

    auto conv_op = py::class_<ConvolutionOperator, ConvolutionOperatorBase>(mod, "ConvolutionOperator")
            .def(py::init(
                     [](const Shape_t &offset, py::array_t<Real, py::array::f_style | py::array::forcecast> array) {
                         // Array should have shape (directions, quadrature-points, nodal-points, pixels)
                         // pixels portion has nb_dims. Everything in front is omitted.
                         const auto nb_dims{offset.size()};
                         if (nb_dims != 1 && nb_dims != 2 && nb_dims != 3) {
                             throw std::runtime_error("Stencil must be 1D, 2D or 3D");
                         }
                         ssize_t nb_operators{1};
                         if (static_cast<size_t>(array.ndim()) > nb_dims) {
                             nb_operators = array.shape(0);
                         }
                         ssize_t nb_quad_pts{1};
                         if (static_cast<size_t>(array.ndim()) > nb_dims + 1) {
                             nb_quad_pts = array.shape(1);
                         }
                         ssize_t nb_nodal_pts{1};
                         if (static_cast<size_t>(array.ndim()) > nb_dims + 2) {
                             nb_nodal_pts = array.shape(2);
                         }
                         Shape_t nb_stencil_pts(nb_dims);
                         // .shape() returns a pointer to dimension array
                         std::copy(array.shape() + array.ndim() - nb_dims, array.shape() + array.ndim(),
                                   nb_stencil_pts.begin());

                        // Number of entries in the operator
                         const auto nb_entries{
                            nb_operators * nb_quad_pts * nb_nodal_pts *
                            std::accumulate(nb_stencil_pts.begin(),
                                            nb_stencil_pts.end(), 1,
                                            std::multiplies<Index_t>())};
                         return ConvolutionOperator(offset, std::span<const Real>(array.data(), nb_entries),
                                                    nb_stencil_pts, nb_nodal_pts, nb_quad_pts, nb_operators);
                    }),
                 "nb_spatial_dims"_a, "pixel_operator"_a)
            // Host field overloads (always available)
            .def("apply",
                 static_cast<ApplyHostFn>(&ConvolutionOperator::apply),
                 "nodal_field"_a, "quadrature_point_field"_a,
                 "Apply convolution to host (CPU) fields")
            .def("transpose",
                 static_cast<TransposeHostFn>(&ConvolutionOperator::transpose),
                 "quadrature_point_field"_a, "nodal_field"_a,
                 "weights"_a = std::vector<Real>{},
                 "Apply transpose convolution to host (CPU) fields")
            .def_property_readonly("pixel_operator", &ConvolutionOperator::get_pixel_operator)
            .def_property_readonly("spatial_dim", &ConvolutionOperator::get_spatial_dim)
            .def_property_readonly("nb_quad_pts", &ConvolutionOperator::get_nb_quad_pts)
            .def_property_readonly("nb_nodal_pts", &ConvolutionOperator::get_nb_nodal_pts)
            .def_property_readonly("nb_operators", &ConvolutionOperator::get_nb_operators);

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    // Device field overloads (only when GPU backend is enabled)
    using ApplyDeviceFn = void (ConvolutionOperator::*)(
        const RealFieldDevice&, RealFieldDevice&) const;
    using TransposeDeviceFn = void (ConvolutionOperator::*)(
        const RealFieldDevice&, RealFieldDevice&, const std::vector<Real>&) const;

    conv_op
        .def("apply",
             static_cast<ApplyDeviceFn>(&ConvolutionOperator::apply),
             "nodal_field"_a, "quadrature_point_field"_a,
             "Apply convolution to device (GPU) fields")
        .def("transpose",
             static_cast<TransposeDeviceFn>(&ConvolutionOperator::transpose),
             "quadrature_point_field"_a, "nodal_field"_a,
             "weights"_a = std::vector<Real>{},
             "Apply transpose convolution to device (GPU) fields");
#endif
}


// Bind class LaplaceOperator (hard-coded stencil for benchmarking)
void add_laplace_operator(py::module &mod) {
    // Function pointer types for explicit overload selection
    using ApplyHostFn = void (LaplaceOperator::*)(
        const RealFieldHost&, RealFieldHost&) const;

    auto laplace_op = py::class_<LaplaceOperator>(mod, "LaplaceOperator",
        R"pbdoc(
        Hard-coded Laplace operator for benchmarking purposes.

        This operator provides optimized implementations of the discrete Laplace
        operator using:
        - 5-point stencil for 2D grids: [0,1,0; 1,-4,1; 0,1,0]
        - 7-point stencil for 3D grids: center=-6, neighbors=+1

        The output is multiplied by a scale factor, which can be used to
        incorporate grid spacing and sign conventions (e.g., for making
        the operator positive-definite).

        The implementation is designed for benchmarking and performance
        comparison with the generic sparse convolution operator.
        )pbdoc")
        .def(py::init<Index_t, Real>(),
             "spatial_dim"_a, "scale"_a = 1.0,
             "Construct a Laplace operator for the given dimension (2 or 3) "
             "and optional scale factor (default: 1.0)")
        .def("apply",
             static_cast<ApplyHostFn>(&LaplaceOperator::apply),
             "input_field"_a, "output_field"_a,
             "Apply the Laplace operator to host (CPU) fields")
        .def_property_readonly("spatial_dim", &LaplaceOperator::get_spatial_dim,
             "Spatial dimension (2 or 3)")
        .def_property_readonly("nb_stencil_pts", &LaplaceOperator::get_nb_stencil_pts,
             "Number of stencil points (5 for 2D, 7 for 3D)")
        .def_property_readonly("scale", &LaplaceOperator::get_scale,
             "Scale factor applied to output");

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    // Device field overloads (only when GPU backend is enabled)
    using ApplyDeviceFn = void (LaplaceOperator::*)(
        const RealFieldDevice&, RealFieldDevice&) const;

    laplace_op
        .def("apply",
             static_cast<ApplyDeviceFn>(&LaplaceOperator::apply),
             "input_field"_a, "output_field"_a,
             "Apply the Laplace operator to device (GPU) fields");
#endif
}


void add_convolution_operator_classes(py::module &mod) {
    add_convolution_operator_base(mod);
    add_convolution_operator_default(mod);
    add_laplace_operator(mod);
}
