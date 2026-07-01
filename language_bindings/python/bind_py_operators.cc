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
#include "operators/linear.hh"
#include "operators/generic.hh"
#include "operators/laplace.hh"
#include "operators/fem_gradient.hh"
#include "operators/solids/isotropic_stiffness.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <sstream>

using muGrid::GhostRequirement;
using muGrid::LinearOperator;
using muGrid::GenericLinearOperator;
using muGrid::LaplaceOperator2D;
using muGrid::LaplaceOperator3D;
using muGrid::FEMGradientOperator2D;
using muGrid::FEMGradientOperator3D;
using muGrid::FEMGradientOperatorQ1_2D;
using muGrid::FEMGradientOperatorQ1_3D;
using muGrid::IsotropicStiffnessOperator2D;
using muGrid::IsotropicStiffnessOperator3D;
using muGrid::FEMElementKind;
using muGrid::TypedFieldBase;
using muGrid::Real;
using muGrid::Real32;
using muGrid::Dim_t;
using muGrid::Index_t;
using muGrid::Shape_t;
using muGrid::HostSpace;
using pybind11::literals::operator""_a;

// Backwards compatibility aliases
using ConvolutionOperatorBase = LinearOperator;
using ConvolutionOperator = GenericLinearOperator;

namespace py = pybind11;

// Type aliases for host fields
using RealFieldHost = TypedFieldBase<Real, HostSpace>;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
using DeviceSpace = muGrid::DefaultDeviceSpace;
using RealFieldDevice = TypedFieldBase<Real, DeviceSpace>;
#endif

// A helper class that bounces calls to virtual methods back to Python
class PyGradientOperator : public LinearOperator {
   public:
    // Inherit the constructors
    using LinearOperator::LinearOperator;

    // Trampoline (one for each virtual function)

    void apply(const TypedFieldBase<Real> & nodal_field,
               TypedFieldBase<Real> & quadrature_point_field) const override {
        PYBIND11_OVERRIDE_PURE(void, LinearOperator, apply, nodal_field,
                               quadrature_point_field);
    }

    void apply_increment(
        const TypedFieldBase<Real> & nodal_field, const Real & alpha,
        TypedFieldBase<Real> & quadrature_point_field) const override {
        PYBIND11_OVERRIDE_PURE(void, LinearOperator, apply_increment,
                               nodal_field, alpha, quadrature_point_field);
    }

    void transpose(const TypedFieldBase<Real> & quadrature_point_field,
                   TypedFieldBase<Real> & nodal_field,
                   const std::vector<Real> & weights = {}) const override {
        PYBIND11_OVERRIDE_PURE(void, LinearOperator, transpose,
                               quadrature_point_field, nodal_field, weights);
    }

    void
    transpose_increment(const TypedFieldBase<Real> & quadrature_point_field,
                        const Real & alpha, TypedFieldBase<Real> & nodal_field,
                        const std::vector<Real> & weights = {}) const override {
        PYBIND11_OVERRIDE_PURE(void, LinearOperator, transpose_increment,
                               quadrature_point_field, alpha, nodal_field,
                               weights);
    }

    Index_t get_nb_output_components() const override {
        PYBIND11_OVERRIDE_PURE(Index_t, LinearOperator,
                               get_nb_output_components, );
    }

    Index_t get_nb_quad_pts() const override {
        PYBIND11_OVERRIDE_PURE(Index_t, LinearOperator, get_nb_quad_pts, );
    }

    Index_t get_nb_input_components() const override {
        PYBIND11_OVERRIDE_PURE(Index_t, LinearOperator,
                               get_nb_input_components, );
    }

    Dim_t get_spatial_dim() const override {
        PYBIND11_OVERRIDE_PURE(Index_t, LinearOperator, get_spatial_dim, );
    }

    muGrid::Shape_t get_offset() const override {
        PYBIND11_OVERRIDE_PURE(muGrid::Shape_t, LinearOperator, get_offset, );
    }

    muGrid::Shape_t get_stencil_shape() const override {
        PYBIND11_OVERRIDE_PURE(muGrid::Shape_t, LinearOperator,
                               get_stencil_shape, );
    }
};

// Convert a GhostRequirement to a ((left...), (right...)) pair of tuples
static py::tuple ghost_requirement_to_python(const GhostRequirement & req) {
    auto shape_to_tuple{[](const muGrid::Shape_t & shape) {
        py::tuple t(shape.size());
        for (std::size_t i{0}; i < shape.size(); ++i) {
            t[i] = shape[i];
        }
        return t;
    }};
    return py::make_tuple(shape_to_tuple(req.left), shape_to_tuple(req.right));
}

// Bind class GradientOperator
void add_gradient_operator(py::module & mod) {
    py::class_<LinearOperator, PyGradientOperator>(mod, "GradientOperator")
        .def(py::init<>())
        .def("apply",
             static_cast<void (LinearOperator::*)(
                 const TypedFieldBase<Real> &, TypedFieldBase<Real> &) const>(
                 &LinearOperator::apply),
             "nodal_field"_a, "quadrature_point_field"_a)
        .def_property_readonly("nb_quad_pts",
                               &LinearOperator::get_nb_quad_pts)
        .def_property_readonly("nb_input_components",
                               &LinearOperator::get_nb_input_components)
        .def_property_readonly("nb_output_components",
                               &LinearOperator::get_nb_output_components)
        .def_property_readonly("spatial_dim",
                               &LinearOperator::get_spatial_dim)
        .def_property_readonly(
            "apply_ghost_requirement",
            [](const LinearOperator & op) {
                return ghost_requirement_to_python(
                    op.get_apply_ghost_requirement());
            },
            "Ghost layers (left, right) required by apply()")
        .def_property_readonly(
            "transpose_ghost_requirement",
            [](const LinearOperator & op) {
                return ghost_requirement_to_python(
                    op.get_transpose_ghost_requirement());
            },
            "Ghost layers (left, right) required by transpose()")
        .def_property_readonly(
            "ghost_requirement",
            [](const LinearOperator & op) {
                return ghost_requirement_to_python(op.get_ghost_requirement());
            },
            "Ghost layers (left, right) sufficient for both apply() and "
            "transpose(); pass an operator (or a list of operators) as the "
            "`ghosts` argument of CartesianDecomposition or FFTEngine to "
            "size the ghost buffers automatically");
}

// Bind class GenericLinearOperator
void add_stencil_gradient_operator(py::module & mod) {
    // Function pointer types for explicit overload selection
    using ApplyHostFn = void (GenericLinearOperator::*)(
        const RealFieldHost &, RealFieldHost &) const;
    using TransposeHostFn = void (GenericLinearOperator::*)(
        const RealFieldHost &, RealFieldHost &, const std::vector<Real> &)
        const;

    auto conv_op = py::class_<GenericLinearOperator, LinearOperator>(mod, "GenericLinearOperator")
            .def(py::init(
                     [](const Shape_t &offset, py::array_t<Real, py::array::f_style | py::array::forcecast> array) {
                         // Array should have shape (directions, quadrature-points, nodal-points, pixels)
                         // pixels portion has nb_dims. Everything in front is omitted.
                         const auto nb_dims{offset.size()};
                         if (nb_dims != 1 && nb_dims != 2 && nb_dims != 3) {
                             throw std::runtime_error("Stencil must be 1D, 2D or 3D");
                         }
                         py::ssize_t nb_output_components{1};
                         if (static_cast<size_t>(array.ndim()) > nb_dims) {
                             nb_output_components = array.shape(0);
                         }
                         py::ssize_t nb_quad_pts{1};
                         if (static_cast<size_t>(array.ndim()) > nb_dims + 1) {
                             nb_quad_pts = array.shape(1);
                         }
                         py::ssize_t nb_input_components{1};
                         if (static_cast<size_t>(array.ndim()) > nb_dims + 2) {
                             nb_input_components = array.shape(2);
                         }
                         Shape_t nb_stencil_pts(nb_dims);
                         // .shape() returns a pointer to dimension array
                         std::copy(array.shape() + array.ndim() - nb_dims, array.shape() + array.ndim(),
                                   nb_stencil_pts.begin());

                        // Number of entries in the operator
                         const auto nb_entries{
                            nb_output_components * nb_quad_pts * nb_input_components *
                            std::accumulate(nb_stencil_pts.begin(),
                                            nb_stencil_pts.end(), 1,
                                            std::multiplies<Index_t>())};
                         return GenericLinearOperator(offset, std::span<const Real>(array.data(), nb_entries),
                                                        nb_stencil_pts, nb_input_components, nb_quad_pts, nb_output_components);
                    }),
                 "offset"_a, "coefficients"_a)
            // Host field overloads (always available)
            .def("apply",
                 static_cast<ApplyHostFn>(&GenericLinearOperator::apply),
                 "nodal_field"_a, "quadrature_point_field"_a,
                 "Apply gradient operator to host (CPU) fields")
            .def("transpose",
                 static_cast<TransposeHostFn>(&GenericLinearOperator::transpose),
                 "quadrature_point_field"_a, "nodal_field"_a,
                 "weights"_a = std::vector<Real>{},
                 "Apply transpose (divergence) to host (CPU) fields")
            .def("fourier",
                 [](const GenericLinearOperator & op,
                    py::array_t<Real, py::array::f_style> phases) {
                     // Extract buffer info from input phases array
                     py::buffer_info phases_info = phases.request();

                     // Validate input shape
                     if (phases_info.ndim < 1) {
                         throw std::runtime_error("phases array must have at least 1 dimension");
                     }

                     // First dimension = spatial dimensions; remaining = entries
                     py::ssize_t nb_components = phases_info.shape[0];
                     if (nb_components != op.get_spatial_dim()) {
                         std::ostringstream err_msg;
                         err_msg << "Phase dimension mismatch: expected "
                                 << op.get_spatial_dim() << " but got "
                                 << nb_components;
                         throw std::runtime_error(err_msg.str());
                     }

                     // Calculate number of phase vectors and output shape
                     py::ssize_t nb_entries = 1;
                     std::vector<py::ssize_t> output_shape;
                     for (py::ssize_t i = 1; i < phases_info.ndim; ++i) {
                         output_shape.push_back(phases_info.shape[i]);
                         nb_entries *= phases_info.shape[i];
                     }

                     // Create output array for Complex coefficients
                     py::array_t<muGrid::Complex, py::array::f_style> coeffs(output_shape);
                     py::buffer_info coeffs_info = coeffs.request();

                     // Loop over all entries, calling fourier() for each phase vector
                     auto phase_ptr = static_cast<const Real *>(phases_info.ptr);
                     auto coeffs_ptr = static_cast<muGrid::Complex *>(coeffs_info.ptr);

                     for (py::ssize_t i = 0; i < nb_entries; ++i) {
                         // Map phase data to Eigen vector
                         Eigen::Map<const Eigen::VectorXd> phase_vec(phase_ptr, nb_components);
                         coeffs_ptr[i] = op.fourier(phase_vec);
                         phase_ptr += nb_components;
                     }

                     return coeffs;
                 },
                 "phases"_a,
                 R"pbdoc(
                 Compute the Fourier representation of this convolution operator.

                 This method converts a translationally invariant linear combination of
                 grid values into a multiplication with a complex number in Fourier space.

                 Parameters
                 ----------
                 phases : numpy.ndarray
                     Array of phase vectors. First dimension must match spatial_dim.
                     Remaining dimensions represent the batch of phase vectors.
                     Each phase is the wavevector times cell dimension (lacking factor of 2π).

                     Examples:
                     - 1D: shape (1,) for single phase, (1, N) for N phases
                     - 2D: shape (2,) for single phase, (2, N) or (2, M, N) for batches
                     - 3D: shape (3,) for single phase, (3, N, M, K) for batches

                 Returns
                 -------
                 numpy.ndarray of complex128
                     Complex Fourier coefficients with shape matching the batch dimensions
                     (i.e., input shape with first dimension removed).

                 Examples
                 --------
                 >>> # Single phase vector in 2D
                 >>> phase = np.array([0.25, 0.5])
                 >>> coeff = op.fourier(phase)  # Returns scalar complex number
                 >>>
                 >>> # Multiple phase vectors in 2D
                 >>> phases = np.array([[0.1, 0.2, 0.3],
                 ...                    [0.4, 0.5, 0.6]])  # shape (2, 3)
                 >>> coeffs = op.fourier(phases)  # Returns array of shape (3,)
                 >>>
                 >>> # Grid of phase vectors in 2D
                 >>> qx, qy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
                 >>> phases = np.stack([qx, qy], axis=0)  # shape (2, 10, 10)
                 >>> coeffs = op.fourier(phases)  # Returns array of shape (10, 10)
                 )pbdoc")
            .def_property_readonly("offset",
                [](const GenericLinearOperator & op) {
                    const auto& offset = op.get_offset();
                    return py::array_t<Index_t>(offset.size(), offset.data());
                },
                "Stencil offset in number of pixels")
            .def_property_readonly("stencil_shape",
                [](const GenericLinearOperator & op) {
                    const auto& shape = op.get_stencil_shape();
                    return py::array_t<Index_t>(shape.size(), shape.data());
                },
                "Shape of the convolution stencil")
            .def_property_readonly("coefficients",
                [](const GenericLinearOperator & op) {
                    const auto& flat_op = op.get_coefficients();
                    const auto& stencil_shape = op.get_stencil_shape();
                    const auto nb_output_components = op.get_nb_output_components();
                    const auto nb_quad_pts = op.get_nb_quad_pts();
                    const auto nb_input_components = op.get_nb_input_components();

                    // Build the full shape: (nb_output_components, nb_quad_pts, nb_input_components, *stencil_shape)
                    std::vector<py::ssize_t> full_shape;
                    full_shape.push_back(nb_output_components);
                    full_shape.push_back(nb_quad_pts);
                    full_shape.push_back(nb_input_components);
                    for (const auto& dim : stencil_shape) {
                        full_shape.push_back(dim);
                    }

                    // Create a Fortran-ordered (column-major) array
                    py::array_t<Real, py::array::f_style> result(full_shape);
                    auto result_ptr = result.mutable_data();

                    // Copy data
                    std::copy(flat_op.begin(), flat_op.end(), result_ptr);

                    return result;
                },
                R"pbdoc(
                Stencil coefficients in reshaped form.

                Returns the stencil coefficients with shape
                (nb_operators, nb_quad_pts, nb_nodal_pts, *shape),
                where shape contains the spatial dimensions of the stencil.

                Returns
                -------
                numpy.ndarray
                    Stencil coefficients with shape matching the original stencil structure.

                Examples
                --------
                >>> # 2D Laplacian stencil
                >>> stencil_2d = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                >>> op = muGrid.GenericLinearOperator([-1, -1], stencil_2d)
                >>> reshaped = op.coefficients
                >>> reshaped.shape  # (1, 1, 1, 3, 3) for 1 output component, 1 quad pt, 1 input component
                )pbdoc")
            .def_property_readonly("spatial_dim", &GenericLinearOperator::get_spatial_dim)
            .def_property_readonly("nb_quad_pts", &GenericLinearOperator::get_nb_quad_pts)
            .def_property_readonly("nb_input_components", &GenericLinearOperator::get_nb_input_components)
            .def_property_readonly("nb_output_components", &GenericLinearOperator::get_nb_output_components);

    // Single-precision (float32) host overloads.
    {
        using F = TypedFieldBase<Real32>;
        conv_op
            .def("apply",
                 static_cast<void (GenericLinearOperator::*)(const F &, F &)
                                 const>(&GenericLinearOperator::apply),
                 "nodal_field"_a, "quadrature_point_field"_a,
                 "Apply convolution to host float32 fields")
            .def("transpose",
                 static_cast<void (GenericLinearOperator::*)(
                     const F &, F &, const std::vector<Real> &) const>(
                     &GenericLinearOperator::transpose),
                 "quadrature_point_field"_a, "nodal_field"_a,
                 "weights"_a = std::vector<Real>{},
                 "Apply transpose convolution to host float32 fields");
    }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    // Device field overloads (only when GPU backend is enabled)
    using ApplyDeviceFn = void (GenericLinearOperator::*)(
        const RealFieldDevice &, RealFieldDevice &) const;
    using TransposeDeviceFn = void (GenericLinearOperator::*)(
        const RealFieldDevice &, RealFieldDevice &, const std::vector<Real> &)
        const;

    conv_op
        .def("apply",
             static_cast<ApplyDeviceFn>(&GenericLinearOperator::apply),
             "nodal_field"_a, "quadrature_point_field"_a,
             "Apply convolution to device (GPU) fields")
        .def(
            "transpose",
            static_cast<TransposeDeviceFn>(&GenericLinearOperator::transpose),
            "quadrature_point_field"_a, "nodal_field"_a,
            "weights"_a = std::vector<Real>{},
            "Apply transpose convolution to device (GPU) fields");

    // Single-precision (float32) device overloads.
    {
        using FD = TypedFieldBase<Real32, muGrid::DefaultDeviceSpace>;
        conv_op
            .def("apply",
                 static_cast<void (GenericLinearOperator::*)(const FD &, FD &)
                                 const>(&GenericLinearOperator::apply),
                 "nodal_field"_a, "quadrature_point_field"_a,
                 "Apply convolution to device float32 fields")
            .def("transpose",
                 static_cast<void (GenericLinearOperator::*)(
                     const FD &, FD &, const std::vector<Real> &) const>(
                     &GenericLinearOperator::transpose),
                 "quadrature_point_field"_a, "nodal_field"_a,
                 "weights"_a = std::vector<Real>{},
                 "Apply transpose convolution to device float32 fields");
    }
#endif
}

// Add the single-precision (float32) apply/transpose overloads to an
// already-bound LaplaceOperator class (host, and device under CUDA/HIP).
template <class Op>
static void add_laplace_real32_overloads(py::class_<Op, LinearOperator> & cls) {
    using F = TypedFieldBase<Real32>;
    cls.def("apply",
            static_cast<void (Op::*)(const F &, F &) const>(&Op::apply),
            "input_field"_a, "output_field"_a,
            "Single-precision (float32) apply")
        .def("apply_increment",
             static_cast<void (Op::*)(const F &, const Real32 &, F &) const>(
                 &Op::apply_increment),
             "input_field"_a, "alpha"_a, "output_field"_a)
        .def("transpose",
             static_cast<void (Op::*)(const F &, F &,
                                      const std::vector<Real> &) const>(
                 &Op::transpose),
             "input_field"_a, "output_field"_a,
             "weights"_a = std::vector<Real>{});
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    using FD = TypedFieldBase<Real32, muGrid::DefaultDeviceSpace>;
    cls.def("apply",
            static_cast<void (Op::*)(const FD &, FD &) const>(&Op::apply),
            "input_field"_a, "output_field"_a,
            "Single-precision (float32) apply on device fields")
        .def("apply_increment",
             static_cast<void (Op::*)(const FD &, const Real32 &, FD &) const>(
                 &Op::apply_increment),
             "input_field"_a, "alpha"_a, "output_field"_a);
#endif
}

// Bind class LaplaceOperator2D (dimension-specific)
void add_laplace_operator_2d(py::module & mod) {
    using ApplyHostFn = void (LaplaceOperator2D::*)(const RealFieldHost &,
                                                    RealFieldHost &) const;
    using ApplyIncrementHostFn =
        void (LaplaceOperator2D::*)(const RealFieldHost &, const Real &,
                                    RealFieldHost &) const;

    auto laplace_op =
        py::class_<LaplaceOperator2D, LinearOperator>(mod,
                                                        "LaplaceOperator2D",
                                                        R"pbdoc(
        Optimized 2D Laplace operator with hard-coded 5-point stencil.

        This operator provides an optimized implementation of the discrete Laplace
        operator using a 5-point stencil: [0,1,0; 1,-4,1; 0,1,0].

        For new code, prefer using this class directly instead of LaplaceOperator(2)
        for slightly better performance (avoids virtual dispatch).
        )pbdoc")
            .def(py::init<Real>(), "scale"_a = 1.0,
                 "Construct a 2D Laplace operator with optional scale factor")
            .def("apply", static_cast<ApplyHostFn>(&LaplaceOperator2D::apply),
                 "input_field"_a, "output_field"_a,
                 "Apply the Laplace operator to host (CPU) fields")
            .def("apply_increment",
                 static_cast<ApplyIncrementHostFn>(
                     &LaplaceOperator2D::apply_increment),
                 "input_field"_a, "alpha"_a, "output_field"_a,
                 "Add alpha * Laplace(input) to output on host (CPU) fields")
            .def("transpose",
                 static_cast<void (LaplaceOperator2D::*)(
                     const RealFieldHost &, RealFieldHost &,
                     const std::vector<Real> &) const>(
                     &LaplaceOperator2D::transpose),
                 "input_field"_a, "output_field"_a,
                 "weights"_a = std::vector<Real>{},
                 "Apply the transpose; identical to apply for the self-adjoint "
                 "Laplacian (weights are ignored)")
            .def_property_readonly("nb_stencil_pts",
                                   &LaplaceOperator2D::get_nb_stencil_pts)
            .def_property_readonly("scale", &LaplaceOperator2D::get_scale)
            .def_property_readonly("offset",
                                   [](const LaplaceOperator2D & op) {
                                       const auto & offset = op.get_offset();
                                       return py::array_t<Index_t>(
                                           offset.size(), offset.data());
                                   })
            .def_property_readonly(
                "stencil_shape",
                [](const LaplaceOperator2D & op) {
                    const auto & shape = op.get_stencil_shape();
                    return py::array_t<Index_t>(shape.size(), shape.data());
                })
            .def_property_readonly(
                "coefficients", [](const LaplaceOperator2D & op) {
                    const auto & flat_op = op.get_coefficients();
                    const auto & stencil_shape = op.get_stencil_shape();
                    std::vector<py::ssize_t> full_shape{1, 1, 1};
                    for (const auto & dim : stencil_shape) {
                        full_shape.push_back(dim);
                    }
                    py::array_t<Real, py::array::f_style> result(full_shape);
                    std::copy(flat_op.begin(), flat_op.end(),
                              result.mutable_data());
                    return result;
                });

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    using ApplyDeviceFn = void (LaplaceOperator2D::*)(const RealFieldDevice &,
                                                      RealFieldDevice &) const;
    laplace_op.def("apply",
                   static_cast<ApplyDeviceFn>(&LaplaceOperator2D::apply),
                   "input_field"_a, "output_field"_a,
                   "Apply the Laplace operator to device (GPU) fields");
#endif
    add_laplace_real32_overloads(laplace_op);
}

// Bind class LaplaceOperator3D (dimension-specific)
void add_laplace_operator_3d(py::module & mod) {
    using ApplyHostFn = void (LaplaceOperator3D::*)(const RealFieldHost &,
                                                    RealFieldHost &) const;
    using ApplyIncrementHostFn =
        void (LaplaceOperator3D::*)(const RealFieldHost &, const Real &,
                                    RealFieldHost &) const;

    auto laplace_op =
        py::class_<LaplaceOperator3D, LinearOperator>(mod,
                                                        "LaplaceOperator3D",
                                                        R"pbdoc(
        Optimized 3D Laplace operator with hard-coded 7-point stencil.

        This operator provides an optimized implementation of the discrete Laplace
        operator using a 7-point stencil: center=-6, neighbors=+1.

        For new code, prefer using this class directly instead of LaplaceOperator(3)
        for slightly better performance (avoids virtual dispatch).
        )pbdoc")
            .def(py::init<Real>(), "scale"_a = 1.0,
                 "Construct a 3D Laplace operator with optional scale factor")
            .def("apply", static_cast<ApplyHostFn>(&LaplaceOperator3D::apply),
                 "input_field"_a, "output_field"_a,
                 "Apply the Laplace operator to host (CPU) fields")
            .def("apply_increment",
                 static_cast<ApplyIncrementHostFn>(
                     &LaplaceOperator3D::apply_increment),
                 "input_field"_a, "alpha"_a, "output_field"_a,
                 "Add alpha * Laplace(input) to output on host (CPU) fields")
            .def("transpose",
                 static_cast<void (LaplaceOperator3D::*)(
                     const RealFieldHost &, RealFieldHost &,
                     const std::vector<Real> &) const>(
                     &LaplaceOperator3D::transpose),
                 "input_field"_a, "output_field"_a,
                 "weights"_a = std::vector<Real>{},
                 "Apply the transpose; identical to apply for the self-adjoint "
                 "Laplacian (weights are ignored)")
            .def_property_readonly("nb_stencil_pts",
                                   &LaplaceOperator3D::get_nb_stencil_pts)
            .def_property_readonly("scale", &LaplaceOperator3D::get_scale)
            .def_property_readonly("offset",
                                   [](const LaplaceOperator3D & op) {
                                       const auto & offset = op.get_offset();
                                       return py::array_t<Index_t>(
                                           offset.size(), offset.data());
                                   })
            .def_property_readonly(
                "stencil_shape",
                [](const LaplaceOperator3D & op) {
                    const auto & shape = op.get_stencil_shape();
                    return py::array_t<Index_t>(shape.size(), shape.data());
                })
            .def_property_readonly(
                "coefficients", [](const LaplaceOperator3D & op) {
                    const auto & flat_op = op.get_coefficients();
                    const auto & stencil_shape = op.get_stencil_shape();
                    std::vector<py::ssize_t> full_shape{1, 1, 1};
                    for (const auto & dim : stencil_shape) {
                        full_shape.push_back(dim);
                    }
                    py::array_t<Real, py::array::f_style> result(full_shape);
                    std::copy(flat_op.begin(), flat_op.end(),
                              result.mutable_data());
                    return result;
                });

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    using ApplyDeviceFn = void (LaplaceOperator3D::*)(const RealFieldDevice &,
                                                      RealFieldDevice &) const;
    laplace_op.def("apply",
                   static_cast<ApplyDeviceFn>(&LaplaceOperator3D::apply),
                   "input_field"_a, "output_field"_a,
                   "Apply the Laplace operator to device (GPU) fields");
#endif
    add_laplace_real32_overloads(laplace_op);
}

// Bind a FEMGradientOperator<Element> instantiation. The 2D/3D and
// simplex/Q1 bindings are identical apart from the type and the docstring,
// so they share this template.
template <class Op>
static void bind_fem_gradient_operator(py::module & mod, const char * name,
                                       const char * doc) {
    // The operator's scalar type (Real or Real32); the host/device field types
    // and the apply/transpose argument types follow from it, so one helper
    // binds both the double- and single-precision instantiations.
    using T = typename Op::Scalar;
    using FieldHost = TypedFieldBase<T>;
    using ApplyHostFn =
        void (Op::*)(const FieldHost &, FieldHost &) const;
    using TransposeHostFn = void (Op::*)(
        const FieldHost &, FieldHost &, const std::vector<Real> &) const;
    using ApplyIncrementHostFn =
        void (Op::*)(const FieldHost &, const T &, FieldHost &) const;
    using TransposeIncrementHostFn =
        void (Op::*)(const FieldHost &, const T &, FieldHost &,
                     const std::vector<Real> &) const;

    auto fem_grad_op =
        py::class_<Op, LinearOperator>(mod, name, doc)
            .def(py::init<std::vector<Real>>(),
                 "grid_spacing"_a = std::vector<Real>{},
                 "Construct with optional grid spacing")
            .def("apply", static_cast<ApplyHostFn>(&Op::apply), "nodal_field"_a,
                 "gradient_field"_a, "Apply gradient operator to host fields")
            .def("apply_increment",
                 static_cast<ApplyIncrementHostFn>(&Op::apply_increment),
                 "nodal_field"_a, "alpha"_a, "gradient_field"_a,
                 "Add alpha * grad(nodal_field) to gradient_field on host "
                 "fields")
            .def("transpose", static_cast<TransposeHostFn>(&Op::transpose),
                 "gradient_field"_a, "nodal_field"_a,
                 "weights"_a = std::vector<Real>{},
                 "Apply transpose (divergence) to host fields")
            .def("transpose_increment",
                 static_cast<TransposeIncrementHostFn>(&Op::transpose_increment),
                 "gradient_field"_a, "alpha"_a, "nodal_field"_a,
                 "weights"_a = std::vector<Real>{},
                 "Add alpha * (-div(gradient_field)) to nodal_field on host "
                 "fields")
            .def_property_readonly("grid_spacing", &Op::get_grid_spacing)
            .def_property_readonly("quadrature_weights",
                                   &Op::get_quadrature_weights)
            .def_property_readonly("offset",
                                   [](const Op & op) {
                                       const auto & offset = op.get_offset();
                                       return py::array_t<Index_t>(
                                           offset.size(), offset.data());
                                   })
            .def_property_readonly("stencil_shape",
                                   [](const Op & op) {
                                       const auto & shape =
                                           op.get_stencil_shape();
                                       return py::array_t<Index_t>(
                                           shape.size(), shape.data());
                                   })
            .def_property_readonly("coefficients", [](const Op & op) {
                const auto & flat_op = op.get_coefficients();
                const auto & stencil_shape = op.get_stencil_shape();
                const auto nb_output = op.get_nb_output_components();
                const auto nb_quad = op.get_nb_quad_pts();
                const auto nb_input = op.get_nb_input_components();
                std::vector<py::ssize_t> full_shape{nb_output, nb_quad,
                                                    nb_input};
                for (const auto & dim : stencil_shape) {
                    full_shape.push_back(dim);
                }
                py::array_t<Real, py::array::f_style> result(full_shape);
                std::copy(flat_op.begin(), flat_op.end(), result.mutable_data());
                return result;
            });

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    using FieldDevice = TypedFieldBase<T, muGrid::DefaultDeviceSpace>;
    using ApplyDeviceFn =
        void (Op::*)(const FieldDevice &, FieldDevice &) const;
    using TransposeDeviceFn = void (Op::*)(
        const FieldDevice &, FieldDevice &, const std::vector<Real> &)
        const;
    using ApplyIncrementDeviceFn =
        void (Op::*)(const FieldDevice &, const T &, FieldDevice &)
            const;
    using TransposeIncrementDeviceFn =
        void (Op::*)(const FieldDevice &, const T &, FieldDevice &,
                     const std::vector<Real> &) const;
    fem_grad_op
        .def("apply", static_cast<ApplyDeviceFn>(&Op::apply), "nodal_field"_a,
             "gradient_field"_a, "Apply gradient operator to device fields")
        .def("apply_increment",
             static_cast<ApplyIncrementDeviceFn>(&Op::apply_increment),
             "nodal_field"_a, "alpha"_a, "gradient_field"_a,
             "Add alpha * grad(nodal_field) to gradient_field on device fields")
        .def("transpose", static_cast<TransposeDeviceFn>(&Op::transpose),
             "gradient_field"_a, "nodal_field"_a,
             "weights"_a = std::vector<Real>{},
             "Apply transpose to device fields")
        .def("transpose_increment",
             static_cast<TransposeIncrementDeviceFn>(&Op::transpose_increment),
             "gradient_field"_a, "alpha"_a, "nodal_field"_a,
             "weights"_a = std::vector<Real>{},
             "Add alpha * (-div(gradient_field)) to nodal_field on device "
             "fields");
#endif
}

// Bind the FEM gradient operators: linear simplices (historical names) and Q1.
void add_fem_gradient_operator_2d(py::module & mod) {
    bind_fem_gradient_operator<FEMGradientOperator2D>(
        mod, "FEMGradientOperator2D",
        "2D linear FEM gradient operator (2 triangles per pixel, 2 quadrature "
        "points).");
    bind_fem_gradient_operator<FEMGradientOperatorQ1_2D>(
        mod, "FEMGradientOperatorQ1_2D",
        "2D Q1 (bilinear quad) FEM gradient operator (2x2 Gauss, 4 quadrature "
        "points).");
    bind_fem_gradient_operator<muGrid::FEMGradientOperator2D_32>(
        mod, "FEMGradientOperator2D_32",
        "Single-precision (float32) 2D linear FEM gradient operator.");
    bind_fem_gradient_operator<muGrid::FEMGradientOperatorQ1_2D_32>(
        mod, "FEMGradientOperatorQ1_2D_32",
        "Single-precision (float32) 2D Q1 FEM gradient operator.");
}

void add_fem_gradient_operator_3d(py::module & mod) {
    bind_fem_gradient_operator<FEMGradientOperator3D>(
        mod, "FEMGradientOperator3D",
        "3D linear FEM gradient operator (5 Kuhn tetrahedra per voxel, 5 "
        "quadrature points).");
    bind_fem_gradient_operator<FEMGradientOperatorQ1_3D>(
        mod, "FEMGradientOperatorQ1_3D",
        "3D Q1 (trilinear hex) FEM gradient operator (2x2x2 Gauss, 8 "
        "quadrature points).");
    bind_fem_gradient_operator<muGrid::FEMGradientOperator3D_32>(
        mod, "FEMGradientOperator3D_32",
        "Single-precision (float32) 3D linear FEM gradient operator.");
    bind_fem_gradient_operator<muGrid::FEMGradientOperatorQ1_3D_32>(
        mod, "FEMGradientOperatorQ1_3D_32",
        "Single-precision (float32) 3D Q1 FEM gradient operator.");
}

// Add the single-precision (Real32) apply overloads to an already-bound
// IsotropicStiffnessOperator class. The operator is monomorphic per call: the
// same C++ object serves both precisions through overloaded apply methods, so
// this just registers the float32-field overloads alongside the double ones.
// DD is Dim*Dim (the flattened macro-strain length: 4 in 2D, 9 in 3D).
template <class Op, std::size_t DD>
static void add_stiffness_real32_overloads(py::class_<Op> & cls) {
    using F = TypedFieldBase<Real32>;
    cls.def("apply",
            static_cast<void (Op::*)(const F &, const F &, const F &, F &)
                            const>(&Op::apply),
            "displacement"_a, "lambda_field"_a, "mu_field"_a, "force"_a,
            "Single-precision (float32) apply: force = K @ displacement")
        .def("apply_increment",
             static_cast<void (Op::*)(const F &, const F &, const F &, Real32,
                                      F &) const>(&Op::apply_increment),
             "displacement"_a, "lambda_field"_a, "mu_field"_a, "alpha"_a,
             "force"_a)
        .def("apply_uniform",
             static_cast<void (Op::*)(const F &, Real32, Real32, F &) const>(
                 &Op::apply_uniform),
             "displacement"_a, "lambda"_a, "mu"_a, "force"_a)
        .def("apply_uniform_increment",
             static_cast<void (Op::*)(const F &, Real32, Real32, Real32, F &)
                             const>(&Op::apply_uniform_increment),
             "displacement"_a, "lambda"_a, "mu"_a, "alpha"_a, "force"_a)
        .def("apply_macro_rhs",
             static_cast<void (Op::*)(const F &, const F &,
                                      const std::array<Real, DD> &, F &) const>(
                 &Op::apply_macro_rhs),
             "lambda_field"_a, "mu_field"_a, "E_macro"_a, "force"_a)
        .def("average_stress",
             static_cast<std::array<Real, DD> (Op::*)(
                 const F &, const F &, const F &, const std::array<Real, DD> &)
                             const>(&Op::average_stress),
             "displacement"_a, "lambda_field"_a, "mu_field"_a, "E_macro"_a)
        .def("assemble_diagonal",
             static_cast<void (Op::*)(const F &, const F &, F &) const>(
                 &Op::assemble_diagonal),
             "lambda_field"_a, "mu_field"_a, "diagonal"_a)
        .def("compute_sensitivity",
             static_cast<void (Op::*)(
                 const F &, const std::array<Real, DD> &, const F &,
                 const std::array<Real, DD> &, F &, F &) const>(
                 &Op::compute_sensitivity),
             "forward_disp"_a, "forward_macro"_a, "costate_disp"_a,
             "costate_macro"_a, "g_shear"_a, "g_vol"_a);
}

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
// Single-precision (float32) device apply overloads on an IsotropicStiffness
// class (mirrors add_stiffness_real32_overloads for device fields).
template <class Op, std::size_t DD>
static void add_stiffness_real32_device_overloads(py::class_<Op> & cls) {
    using F = TypedFieldBase<Real32, muGrid::DefaultDeviceSpace>;
    cls.def("apply",
            static_cast<void (Op::*)(const F &, const F &, const F &, F &)
                            const>(&Op::apply),
            "displacement"_a, "lambda_field"_a, "mu_field"_a, "force"_a,
            "Single-precision (float32) apply on device fields")
        .def("apply_increment",
             static_cast<void (Op::*)(const F &, const F &, const F &, Real32,
                                      F &) const>(&Op::apply_increment),
             "displacement"_a, "lambda_field"_a, "mu_field"_a, "alpha"_a,
             "force"_a)
        .def("apply_uniform",
             static_cast<void (Op::*)(const F &, Real32, Real32, F &) const>(
                 &Op::apply_uniform),
             "displacement"_a, "lambda"_a, "mu"_a, "force"_a)
        .def("apply_uniform_increment",
             static_cast<void (Op::*)(const F &, Real32, Real32, Real32, F &)
                             const>(&Op::apply_uniform_increment),
             "displacement"_a, "lambda"_a, "mu"_a, "alpha"_a, "force"_a)
        .def("apply_macro_rhs",
             static_cast<void (Op::*)(const F &, const F &,
                                      const std::array<Real, DD> &, F &) const>(
                 &Op::apply_macro_rhs),
             "lambda_field"_a, "mu_field"_a, "E_macro"_a, "force"_a)
        .def("average_stress",
             static_cast<std::array<Real, DD> (Op::*)(
                 const F &, const F &, const F &, const std::array<Real, DD> &)
                             const>(&Op::average_stress),
             "displacement"_a, "lambda_field"_a, "mu_field"_a, "E_macro"_a)
        .def("assemble_diagonal",
             static_cast<void (Op::*)(const F &, const F &, F &) const>(
                 &Op::assemble_diagonal),
             "lambda_field"_a, "mu_field"_a, "diagonal"_a)
        .def("compute_sensitivity",
             static_cast<void (Op::*)(
                 const F &, const std::array<Real, DD> &, const F &,
                 const std::array<Real, DD> &, F &, F &) const>(
                 &Op::compute_sensitivity),
             "forward_disp"_a, "forward_macro"_a, "costate_disp"_a,
             "costate_macro"_a, "g_shear"_a, "g_vol"_a);
}
#endif

// Bind class IsotropicStiffnessOperator2D
void add_isotropic_stiffness_operator_2d(py::module & mod) {
    // The finite element used by the fused stiffness operator (selected at
    // construction; affects only the precomputed geometry, not the kernels).
    py::enum_<FEMElementKind>(mod, "FEMElement",
                              "Finite element for the fused stiffness operator")
        .value("p1", FEMElementKind::P1,
               "P1 linear simplices (2 triangles / 5 tetrahedra per pixel)")
        .value("q1", FEMElementKind::Q1,
               "Q1 (bilinear quad / trilinear hex) with Gauss quadrature");

    // Function pointer types for explicit overload selection
    using ApplyHostFn = void (IsotropicStiffnessOperator2D::*)(
        const RealFieldHost &, const RealFieldHost &, const RealFieldHost &,
        RealFieldHost &) const;
    using ApplyIncrementHostFn = void (IsotropicStiffnessOperator2D::*)(
        const RealFieldHost &, const RealFieldHost &, const RealFieldHost &,
        Real, RealFieldHost &) const;

    auto op =
        py::class_<IsotropicStiffnessOperator2D>(mod,
                                                 "IsotropicStiffnessOperator2D",
                                                 R"pbdoc(
        Fused stiffness operator for 2D isotropic linear elastic materials.

        This operator computes K @ u = B^T C B @ u for 2D linear triangular
        elements, where C is the isotropic elasticity tensor parameterized by
        Lamé constants λ (lambda) and μ (mu).

        Instead of storing the full stiffness matrix K, it exploits the
        isotropic structure: K = 2μ G + λ V, where G and V are geometry-only
        matrices precomputed at construction time.

        This reduces memory from O(N × 64) for full K storage to O(N × 2) for
        spatially-varying isotropic materials, plus O(1) for the shared G and V.

        Parameters
        ----------
        grid_spacing : list of float
            Grid spacing [hx, hy] in each direction.

        Notes
        -----
        - Displacement field shape: [2, nx, ny] (2 DOFs per node)
        - Material fields (lambda, mu) shape: [nx-1, ny-1] (one value per pixel)
        - Force field shape: [2, nx, ny] (same as displacement)
        )pbdoc")
            .def(py::init<const std::vector<Real> &, FEMElementKind>(),
                 "grid_spacing"_a,
                 "element"_a = FEMElementKind::Q1,
                 "Construct with grid spacing [hx, hy] and element type")
            .def("apply",
                 static_cast<ApplyHostFn>(&IsotropicStiffnessOperator2D::apply),
                 "displacement"_a, "lambda_field"_a, "mu_field"_a, "force"_a,
                 "Apply stiffness operator: force = K @ displacement")
            .def("apply_increment",
                 static_cast<ApplyIncrementHostFn>(
                     &IsotropicStiffnessOperator2D::apply_increment),
                 "displacement"_a, "lambda_field"_a, "mu_field"_a, "alpha"_a,
                 "force"_a,
                 "Apply with increment: force += alpha * K @ displacement")
            .def("apply_uniform",
                 static_cast<void (IsotropicStiffnessOperator2D::*)(
                     const RealFieldHost &, Real, Real, RealFieldHost &) const>(
                     &IsotropicStiffnessOperator2D::apply_uniform),
                 "displacement"_a, "lambda"_a, "mu"_a, "force"_a,
                 "Apply with spatially uniform Lamé scalars (no material "
                 "fields): force = K(lambda, mu) @ displacement")
            .def("apply_uniform_increment",
                 static_cast<void (IsotropicStiffnessOperator2D::*)(
                     const RealFieldHost &, Real, Real, Real, RealFieldHost &)
                                 const>(
                     &IsotropicStiffnessOperator2D::apply_uniform_increment),
                 "displacement"_a, "lambda"_a, "mu"_a, "alpha"_a, "force"_a,
                 "force += alpha * K(lambda, mu) @ displacement, uniform "
                 "scalars")
            .def("apply_macro_rhs",
                 static_cast<void (IsotropicStiffnessOperator2D::*)(
                     const RealFieldHost &, const RealFieldHost &,
                     const std::array<Real, 4> &, RealFieldHost &) const>(
                     &IsotropicStiffnessOperator2D::apply_macro_rhs),
                 "lambda_field"_a, "mu_field"_a, "E_macro"_a, "force"_a,
                 "Assemble force = B^T C(lambda, mu) E_macro (the divergence "
                 "of the macro-strain stress); the homogenization RHS is the "
                 "negative of this. E_macro is the flattened 2x2 strain.")
            .def("average_stress",
                 static_cast<std::array<Real, 4> (
                     IsotropicStiffnessOperator2D::*)(
                     const RealFieldHost &, const RealFieldHost &,
                     const RealFieldHost &, const std::array<Real, 4> &) const>(
                     &IsotropicStiffnessOperator2D::average_stress),
                 "displacement"_a, "lambda_field"_a, "mu_field"_a, "E_macro"_a,
                 "Local volume integral of sigma = C:(E_macro + sym grad u), "
                 "returned as a flattened 2x2 tensor. Sum across ranks and "
                 "divide by total volume for the homogenized stress.")
            .def("assemble_diagonal",
                 static_cast<void (IsotropicStiffnessOperator2D::*)(
                     const RealFieldHost &, const RealFieldHost &,
                     RealFieldHost &) const>(
                     &IsotropicStiffnessOperator2D::assemble_diagonal),
                 "lambda_field"_a, "mu_field"_a, "diagonal"_a,
                 "Assemble diag(K) into the nodal field 'diagonal' (same shape "
                 "as displacement/force). This is the Jacobi ingredient of the "
                 "Green-Jacobi (J-FFT) preconditioner.")
            .def("compute_sensitivity",
                 static_cast<void (IsotropicStiffnessOperator2D::*)(
                     const RealFieldHost &, const std::array<Real, 4> &,
                     const RealFieldHost &, const std::array<Real, 4> &,
                     RealFieldHost &, RealFieldHost &) const>(
                     &IsotropicStiffnessOperator2D::compute_sensitivity),
                 "forward_disp"_a, "forward_macro"_a, "costate_disp"_a,
                 "costate_macro"_a, "g_shear"_a, "g_vol"_a,
                 "Per-pixel sensitivity contractions g_shear = a^T G b, g_vol = "
                 "a^T V b of the total forward strain (forward_disp + "
                 "forward_macro) with the total costate strain (costate_disp + "
                 "costate_macro). Multiply by d(2mu)/drho and dlambda/drho for "
                 "the SIMP material-derivative sensitivity.")
            .def_property_readonly(
                "G",
                [](const IsotropicStiffnessOperator2D & op) {
                    const auto & G = op.get_G();
                    return py::array_t<Real>({8, 8}, G.data());
                },
                "Precomputed G matrix (shear stiffness geometry)")
            .def_property_readonly(
                "V",
                [](const IsotropicStiffnessOperator2D & op) {
                    const auto & V = op.get_V();
                    return py::array_t<Real>({8, 8}, V.data());
                },
                "Precomputed V matrix (volumetric stiffness geometry)")
            .def_property_readonly(
                "apply_ghost_requirement",
                [](const IsotropicStiffnessOperator2D & op) {
                    return ghost_requirement_to_python(
                        op.get_apply_ghost_requirement());
                },
                "Ghost layers (left, right) required by apply()")
            .def_property_readonly(
                "ghost_requirement",
                [](const IsotropicStiffnessOperator2D & op) {
                    return ghost_requirement_to_python(
                        op.get_ghost_requirement());
                },
                "Ghost layers (left, right) sufficient for all operations");

    // Single-precision (float32) host apply overloads on the same class.
    add_stiffness_real32_overloads<IsotropicStiffnessOperator2D, 4>(op);

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    using ApplyDeviceFn = void (IsotropicStiffnessOperator2D::*)(
        const RealFieldDevice &, const RealFieldDevice &,
        const RealFieldDevice &, RealFieldDevice &) const;
    using ApplyIncrementDeviceFn = void (IsotropicStiffnessOperator2D::*)(
        const RealFieldDevice &, const RealFieldDevice &,
        const RealFieldDevice &, Real, RealFieldDevice &) const;

    op.def("apply",
           static_cast<ApplyDeviceFn>(&IsotropicStiffnessOperator2D::apply),
           "displacement"_a, "lambda_field"_a, "mu_field"_a, "force"_a,
           "Apply stiffness operator to device (GPU) fields")
        .def("apply_increment",
             static_cast<ApplyIncrementDeviceFn>(
                 &IsotropicStiffnessOperator2D::apply_increment),
             "displacement"_a, "lambda_field"_a, "mu_field"_a, "alpha"_a,
             "force"_a, "Apply with increment to device (GPU) fields")
        .def("apply_uniform",
             static_cast<void (IsotropicStiffnessOperator2D::*)(
                 const RealFieldDevice &, Real, Real, RealFieldDevice &) const>(
                 &IsotropicStiffnessOperator2D::apply_uniform),
             "displacement"_a, "lambda"_a, "mu"_a, "force"_a,
             "Apply with uniform Lamé scalars to device (GPU) fields")
        .def("apply_uniform_increment",
             static_cast<void (IsotropicStiffnessOperator2D::*)(
                 const RealFieldDevice &, Real, Real, Real, RealFieldDevice &)
                             const>(
                 &IsotropicStiffnessOperator2D::apply_uniform_increment),
             "displacement"_a, "lambda"_a, "mu"_a, "alpha"_a, "force"_a,
             "Uniform-scalar increment on device (GPU) fields")
        .def("apply_macro_rhs",
             static_cast<void (IsotropicStiffnessOperator2D::*)(
                 const RealFieldDevice &, const RealFieldDevice &,
                 const std::array<Real, 4> &, RealFieldDevice &) const>(
                 &IsotropicStiffnessOperator2D::apply_macro_rhs),
             "lambda_field"_a, "mu_field"_a, "E_macro"_a, "force"_a,
             "Assemble macro-strain RHS divergence on device (GPU) fields")
        .def("average_stress",
             static_cast<std::array<Real, 4> (IsotropicStiffnessOperator2D::*)(
                 const RealFieldDevice &, const RealFieldDevice &,
                 const RealFieldDevice &, const std::array<Real, 4> &) const>(
                 &IsotropicStiffnessOperator2D::average_stress),
             "displacement"_a, "lambda_field"_a, "mu_field"_a, "E_macro"_a,
             "Local volume integral of stress on device (GPU) fields")
        .def("assemble_diagonal",
             static_cast<void (IsotropicStiffnessOperator2D::*)(
                 const RealFieldDevice &, const RealFieldDevice &,
                 RealFieldDevice &) const>(
                 &IsotropicStiffnessOperator2D::assemble_diagonal),
             "lambda_field"_a, "mu_field"_a, "diagonal"_a,
             "Assemble diag(K) into a device (GPU) nodal field")
        .def("compute_sensitivity",
             static_cast<void (IsotropicStiffnessOperator2D::*)(
                 const RealFieldDevice &, const std::array<Real, 4> &,
                 const RealFieldDevice &, const std::array<Real, 4> &,
                 RealFieldDevice &, RealFieldDevice &) const>(
                 &IsotropicStiffnessOperator2D::compute_sensitivity),
             "forward_disp"_a, "forward_macro"_a, "costate_disp"_a,
             "costate_macro"_a, "g_shear"_a, "g_vol"_a,
             "Per-pixel sensitivity contractions on device (GPU) fields");
    // Single-precision (float32) device apply overloads.
    add_stiffness_real32_device_overloads<IsotropicStiffnessOperator2D, 4>(op);
#endif
}

// Bind class IsotropicStiffnessOperator3D
void add_isotropic_stiffness_operator_3d(py::module & mod) {
    // Function pointer types for explicit overload selection
    using ApplyHostFn = void (IsotropicStiffnessOperator3D::*)(
        const RealFieldHost &, const RealFieldHost &, const RealFieldHost &,
        RealFieldHost &) const;
    using ApplyIncrementHostFn = void (IsotropicStiffnessOperator3D::*)(
        const RealFieldHost &, const RealFieldHost &, const RealFieldHost &,
        Real, RealFieldHost &) const;

    auto op =
        py::class_<IsotropicStiffnessOperator3D>(mod,
                                                 "IsotropicStiffnessOperator3D",
                                                 R"pbdoc(
        Fused stiffness operator for 3D isotropic linear elastic materials.

        This operator computes K @ u = B^T C B @ u for 3D linear tetrahedral
        elements using a 5-tetrahedra decomposition per voxel (Kuhn triangulation).

        The isotropic structure K = 2μ G + λ V is exploited to reduce memory
        requirements from O(N × 576) for full K storage to O(N × 2) for
        spatially-varying isotropic materials.

        Parameters
        ----------
        grid_spacing : list of float
            Grid spacing [hx, hy, hz] in each direction.

        Notes
        -----
        - Displacement field shape: [3, nx, ny, nz] (3 DOFs per node)
        - Material fields (lambda, mu) shape: [nx-1, ny-1, nz-1] (one value per voxel)
        - Force field shape: [3, nx, ny, nz] (same as displacement)
        )pbdoc")
            .def(py::init<const std::vector<Real> &, FEMElementKind>(),
                 "grid_spacing"_a,
                 "element"_a = FEMElementKind::Q1,
                 "Construct with grid spacing [hx, hy, hz] and element type")
            .def("apply",
                 static_cast<ApplyHostFn>(&IsotropicStiffnessOperator3D::apply),
                 "displacement"_a, "lambda_field"_a, "mu_field"_a, "force"_a,
                 "Apply stiffness operator: force = K @ displacement")
            .def("apply_increment",
                 static_cast<ApplyIncrementHostFn>(
                     &IsotropicStiffnessOperator3D::apply_increment),
                 "displacement"_a, "lambda_field"_a, "mu_field"_a, "alpha"_a,
                 "force"_a,
                 "Apply with increment: force += alpha * K @ displacement")
            .def("apply_uniform",
                 static_cast<void (IsotropicStiffnessOperator3D::*)(
                     const RealFieldHost &, Real, Real, RealFieldHost &) const>(
                     &IsotropicStiffnessOperator3D::apply_uniform),
                 "displacement"_a, "lambda"_a, "mu"_a, "force"_a,
                 "Apply with spatially uniform Lamé scalars (no material "
                 "fields): force = K(lambda, mu) @ displacement")
            .def("apply_uniform_increment",
                 static_cast<void (IsotropicStiffnessOperator3D::*)(
                     const RealFieldHost &, Real, Real, Real, RealFieldHost &)
                                 const>(
                     &IsotropicStiffnessOperator3D::apply_uniform_increment),
                 "displacement"_a, "lambda"_a, "mu"_a, "alpha"_a, "force"_a,
                 "force += alpha * K(lambda, mu) @ displacement, uniform "
                 "scalars")
            .def("apply_macro_rhs",
                 static_cast<void (IsotropicStiffnessOperator3D::*)(
                     const RealFieldHost &, const RealFieldHost &,
                     const std::array<Real, 9> &, RealFieldHost &) const>(
                     &IsotropicStiffnessOperator3D::apply_macro_rhs),
                 "lambda_field"_a, "mu_field"_a, "E_macro"_a, "force"_a,
                 "Assemble force = B^T C(lambda, mu) E_macro (the divergence "
                 "of the macro-strain stress); the homogenization RHS is the "
                 "negative of this. E_macro is the flattened 3x3 strain.")
            .def("average_stress",
                 static_cast<std::array<Real, 9> (
                     IsotropicStiffnessOperator3D::*)(
                     const RealFieldHost &, const RealFieldHost &,
                     const RealFieldHost &, const std::array<Real, 9> &) const>(
                     &IsotropicStiffnessOperator3D::average_stress),
                 "displacement"_a, "lambda_field"_a, "mu_field"_a, "E_macro"_a,
                 "Local volume integral of sigma = C:(E_macro + sym grad u), "
                 "returned as a flattened 3x3 tensor. Sum across ranks and "
                 "divide by total volume for the homogenized stress.")
            .def("assemble_diagonal",
                 static_cast<void (IsotropicStiffnessOperator3D::*)(
                     const RealFieldHost &, const RealFieldHost &,
                     RealFieldHost &) const>(
                     &IsotropicStiffnessOperator3D::assemble_diagonal),
                 "lambda_field"_a, "mu_field"_a, "diagonal"_a,
                 "Assemble diag(K) into the nodal field 'diagonal' (same shape "
                 "as displacement/force). This is the Jacobi ingredient of the "
                 "Green-Jacobi (J-FFT) preconditioner.")
            .def("compute_sensitivity",
                 static_cast<void (IsotropicStiffnessOperator3D::*)(
                     const RealFieldHost &, const std::array<Real, 9> &,
                     const RealFieldHost &, const std::array<Real, 9> &,
                     RealFieldHost &, RealFieldHost &) const>(
                     &IsotropicStiffnessOperator3D::compute_sensitivity),
                 "forward_disp"_a, "forward_macro"_a, "costate_disp"_a,
                 "costate_macro"_a, "g_shear"_a, "g_vol"_a,
                 "Per-pixel sensitivity contractions g_shear = a^T G b, g_vol = "
                 "a^T V b of the total forward strain with the total costate "
                 "strain. Multiply by d(2mu)/drho and dlambda/drho for the SIMP "
                 "material-derivative sensitivity.")
            .def_property_readonly(
                "G",
                [](const IsotropicStiffnessOperator3D & op) {
                    const auto & G = op.get_G();
                    return py::array_t<Real>({24, 24}, G.data());
                },
                "Precomputed G matrix (shear stiffness geometry)")
            .def_property_readonly(
                "V",
                [](const IsotropicStiffnessOperator3D & op) {
                    const auto & V = op.get_V();
                    return py::array_t<Real>({24, 24}, V.data());
                },
                "Precomputed V matrix (volumetric stiffness geometry)")
            .def_property_readonly(
                "apply_ghost_requirement",
                [](const IsotropicStiffnessOperator3D & op) {
                    return ghost_requirement_to_python(
                        op.get_apply_ghost_requirement());
                },
                "Ghost layers (left, right) required by apply()")
            .def_property_readonly(
                "ghost_requirement",
                [](const IsotropicStiffnessOperator3D & op) {
                    return ghost_requirement_to_python(
                        op.get_ghost_requirement());
                },
                "Ghost layers (left, right) sufficient for all operations");

    // Single-precision (float32) host apply overloads on the same class.
    add_stiffness_real32_overloads<IsotropicStiffnessOperator3D, 9>(op);

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    using ApplyDeviceFn = void (IsotropicStiffnessOperator3D::*)(
        const RealFieldDevice &, const RealFieldDevice &,
        const RealFieldDevice &, RealFieldDevice &) const;
    using ApplyIncrementDeviceFn = void (IsotropicStiffnessOperator3D::*)(
        const RealFieldDevice &, const RealFieldDevice &,
        const RealFieldDevice &, Real, RealFieldDevice &) const;

    op.def("apply",
           static_cast<ApplyDeviceFn>(&IsotropicStiffnessOperator3D::apply),
           "displacement"_a, "lambda_field"_a, "mu_field"_a, "force"_a,
           "Apply stiffness operator to device (GPU) fields")
        .def("apply_increment",
             static_cast<ApplyIncrementDeviceFn>(
                 &IsotropicStiffnessOperator3D::apply_increment),
             "displacement"_a, "lambda_field"_a, "mu_field"_a, "alpha"_a,
             "force"_a, "Apply with increment to device (GPU) fields")
        .def("apply_uniform",
             static_cast<void (IsotropicStiffnessOperator3D::*)(
                 const RealFieldDevice &, Real, Real, RealFieldDevice &) const>(
                 &IsotropicStiffnessOperator3D::apply_uniform),
             "displacement"_a, "lambda"_a, "mu"_a, "force"_a,
             "Apply with uniform Lamé scalars to device (GPU) fields")
        .def("apply_uniform_increment",
             static_cast<void (IsotropicStiffnessOperator3D::*)(
                 const RealFieldDevice &, Real, Real, Real, RealFieldDevice &)
                             const>(
                 &IsotropicStiffnessOperator3D::apply_uniform_increment),
             "displacement"_a, "lambda"_a, "mu"_a, "alpha"_a, "force"_a,
             "Uniform-scalar increment on device (GPU) fields")
        .def("apply_macro_rhs",
             static_cast<void (IsotropicStiffnessOperator3D::*)(
                 const RealFieldDevice &, const RealFieldDevice &,
                 const std::array<Real, 9> &, RealFieldDevice &) const>(
                 &IsotropicStiffnessOperator3D::apply_macro_rhs),
             "lambda_field"_a, "mu_field"_a, "E_macro"_a, "force"_a,
             "Assemble macro-strain RHS divergence on device (GPU) fields")
        .def("average_stress",
             static_cast<std::array<Real, 9> (IsotropicStiffnessOperator3D::*)(
                 const RealFieldDevice &, const RealFieldDevice &,
                 const RealFieldDevice &, const std::array<Real, 9> &) const>(
                 &IsotropicStiffnessOperator3D::average_stress),
             "displacement"_a, "lambda_field"_a, "mu_field"_a, "E_macro"_a,
             "Local volume integral of stress on device (GPU) fields")
        .def("assemble_diagonal",
             static_cast<void (IsotropicStiffnessOperator3D::*)(
                 const RealFieldDevice &, const RealFieldDevice &,
                 RealFieldDevice &) const>(
                 &IsotropicStiffnessOperator3D::assemble_diagonal),
             "lambda_field"_a, "mu_field"_a, "diagonal"_a,
             "Assemble diag(K) into a device (GPU) nodal field")
        .def("compute_sensitivity",
             static_cast<void (IsotropicStiffnessOperator3D::*)(
                 const RealFieldDevice &, const std::array<Real, 9> &,
                 const RealFieldDevice &, const std::array<Real, 9> &,
                 RealFieldDevice &, RealFieldDevice &) const>(
                 &IsotropicStiffnessOperator3D::compute_sensitivity),
             "forward_disp"_a, "forward_macro"_a, "costate_disp"_a,
             "costate_macro"_a, "g_shear"_a, "g_vol"_a,
             "Per-pixel sensitivity contractions on device (GPU) fields");
    // Single-precision (float32) device apply overloads.
    add_stiffness_real32_device_overloads<IsotropicStiffnessOperator3D, 9>(op);
#endif
}

void add_convolution_operator_classes(py::module & mod) {
    add_gradient_operator(mod);
    add_stencil_gradient_operator(mod);
    add_laplace_operator_2d(mod);
    add_laplace_operator_3d(mod);
    add_fem_gradient_operator_2d(mod);
    add_fem_gradient_operator_3d(mod);
    add_isotropic_stiffness_operator_2d(mod);
    add_isotropic_stiffness_operator_3d(mod);

    // Backwards compatibility aliases
    mod.attr("ConvolutionOperatorBase") = mod.attr("GradientOperator");
    mod.attr("ConvolutionOperator") = mod.attr("GenericLinearOperator");
}
