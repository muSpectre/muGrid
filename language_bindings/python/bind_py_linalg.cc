/**
 * @file   bind_py_linalg.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2024
 *
 * @brief  Python bindings for linear algebra operations on fields
 *
 * Copyright © 2024 Lars Pastewka
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

#include "core/types.hh"
#include "field/field_typed.hh"
#include "linalg/linalg.hh"

#include <pybind11/pybind11.h>

using muGrid::Real;
using muGrid::Complex;
using muGrid::TypedField;
using muGrid::HostSpace;
using pybind11::literals::operator""_a;

namespace py = pybind11;

// Type aliases for host fields
using RealFieldHost = TypedField<Real, HostSpace>;
using ComplexFieldHost = TypedField<Complex, HostSpace>;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
using DeviceSpace = muGrid::DefaultDeviceSpace;
using RealFieldDevice = TypedField<Real, DeviceSpace>;
using ComplexFieldDevice = TypedField<Complex, DeviceSpace>;
#endif

void add_linalg_functions(py::module &mod) {
    // Create linalg submodule
    auto linalg = mod.def_submodule("linalg",
        R"pbdoc(
        Linear algebra operations for muGrid fields.

        This module provides efficient linear algebra operations that operate
        directly on muGrid fields, avoiding the overhead of creating non-contiguous
        views. Operations follow the Array API specification where applicable:
        https://data-apis.org/array-api/latest/

        For fields with ghost regions (GlobalFieldCollection), reduction operations
        (vecdot, norm_sq) only iterate over the interior region to ensure correct
        MPI-parallel semantics. Update operations (axpy, scal, copy) operate on
        the full buffer for efficiency.

        Note: These functions return local (process-local) results for parallel
        computations. Use comm.sum() to reduce across MPI ranks.
        )pbdoc");

    // --- Real field operations (host) ---

    linalg.def("vecdot",
        static_cast<Real (*)(const RealFieldHost&, const RealFieldHost&)>(
            &muGrid::linalg::vecdot<Real, HostSpace>),
        "a"_a, "b"_a,
        R"pbdoc(
        Compute vector dot product of two real fields (interior only).

        Computes sum_i(a[i] * b[i]) over all interior pixels and components.
        Ghost regions are excluded for correct MPI-parallel semantics.

        Parameters
        ----------
        a : RealField
            First field (host memory)
        b : RealField
            Second field (must have same shape as a)

        Returns
        -------
        float
            Local dot product (not MPI-reduced)

        Notes
        -----
        Following Array API vecdot semantics:
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.vecdot.html
        )pbdoc");

    linalg.def("norm_sq",
        static_cast<Real (*)(const RealFieldHost&)>(
            &muGrid::linalg::norm_sq<Real, HostSpace>),
        "x"_a,
        R"pbdoc(
        Compute squared L2 norm of a real field (interior only).

        Equivalent to vecdot(x, x).

        Parameters
        ----------
        x : RealField
            Input field (host memory)

        Returns
        -------
        float
            Local squared norm (not MPI-reduced)
        )pbdoc");

    linalg.def("axpy",
        static_cast<void (*)(Real, const RealFieldHost&, RealFieldHost&)>(
            &muGrid::linalg::axpy<Real, HostSpace>),
        "alpha"_a, "x"_a, "y"_a,
        R"pbdoc(
        AXPY operation: y = alpha * x + y (full buffer).

        Parameters
        ----------
        alpha : float
            Scalar multiplier
        x : RealField
            Input field (host memory)
        y : RealField
            Input/output field (modified in place)
        )pbdoc");

    linalg.def("scal",
        static_cast<void (*)(Real, RealFieldHost&)>(
            &muGrid::linalg::scal<Real, HostSpace>),
        "alpha"_a, "x"_a,
        R"pbdoc(
        Scale operation: x = alpha * x (full buffer).

        Parameters
        ----------
        alpha : float
            Scalar multiplier
        x : RealField
            Input/output field (modified in place)
        )pbdoc");

    linalg.def("copy",
        static_cast<void (*)(const RealFieldHost&, RealFieldHost&)>(
            &muGrid::linalg::copy<Real, HostSpace>),
        "src"_a, "dst"_a,
        R"pbdoc(
        Copy operation: dst = src (full buffer).

        Parameters
        ----------
        src : RealField
            Source field (host memory)
        dst : RealField
            Destination field (modified in place)
        )pbdoc");

    // --- Complex field operations (host) ---

    linalg.def("vecdot",
        static_cast<Complex (*)(const ComplexFieldHost&, const ComplexFieldHost&)>(
            &muGrid::linalg::vecdot<Complex, HostSpace>),
        "a"_a, "b"_a,
        "Compute vector dot product of two complex fields (interior only).");

    linalg.def("norm_sq",
        static_cast<Complex (*)(const ComplexFieldHost&)>(
            &muGrid::linalg::norm_sq<Complex, HostSpace>),
        "x"_a,
        "Compute squared L2 norm of a complex field (interior only).");

    linalg.def("axpy",
        static_cast<void (*)(Complex, const ComplexFieldHost&, ComplexFieldHost&)>(
            &muGrid::linalg::axpy<Complex, HostSpace>),
        "alpha"_a, "x"_a, "y"_a,
        "AXPY operation for complex fields: y = alpha * x + y.");

    linalg.def("scal",
        static_cast<void (*)(Complex, ComplexFieldHost&)>(
            &muGrid::linalg::scal<Complex, HostSpace>),
        "alpha"_a, "x"_a,
        "Scale operation for complex fields: x = alpha * x.");

    linalg.def("copy",
        static_cast<void (*)(const ComplexFieldHost&, ComplexFieldHost&)>(
            &muGrid::linalg::copy<Complex, HostSpace>),
        "src"_a, "dst"_a,
        "Copy operation for complex fields: dst = src.");

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    // --- Real field operations (device) ---

    linalg.def("vecdot",
        static_cast<Real (*)(const RealFieldDevice&, const RealFieldDevice&)>(
            &muGrid::linalg::vecdot<Real, DeviceSpace>),
        "a"_a, "b"_a,
        "Compute vector dot product of two real fields on device (GPU).");

    linalg.def("norm_sq",
        static_cast<Real (*)(const RealFieldDevice&)>(
            &muGrid::linalg::norm_sq<Real, DeviceSpace>),
        "x"_a,
        "Compute squared L2 norm of a real field on device (GPU).");

    linalg.def("axpy",
        static_cast<void (*)(Real, const RealFieldDevice&, RealFieldDevice&)>(
            &muGrid::linalg::axpy<Real, DeviceSpace>),
        "alpha"_a, "x"_a, "y"_a,
        "AXPY operation on device (GPU): y = alpha * x + y.");

    linalg.def("scal",
        static_cast<void (*)(Real, RealFieldDevice&)>(
            &muGrid::linalg::scal<Real, DeviceSpace>),
        "alpha"_a, "x"_a,
        "Scale operation on device (GPU): x = alpha * x.");

    linalg.def("copy",
        static_cast<void (*)(const RealFieldDevice&, RealFieldDevice&)>(
            &muGrid::linalg::copy<Real, DeviceSpace>),
        "src"_a, "dst"_a,
        "Copy operation on device (GPU): dst = src.");

    // --- Complex field operations (device) ---

    linalg.def("vecdot",
        static_cast<Complex (*)(const ComplexFieldDevice&, const ComplexFieldDevice&)>(
            &muGrid::linalg::vecdot<Complex, DeviceSpace>),
        "a"_a, "b"_a,
        "Compute vector dot product of two complex fields on device (GPU).");

    linalg.def("norm_sq",
        static_cast<Complex (*)(const ComplexFieldDevice&)>(
            &muGrid::linalg::norm_sq<Complex, DeviceSpace>),
        "x"_a,
        "Compute squared L2 norm of a complex field on device (GPU).");

    linalg.def("axpy",
        static_cast<void (*)(Complex, const ComplexFieldDevice&, ComplexFieldDevice&)>(
            &muGrid::linalg::axpy<Complex, DeviceSpace>),
        "alpha"_a, "x"_a, "y"_a,
        "AXPY operation for complex fields on device (GPU): y = alpha * x + y.");

    linalg.def("scal",
        static_cast<void (*)(Complex, ComplexFieldDevice&)>(
            &muGrid::linalg::scal<Complex, DeviceSpace>),
        "alpha"_a, "x"_a,
        "Scale operation for complex fields on device (GPU): x = alpha * x.");

    linalg.def("copy",
        static_cast<void (*)(const ComplexFieldDevice&, ComplexFieldDevice&)>(
            &muGrid::linalg::copy<Complex, DeviceSpace>),
        "src"_a, "dst"_a,
        "Copy operation for complex fields on device (GPU): dst = src.");
#endif
}
