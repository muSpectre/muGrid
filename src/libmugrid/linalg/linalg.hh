/**
 * @file   linalg.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2024
 *
 * @brief  Linear algebra operations for muGrid fields
 *
 * This module provides efficient linear algebra operations that operate
 * directly on muGrid fields, avoiding the overhead of creating non-contiguous
 * views. Operations follow the Array API specification where applicable:
 * https://data-apis.org/array-api/latest/
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

#ifndef SRC_LIBMUGRID_LINALG_LINALG_HH_
#define SRC_LIBMUGRID_LINALG_LINALG_HH_

#include "field/field_typed.hh"

#include <array>
#include <string>

namespace muGrid {
namespace linalg {

/**
 * Fused interior reduction for pipelined conjugate gradients.
 *
 * In a single pass over the interior region (reading r, u and w once each),
 * returns the three inner products needed per pipelined-CG iteration:
 *   {0} = (r, u)   {1} = (w, u)   {2} = (r, r)
 * as local (not MPI-reduced) values. On the GPU this is one kernel launch and
 * one device->host copy, replacing the three separate reductions (and three
 * blocking copies) that standard preconditioned CG performs per iteration.
 *
 * Ghost regions are excluded, matching vecdot()/norm_sq().
 *
 * @tparam T Scalar type (Real)
 * @tparam MemorySpace Memory space (HostSpace, CUDASpace, ROCmSpace)
 */
template <typename T, typename MemorySpace>
std::array<T, 3> pipelined_cg_dots(const TypedField<T, MemorySpace>& r,
                                   const TypedField<T, MemorySpace>& u,
                                   const TypedField<T, MemorySpace>& w);

/**
 * Vector dot product on interior pixels only (excludes ghost regions).
 * Computes sum_i(a[i] * b[i]) for all interior pixels and all components.
 *
 * This function iterates only over the interior region of the field,
 * excluding ghost cells. This is essential for MPI-parallel computations
 * where ghost values are duplicated across processes.
 *
 * Following Array API vecdot semantics:
 * https://data-apis.org/array-api/latest/API_specification/generated/array_api.vecdot.html
 *
 * @tparam T Scalar type (Real, Complex, etc.)
 * @tparam MemorySpace Memory space (HostSpace, CUDASpace, ROCmSpace)
 * @param a First field
 * @param b Second field (must have same shape as a)
 * @return Scalar dot product (local, not MPI-reduced)
 * @throws FieldError if fields have incompatible shapes or collections
 */
template <typename T, typename MemorySpace>
T vecdot(const TypedField<T, MemorySpace>& a,
         const TypedField<T, MemorySpace>& b);

/**
 * AXPY operation: y = alpha * x + y
 *
 * Operates on the FULL buffer (including ghost regions) for efficiency,
 * as the underlying memory is contiguous. Ghost values will typically
 * be overwritten by subsequent ghost communication anyway.
 *
 * @tparam T Scalar type (Real, Complex, etc.)
 * @tparam MemorySpace Memory space (HostSpace, CUDASpace, ROCmSpace)
 * @param alpha Scalar multiplier
 * @param x Input field
 * @param y Input/output field (modified in place)
 * @throws FieldError if fields have incompatible shapes
 */
template <typename T, typename MemorySpace>
void axpy(T alpha,
          const TypedField<T, MemorySpace>& x,
          TypedField<T, MemorySpace>& y);

/**
 * Scale operation: x = alpha * x
 *
 * Operates on the FULL buffer for efficiency.
 *
 * @tparam T Scalar type (Real, Complex, etc.)
 * @tparam MemorySpace Memory space (HostSpace, CUDASpace, ROCmSpace)
 * @param alpha Scalar multiplier
 * @param x Input/output field (modified in place)
 */
template <typename T, typename MemorySpace>
void scal(T alpha, TypedField<T, MemorySpace>& x);

/**
 * AXPBY operation: y = alpha * x + beta * y
 *
 * Combined scale-and-add that is more efficient than separate scal + axpy
 * because it reads and writes each element only once.
 *
 * Operates on the FULL buffer (including ghost regions) for efficiency.
 *
 * @tparam T Scalar type (Real, Complex, etc.)
 * @tparam MemorySpace Memory space (HostSpace, CUDASpace, ROCmSpace)
 * @param alpha Scalar multiplier for x
 * @param x Input field
 * @param beta Scalar multiplier for y
 * @param y Input/output field (modified in place)
 * @throws FieldError if fields have incompatible shapes
 */
template <typename T, typename MemorySpace>
void axpby(T alpha,
           const TypedField<T, MemorySpace>& x,
           T beta,
           TypedField<T, MemorySpace>& y);

/**
 * Copy operation: dst = src
 *
 * Operates on the FULL buffer.
 *
 * @tparam T Scalar type (Real, Complex, etc.)
 * @tparam MemorySpace Memory space (HostSpace, CUDASpace, ROCmSpace)
 * @param src Source field
 * @param dst Destination field (modified in place)
 * @throws FieldError if fields have incompatible shapes
 */
template <typename T, typename MemorySpace>
void copy(const TypedField<T, MemorySpace>& src,
          TypedField<T, MemorySpace>& dst);

/**
 * Squared L2 norm on interior pixels: sum_i(x[i]^2)
 *
 * Convenience function, equivalent to vecdot(x, x).
 * Only iterates over interior region (excludes ghosts).
 *
 * @tparam T Scalar type (Real, Complex, etc.)
 * @tparam MemorySpace Memory space (HostSpace, CUDASpace, ROCmSpace)
 * @param x Input field
 * @return Squared norm (local, not MPI-reduced)
 */
template <typename T, typename MemorySpace>
T norm_sq(const TypedField<T, MemorySpace>& x);

/**
 * Fused AXPY + norm_sq: y = alpha * x + y, returns ||y||² (interior only)
 *
 * This fused operation computes both the AXPY update and the squared norm
 * of the result in a single pass through memory. This is more efficient
 * than separate axpy() + norm_sq() calls because:
 * - axpy + norm_sq: 2 reads of x, 2 reads of y, 1 write of y
 * - axpy_norm_sq:   1 read of x, 1 read of y, 1 write of y
 *
 * The AXPY operates on the FULL buffer, while the norm is computed only
 * over interior pixels (excludes ghost regions for MPI correctness).
 *
 * @tparam T Scalar type (Real, Complex, etc.)
 * @tparam MemorySpace Memory space (HostSpace, CUDASpace, ROCmSpace)
 * @param alpha Scalar multiplier
 * @param x Input field
 * @param y Input/output field (modified in place)
 * @return Squared norm of y after update (local, not MPI-reduced)
 * @throws FieldError if fields have incompatible shapes
 */
template <typename T, typename MemorySpace>
T axpy_norm_sq(T alpha,
               const TypedField<T, MemorySpace>& x,
               TypedField<T, MemorySpace>& y);

/**
 * Scale operation with per-pixel multiplier: x[c, i] *= alpha[c, i].
 *
 * Overload of scal() where alpha generalizes from a scalar to a real
 * field on the same collection. A single-component alpha is broadcast
 * over the components of x (e.g. the inverse symbol of an operator in a
 * Fourier-space preconditioner); an alpha with the same number of
 * components as x is applied elementwise (e.g. a per-component Jacobi
 * diagonal). Operates on the FULL buffer; entries of alpha in ghost
 * pixels scale the corresponding ghost entries of x.
 *
 * @tparam MemorySpace Memory space (HostSpace, CUDASpace, ROCmSpace)
 * @param alpha Real field of multipliers (1 or x's number of components)
 * @param x Complex input/output field (modified in place)
 */
template <typename MemorySpace>
void scal(const TypedField<Real, MemorySpace>& alpha,
          TypedField<Complex, MemorySpace>& x);

/**
 * Scale operation with per-pixel multiplier: x[c, i] *= alpha[c, i].
 *
 * Real-field variant of the overload above, with the same broadcast
 * rules; together with copy() this runs a Jacobi preconditioner
 * entirely on the device.
 *
 * @tparam MemorySpace Memory space (HostSpace, CUDASpace, ROCmSpace)
 * @param alpha Real field of multipliers (1 or x's number of components)
 * @param x Real input/output field (modified in place)
 */
template <typename MemorySpace>
void scal(const TypedField<Real, MemorySpace>& alpha,
          TypedField<Real, MemorySpace>& x);

/**
 * Per-pixel three-vector cross product: out = a × b.
 *
 * Fused, single-pass kernel for three-component fields (e.g. the vorticity
 * `ik × û` and the Lamb vector `u × ω` of a pseudo-spectral solver). Computing
 * the cross product this way avoids the temporaries and the several array
 * passes that an `a[1]*b[2] - a[2]*b[1]`-style expression on field views would
 * allocate. Operates on the FULL buffer (ghosts included), matching the other
 * update operations. Honours both SoA and AoS storage orders.
 *
 * `out` must be a distinct field from `a` and `b`: a three-vector cross
 * product cannot be computed safely in place (the first written component
 * would be read back while forming the others).
 *
 * @tparam T Scalar type (Real or Complex)
 * @tparam MemorySpace Memory space (HostSpace, CUDASpace, ROCmSpace)
 * @param a First field (exactly 3 components)
 * @param b Second field (3 components, same collection as a)
 * @param out Output field (3 components, modified in place)
 * @throws FieldError on shape mismatch, wrong component count, or aliasing
 */
template <typename T, typename MemorySpace>
void cross(const TypedField<T, MemorySpace>& a,
           const TypedField<T, MemorySpace>& b,
           TypedField<T, MemorySpace>& out);

/**
 * Fused Leray (Helmholtz) projection update: out[c] -= k[c] * Σ_d invk[d] N[d].
 *
 * Removes the longitudinal (compressible) part of a Fourier-space vector field
 * in a single pass: with `k` the wavevector and `invk = k/|k|²`, subtracting
 * `k (k·N)/|k|²` projects `out` onto the divergence-free subspace. `k` and
 * `invk` are real per-pixel coefficient fields (three components); `N` and
 * `out` are the complex vector fields. Because the coefficients are real, the
 * real and imaginary parts are updated independently, so this also runs on the
 * device by operating on the underlying reals (cf. the field-valued `scal`).
 *
 * Computing the contraction and the rank-1 update together avoids the
 * intermediate `(invk·N)` scalar field and the broadcast multiply that the
 * array form allocates. Operates on the FULL buffer; `out` may alias `N`.
 *
 * @tparam MemorySpace Memory space (HostSpace, CUDASpace, ROCmSpace)
 * @param k Real wavevector field (3 components)
 * @param invk Real field k/|k|² (3 components, k=0 mode regularised by caller)
 * @param N Complex source vector field (3 components)
 * @param out Complex field updated in place (3 components)
 * @throws FieldError on shape mismatch or wrong component count
 */
template <typename MemorySpace>
void leray_project(const TypedField<Real, MemorySpace>& k,
                   const TypedField<Real, MemorySpace>& invk,
                   const TypedField<Complex, MemorySpace>& N,
                   TypedField<Complex, MemorySpace>& out);

namespace internal {

    //! Validate a field-valued alpha for scal(): same collection, same
    //! number of entries, and one or x's number of components
    template <typename AlphaField, typename XField>
    void check_field_alpha(const AlphaField& alpha, const XField& x) {
        if (&x.get_collection() != &alpha.get_collection()) {
            throw FieldError(
                "scal: fields must belong to the same collection");
        }
        if (alpha.get_nb_entries() != x.get_nb_entries()) {
            throw FieldError(
                "scal: fields must have the same number of entries");
        }
        if (alpha.get_nb_components() != 1 &&
            alpha.get_nb_components() != x.get_nb_components()) {
            throw FieldError(
                "scal: the field-valued alpha must have a single component "
                "(broadcast) or the same number of components as x");
        }
    }

    //! Validate that a field lives on `coll` and carries exactly three
    //! components per pixel; used by cross() / leray_project().
    template <typename AnyField>
    void check_three_vector(const char* op, const AnyField& f,
                            const FieldCollection& coll) {
        if (&f.get_collection() != &coll) {
            throw FieldError(std::string(op) +
                             ": fields must belong to the same collection");
        }
        if (f.get_nb_components() != 3) {
            throw FieldError(std::string(op) +
                             ": fields must have exactly three components");
        }
    }

}  // namespace internal

}  // namespace linalg
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_LINALG_LINALG_HH_
