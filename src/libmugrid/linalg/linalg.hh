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

namespace muGrid {
namespace linalg {

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

}  // namespace internal

}  // namespace linalg
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_LINALG_LINALG_HH_
