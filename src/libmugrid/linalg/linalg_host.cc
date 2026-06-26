/**
 * @file   linalg_host.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2025
 *
 * @brief  CPU implementations of linear algebra operations for fields
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

#include "linalg/linalg.hh"
#include "collection/field_collection.hh"
#include "collection/field_collection_global.hh"
#include "core/exception.hh"

#include <complex>

namespace muGrid {
namespace linalg {

namespace internal {

// Sesquilinear product: conj(a)*b for complex, a*b for real.
template <typename T>
inline T conj_product(T a, T b) {
    if constexpr (std::is_same_v<T, Complex>) {
        return std::conj(a) * b;
    } else {
        return a * b;
    }
}

/**
 * True if the collection carries ghost buffers in any direction.
 */
inline bool has_ghosts(const GlobalFieldCollection& coll) {
    const auto& nb_ghosts_left = coll.get_nb_ghosts_left();
    const auto& nb_ghosts_right = coll.get_nb_ghosts_right();
    for (Dim_t d = 0; d < coll.get_spatial_dim(); ++d) {
        if (nb_ghosts_left[d] != 0 || nb_ghosts_right[d] != 0) {
            return true;
        }
    }
    return false;
}

/**
 * Compute the dot product over the interior (non-ghost) region by summing
 * the interior directly.
 *
 * Do NOT compute this as full-buffer product minus ghost contribution: the
 * ghost buffers hold large stale data (stencil operators write results into
 * ghost pixels), so the subtraction cancels catastrophically once the
 * interior values are small. This destroyed converged CG residuals — the
 * reported squared norm carried an absolute error of order
 * eps * ||ghosts||^2 and could even go negative.
 */
template <typename T>
T interior_vecdot(const T* a_data, const T* b_data,
                  const GlobalFieldCollection& coll,
                  Index_t nb_components_per_pixel) {
    const auto spatial_dim = coll.get_spatial_dim();
    const auto& nb_pts = coll.get_nb_subdomain_grid_pts_with_ghosts();
    const auto& nb_ghosts_left = coll.get_nb_ghosts_left();
    const auto& nb_ghosts_right = coll.get_nb_ghosts_right();
    const auto& strides = coll.get_pixels_with_ghosts().get_strides();

    if (spatial_dim < 1 || spatial_dim > 3) {
        throw FieldError("interior_vecdot only supports 1D, 2D and 3D fields");
    }

    // Interior bounds; unused trailing dimensions degenerate to one pass
    Index_t start[3]{0, 0, 0};
    Index_t end[3]{1, 1, 1};
    Index_t stride[3]{0, 0, 0};
    for (Dim_t d = 0; d < spatial_dim; ++d) {
        start[d] = nb_ghosts_left[d];
        end[d] = nb_pts[d] - nb_ghosts_right[d];
        stride[d] = strides[d];
    }

    T result = T{0};
    for (Index_t iz = start[2]; iz < end[2]; ++iz) {
        for (Index_t iy = start[1]; iy < end[1]; ++iy) {
            for (Index_t ix = start[0]; ix < end[0]; ++ix) {
                const Index_t offset =
                    (ix * stride[0] + iy * stride[1] + iz * stride[2]) *
                    nb_components_per_pixel;
                for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                    result +=
                        conj_product(a_data[offset + c], b_data[offset + c]);
                }
            }
        }
    }
    return result;
}

/**
 * Fused interior reduction: {(r,u), (w,u), (r,r)} in a single pass, reading
 * r, u and w once each. Mirrors interior_vecdot's bounds/stride handling.
 */
template <typename T>
std::array<T, 3> interior_three_dots(const T* r_data, const T* u_data,
                                     const T* w_data,
                                     const GlobalFieldCollection& coll,
                                     Index_t nb_components_per_pixel) {
    const auto spatial_dim = coll.get_spatial_dim();
    const auto& nb_pts = coll.get_nb_subdomain_grid_pts_with_ghosts();
    const auto& nb_ghosts_left = coll.get_nb_ghosts_left();
    const auto& nb_ghosts_right = coll.get_nb_ghosts_right();
    const auto& strides = coll.get_pixels_with_ghosts().get_strides();

    if (spatial_dim < 1 || spatial_dim > 3) {
        throw FieldError("interior_three_dots only supports 1D, 2D and 3D fields");
    }

    Index_t start[3]{0, 0, 0};
    Index_t end[3]{1, 1, 1};
    Index_t stride[3]{0, 0, 0};
    for (Dim_t d = 0; d < spatial_dim; ++d) {
        start[d] = nb_ghosts_left[d];
        end[d] = nb_pts[d] - nb_ghosts_right[d];
        stride[d] = strides[d];
    }

    T ru{0}, wu{0}, rr{0};
    for (Index_t iz = start[2]; iz < end[2]; ++iz) {
        for (Index_t iy = start[1]; iy < end[1]; ++iy) {
            for (Index_t ix = start[0]; ix < end[0]; ++ix) {
                const Index_t offset =
                    (ix * stride[0] + iy * stride[1] + iz * stride[2]) *
                    nb_components_per_pixel;
                for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                    const T rv = r_data[offset + c];
                    const T uv = u_data[offset + c];
                    const T wv = w_data[offset + c];
                    ru += conj_product(rv, uv);
                    wu += conj_product(wv, uv);
                    rr += conj_product(rv, rv);
                }
            }
        }
    }
    return {ru, wu, rr};
}

}  // namespace internal

/* ---------------------------------------------------------------------- */
template <>
Real vecdot<Real, HostSpace>(const TypedField<Real, HostSpace>& a,
                              const TypedField<Real, HostSpace>& b) {
    // Verify fields are compatible
    if (&a.get_collection() != &b.get_collection()) {
        throw FieldError("vecdot: fields must belong to the same collection");
    }
    if (a.get_nb_components() != b.get_nb_components()) {
        throw FieldError("vecdot: fields must have the same number of components");
    }
    if (a.get_nb_sub_pts() != b.get_nb_sub_pts()) {
        throw FieldError("vecdot: fields must have the same number of sub-points");
    }

    const auto& coll = a.get_collection();

    // Check if this is a GlobalFieldCollection (has ghost regions)
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (!internal::has_ghosts(global_coll)) {
            return a.eigen_vec().dot(b.eigen_vec());
        }
        const Index_t nb_components_per_pixel =
            a.get_nb_components() * a.get_nb_sub_pts();
        // Sum the interior directly; full-buffer-minus-ghosts cancels
        // catastrophically when the interior values are small
        return internal::interior_vecdot(a.data(), b.data(), global_coll,
                                         nb_components_per_pixel);
    } else {
        // LocalFieldCollection: no ghosts, use Eigen
        return a.eigen_vec().dot(b.eigen_vec());
    }
}

template <>
Complex vecdot<Complex, HostSpace>(const TypedField<Complex, HostSpace>& a,
                                    const TypedField<Complex, HostSpace>& b) {
    // Verify fields are compatible
    if (&a.get_collection() != &b.get_collection()) {
        throw FieldError("vecdot: fields must belong to the same collection");
    }
    if (a.get_nb_components() != b.get_nb_components()) {
        throw FieldError("vecdot: fields must have the same number of components");
    }
    if (a.get_nb_sub_pts() != b.get_nb_sub_pts()) {
        throw FieldError("vecdot: fields must have the same number of sub-points");
    }

    const auto& coll = a.get_collection();

    // Check if this is a GlobalFieldCollection (has ghost regions)
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (!internal::has_ghosts(global_coll)) {
            return a.eigen_vec().dot(b.eigen_vec());
        }
        const Index_t nb_components_per_pixel =
            a.get_nb_components() * a.get_nb_sub_pts();
        // Sum the interior directly; full-buffer-minus-ghosts cancels
        // catastrophically when the interior values are small
        return internal::interior_vecdot(a.data(), b.data(), global_coll,
                                         nb_components_per_pixel);
    } else {
        // LocalFieldCollection: no ghosts, use Eigen
        return a.eigen_vec().dot(b.eigen_vec());
    }
}

/* ---------------------------------------------------------------------- */
template <>
void axpy<Real, HostSpace>(Real alpha, const TypedField<Real, HostSpace>& x,
                            TypedField<Real, HostSpace>& y) {
    // Verify fields are compatible
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpy: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpy: fields must have the same number of entries");
    }

    // Operate on full buffer using Eigen (contiguous, fast)
    y.eigen_vec() += alpha * x.eigen_vec();
}

template <>
void axpy<Complex, HostSpace>(Complex alpha, const TypedField<Complex, HostSpace>& x,
                               TypedField<Complex, HostSpace>& y) {
    // Verify fields are compatible
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpy: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpy: fields must have the same number of entries");
    }

    // Operate on full buffer using Eigen (contiguous, fast)
    y.eigen_vec() += alpha * x.eigen_vec();
}

/* ---------------------------------------------------------------------- */
template <>
void scal<Real, HostSpace>(Real alpha, TypedField<Real, HostSpace>& x) {
    // Operate on full buffer using Eigen
    x.eigen_vec() *= alpha;
}

template <>
void scal<Complex, HostSpace>(Complex alpha, TypedField<Complex, HostSpace>& x) {
    // Operate on full buffer using Eigen
    x.eigen_vec() *= alpha;
}

/* ---------------------------------------------------------------------- */
template <>
void axpby<Real, HostSpace>(Real alpha, const TypedField<Real, HostSpace>& x,
                             Real beta, TypedField<Real, HostSpace>& y) {
    // Verify fields are compatible
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpby: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpby: fields must have the same number of entries");
    }

    // Operate on full buffer using Eigen (contiguous, fast)
    // y = alpha * x + beta * y
    y.eigen_vec() = alpha * x.eigen_vec() + beta * y.eigen_vec();
}

template <>
void axpby<Complex, HostSpace>(Complex alpha, const TypedField<Complex, HostSpace>& x,
                                Complex beta, TypedField<Complex, HostSpace>& y) {
    // Verify fields are compatible
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpby: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpby: fields must have the same number of entries");
    }

    // Operate on full buffer using Eigen (contiguous, fast)
    // y = alpha * x + beta * y
    y.eigen_vec() = alpha * x.eigen_vec() + beta * y.eigen_vec();
}

/* ---------------------------------------------------------------------- */
template <>
void copy<Real, HostSpace>(const TypedField<Real, HostSpace>& src,
                            TypedField<Real, HostSpace>& dst) {
    // Verify fields are compatible
    if (&src.get_collection() != &dst.get_collection()) {
        throw FieldError("copy: fields must belong to the same collection");
    }
    if (src.get_nb_entries() != dst.get_nb_entries() ||
        src.get_nb_components() != dst.get_nb_components()) {
        throw FieldError("copy: fields must have the same number of entries");
    }

    // Operate on full buffer using Eigen
    dst.eigen_vec() = src.eigen_vec();
}

template <>
void copy<Complex, HostSpace>(const TypedField<Complex, HostSpace>& src,
                               TypedField<Complex, HostSpace>& dst) {
    // Verify fields are compatible
    if (&src.get_collection() != &dst.get_collection()) {
        throw FieldError("copy: fields must belong to the same collection");
    }
    if (src.get_nb_entries() != dst.get_nb_entries() ||
        src.get_nb_components() != dst.get_nb_components()) {
        throw FieldError("copy: fields must have the same number of entries");
    }

    // Operate on full buffer using Eigen
    dst.eigen_vec() = src.eigen_vec();
}

/* ---------------------------------------------------------------------- */
template <>
Real norm_sq<Real, HostSpace>(const TypedField<Real, HostSpace>& x) {
    const auto& coll = x.get_collection();

    // Check if this is a GlobalFieldCollection (has ghost regions)
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (!internal::has_ghosts(global_coll)) {
            return x.eigen_vec().squaredNorm();
        }
        const Index_t nb_components_per_pixel =
            x.get_nb_components() * x.get_nb_sub_pts();
        // Sum the interior directly; full-buffer-minus-ghosts cancels
        // catastrophically when the interior values are small
        return internal::interior_vecdot(x.data(), x.data(), global_coll,
                                         nb_components_per_pixel);
    } else {
        // LocalFieldCollection: no ghosts, use Eigen
        return x.eigen_vec().squaredNorm();
    }
}

template <>
std::array<Real, 3> pipelined_cg_dots<Real, HostSpace>(
    const TypedField<Real, HostSpace>& r, const TypedField<Real, HostSpace>& u,
    const TypedField<Real, HostSpace>& w) {
    const auto& coll = r.get_collection();
    const Index_t nb_components_per_pixel =
        r.get_nb_components() * r.get_nb_sub_pts();
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        return internal::interior_three_dots(r.data(), u.data(), w.data(),
                                             global_coll,
                                             nb_components_per_pixel);
    }
    // LocalFieldCollection: no ghosts, use the per-pair Eigen reductions
    return {r.eigen_vec().dot(u.eigen_vec()), w.eigen_vec().dot(u.eigen_vec()),
            r.eigen_vec().squaredNorm()};
}

template <>
Complex norm_sq<Complex, HostSpace>(const TypedField<Complex, HostSpace>& x) {
    const auto& coll = x.get_collection();

    // Check if this is a GlobalFieldCollection (has ghost regions)
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (!internal::has_ghosts(global_coll)) {
            return x.eigen_vec().squaredNorm();
        }
        const Index_t nb_components_per_pixel =
            x.get_nb_components() * x.get_nb_sub_pts();
        // Sum the interior directly; full-buffer-minus-ghosts cancels
        // catastrophically when the interior values are small
        return internal::interior_vecdot(x.data(), x.data(), global_coll,
                                         nb_components_per_pixel);
    } else {
        // LocalFieldCollection: no ghosts, use Eigen
        return x.eigen_vec().squaredNorm();
    }
}

/* ---------------------------------------------------------------------- */
template <>
Real axpy_norm_sq<Real, HostSpace>(Real alpha,
                                    const TypedField<Real, HostSpace>& x,
                                    TypedField<Real, HostSpace>& y) {
    // Verify fields are compatible
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpy_norm_sq: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpy_norm_sq: fields must have the same number of entries");
    }

    // Use Eigen's optimized vectorized operations (two passes, but fast)
    // A true single-pass fusion would require breaking Eigen's vectorization
    y.eigen_vec() += alpha * x.eigen_vec();

    const auto& coll = x.get_collection();

    // For GlobalFieldCollection, sum the interior directly; computing the
    // full-buffer norm and subtracting the ghost contribution cancels
    // catastrophically when the interior values are small
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (!internal::has_ghosts(global_coll)) {
            return y.eigen_vec().squaredNorm();
        }
        const Index_t nb_components_per_pixel =
            x.get_nb_components() * x.get_nb_sub_pts();
        return internal::interior_vecdot(y.data(), y.data(), global_coll,
                                         nb_components_per_pixel);
    }

    return y.eigen_vec().squaredNorm();
}

template <>
Complex axpy_norm_sq<Complex, HostSpace>(Complex alpha,
                                          const TypedField<Complex, HostSpace>& x,
                                          TypedField<Complex, HostSpace>& y) {
    // Verify fields are compatible
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpy_norm_sq: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpy_norm_sq: fields must have the same number of entries");
    }

    // Use Eigen's optimized vectorized operations (two passes, but fast)
    // A true single-pass fusion would require breaking Eigen's vectorization
    y.eigen_vec() += alpha * x.eigen_vec();

    const auto& coll = x.get_collection();

    // For GlobalFieldCollection, sum the interior directly; computing the
    // full-buffer norm and subtracting the ghost contribution cancels
    // catastrophically when the interior values are small
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (!internal::has_ghosts(global_coll)) {
            return y.eigen_vec().squaredNorm();
        }
        const Index_t nb_components_per_pixel =
            x.get_nb_components() * x.get_nb_sub_pts();
        return internal::interior_vecdot(y.data(), y.data(), global_coll,
                                         nb_components_per_pixel);
    }

    return y.eigen_vec().squaredNorm();
}

/* ---------------------------------------------------------------------- */
template <>
void scal<HostSpace>(const TypedField<Real, HostSpace>& alpha,
                     TypedField<Complex, HostSpace>& x) {
    internal::check_field_alpha(alpha, x);

    const Index_t npix = x.get_nb_entries();
    const Index_t ncomp = x.get_nb_components();
    Complex* xd = x.view().data();
    const Real* ad = alpha.view().data();

    if (alpha.get_nb_components() == ncomp) {
        // Elementwise: alpha and x share the same buffer layout
        for (Index_t i{0}; i < npix * ncomp; ++i) {
            xd[i] *= ad[i];
        }
    } else if (x.get_storage_order() == StorageOrder::StructureOfArrays) {
        for (Index_t c{0}; c < ncomp; ++c) {
            Complex* xc = xd + c * npix;
            for (Index_t i{0}; i < npix; ++i) {
                xc[i] *= ad[i];
            }
        }
    } else {
        for (Index_t i{0}; i < npix; ++i) {
            for (Index_t c{0}; c < ncomp; ++c) {
                xd[i * ncomp + c] *= ad[i];
            }
        }
    }
}

/* ---------------------------------------------------------------------- */
template <>
void scal<HostSpace>(const TypedField<Real, HostSpace>& alpha,
                     TypedField<Real, HostSpace>& x) {
    internal::check_field_alpha(alpha, x);

    const Index_t npix = x.get_nb_entries();
    const Index_t ncomp = x.get_nb_components();
    Real* xd = x.view().data();
    const Real* ad = alpha.view().data();

    if (alpha.get_nb_components() == ncomp) {
        // Elementwise: alpha and x share the same buffer layout
        for (Index_t i{0}; i < npix * ncomp; ++i) {
            xd[i] *= ad[i];
        }
    } else if (x.get_storage_order() == StorageOrder::StructureOfArrays) {
        for (Index_t c{0}; c < ncomp; ++c) {
            Real* xc = xd + c * npix;
            for (Index_t i{0}; i < npix; ++i) {
                xc[i] *= ad[i];
            }
        }
    } else {
        for (Index_t i{0}; i < npix; ++i) {
            for (Index_t c{0}; c < ncomp; ++c) {
                xd[i * ncomp + c] *= ad[i];
            }
        }
    }
}

/* ---------------------------------------------------------------------- */
/* Fused per-pixel vector kernels                                          */
/* ---------------------------------------------------------------------- */

namespace internal {

// out = a x b for the npix three-vectors held by the buffers. `soa` selects
// the storage order: SoA keeps each component in a contiguous block of npix
// values, AoS interleaves the three components of a pixel.
template <typename T>
void cross_buffers(const T* a, const T* b, T* out, Index_t npix, bool soa) {
    if (soa) {
        const T* a0 = a; const T* a1 = a + npix; const T* a2 = a + 2 * npix;
        const T* b0 = b; const T* b1 = b + npix; const T* b2 = b + 2 * npix;
        T* o0 = out; T* o1 = out + npix; T* o2 = out + 2 * npix;
        for (Index_t i{0}; i < npix; ++i) {
            o0[i] = a1[i] * b2[i] - a2[i] * b1[i];
            o1[i] = a2[i] * b0[i] - a0[i] * b2[i];
            o2[i] = a0[i] * b1[i] - a1[i] * b0[i];
        }
    } else {
        for (Index_t i{0}; i < npix; ++i) {
            const T* A = a + 3 * i; const T* B = b + 3 * i; T* O = out + 3 * i;
            O[0] = A[1] * B[2] - A[2] * B[1];
            O[1] = A[2] * B[0] - A[0] * B[2];
            O[2] = A[0] * B[1] - A[1] * B[0];
        }
    }
}

// out[c] -= k[c] * sum_d(invk[d] * N[d]) for the npix three-vectors.
void leray_buffers(const Real* k, const Real* invk, const Complex* N,
                   Complex* out, Index_t npix, bool soa) {
    if (soa) {
        const Real* k0 = k; const Real* k1 = k + npix; const Real* k2 = k + 2 * npix;
        const Real* i0 = invk; const Real* i1 = invk + npix; const Real* i2 = invk + 2 * npix;
        const Complex* n0 = N; const Complex* n1 = N + npix; const Complex* n2 = N + 2 * npix;
        Complex* o0 = out; Complex* o1 = out + npix; Complex* o2 = out + 2 * npix;
        for (Index_t i{0}; i < npix; ++i) {
            const Complex s = i0[i] * n0[i] + i1[i] * n1[i] + i2[i] * n2[i];
            o0[i] -= k0[i] * s;
            o1[i] -= k1[i] * s;
            o2[i] -= k2[i] * s;
        }
    } else {
        for (Index_t i{0}; i < npix; ++i) {
            const Real* K = k + 3 * i; const Real* IK = invk + 3 * i;
            const Complex* NN = N + 3 * i; Complex* O = out + 3 * i;
            const Complex s = IK[0] * NN[0] + IK[1] * NN[1] + IK[2] * NN[2];
            O[0] -= K[0] * s;
            O[1] -= K[1] * s;
            O[2] -= K[2] * s;
        }
    }
}

}  // namespace internal

/* ---------------------------------------------------------------------- */
template <typename T>
static void cross_host(const TypedField<T, HostSpace>& a,
                       const TypedField<T, HostSpace>& b,
                       TypedField<T, HostSpace>& out) {
    const auto& coll = a.get_collection();
    internal::check_three_vector("cross", a, coll);
    internal::check_three_vector("cross", b, coll);
    internal::check_three_vector("cross", out, coll);
    const Index_t npix = a.get_nb_entries();
    // An empty subdomain (e.g. an MPI rank with no local pixels) has nothing to
    // compute. Skip it, including the aliasing check below: empty fields share a
    // null data pointer, so that check would otherwise fire spuriously.
    if (npix == 0) {
        return;
    }
    if (out.view().data() == a.view().data() ||
        out.view().data() == b.view().data()) {
        throw FieldError(
            "cross: output must be a field distinct from both inputs");
    }
    const bool soa =
        (out.get_storage_order() == StorageOrder::StructureOfArrays);
    internal::cross_buffers<T>(a.view().data(), b.view().data(),
                               out.view().data(), npix, soa);
}

template <>
void cross<Real, HostSpace>(const TypedField<Real, HostSpace>& a,
                            const TypedField<Real, HostSpace>& b,
                            TypedField<Real, HostSpace>& out) {
    cross_host(a, b, out);
}

template <>
void cross<Complex, HostSpace>(const TypedField<Complex, HostSpace>& a,
                               const TypedField<Complex, HostSpace>& b,
                               TypedField<Complex, HostSpace>& out) {
    cross_host(a, b, out);
}

/* ---------------------------------------------------------------------- */
template <>
void leray_project<HostSpace>(const TypedField<Real, HostSpace>& k,
                              const TypedField<Real, HostSpace>& invk,
                              const TypedField<Complex, HostSpace>& N,
                              TypedField<Complex, HostSpace>& out) {
    const auto& coll = out.get_collection();
    internal::check_three_vector("leray_project", k, coll);
    internal::check_three_vector("leray_project", invk, coll);
    internal::check_three_vector("leray_project", N, coll);
    internal::check_three_vector("leray_project", out, coll);
    // k, invk, N and out share a collection, hence a storage order.
    const Index_t npix = out.get_nb_entries();
    const bool soa =
        (out.get_storage_order() == StorageOrder::StructureOfArrays);
    internal::leray_buffers(k.view().data(), invk.view().data(),
                            N.view().data(), out.view().data(), npix, soa);
}

}  // namespace linalg
}  // namespace muGrid
