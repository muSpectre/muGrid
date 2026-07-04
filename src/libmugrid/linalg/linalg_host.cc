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
#include <sstream>

namespace muGrid {
namespace linalg {

namespace internal {

// Sesquilinear product: conj(a)*b for complex, a*b for real.
template <typename T>
inline T conj_product(T a, T b) {
    if constexpr (std::is_same_v<T, Complex> ||
                  std::is_same_v<T, Complex32>) {
        return std::conj(a) * b;
    } else {
        return a * b;
    }
}

/**
 * Accumulator type for reductions: single-precision sums are promoted to
 * double precision (Real32 -> Real, Complex32 -> Complex) so a long running
 * sum does not lose its small-magnitude tail. For CG this keeps rr/rz/pAp
 * accurate enough that a float32 solve converges in the same number of
 * iterations as a float64 one. The final result is narrowed back to T on
 * return; that last rounding is a single O(eps_f32) relative error and
 * harmless — it is the *running* float32 accumulation over millions of
 * entries that loses ~1e-4 relative accuracy.
 */
template <typename T>
struct promoted {
    using type = T;
};
template <>
struct promoted<Real32> {
    using type = Real;
};
template <>
struct promoted<Complex32> {
    using type = Complex;
};
template <typename T>
using promoted_t = typename promoted<T>::type;

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
/**
 * The interior traversal below addresses dofs as
 * pixel_offset * nb_components_per_pixel + c, which is only valid for
 * array-of-structures storage. Scalar fields are layout-independent.
 */
inline void check_interior_reduction_layout(const GlobalFieldCollection& coll,
                                            Index_t nb_components_per_pixel,
                                            const char* name) {
    if (nb_components_per_pixel > 1 &&
        coll.get_storage_order() != StorageOrder::ArrayOfStructures) {
        std::stringstream error{};
        error << name
              << ": interior (ghost-skipping) reductions over fields with "
                 "more than one degree of freedom per pixel require "
                 "array-of-structures storage, but the field collection uses "
              << coll.get_storage_order() << " storage order";
        throw FieldError(error.str());
    }
}

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
    check_interior_reduction_layout(coll, nb_components_per_pixel,
                                    "interior_vecdot");

    // Interior bounds; unused trailing dimensions degenerate to one pass
    Index_t start[3]{0, 0, 0};
    Index_t end[3]{1, 1, 1};
    Index_t stride[3]{0, 0, 0};
    for (Dim_t d = 0; d < spatial_dim; ++d) {
        start[d] = nb_ghosts_left[d];
        end[d] = nb_pts[d] - nb_ghosts_right[d];
        stride[d] = strides[d];
    }

    using Acc = promoted_t<T>;
    Acc result{0};
    if (stride[0] == 1) {
        // x fastest with unit pixel stride: the interior of one grid row is a
        // single contiguous segment of nnx * nb_components_per_pixel values.
        // Reduce it with a vectorized Eigen dot (promoted accumulation) and
        // sum the per-row results in the promoted type — the scalar
        // pixel-by-pixel loop below runs ~4x under memory bandwidth.
        const Index_t row_len = (end[0] - start[0]) * nb_components_per_pixel;
        for (Index_t iz = start[2]; iz < end[2]; ++iz) {
            for (Index_t iy = start[1]; iy < end[1]; ++iy) {
                const Index_t offset =
                    (start[0] + iy * stride[1] + iz * stride[2]) *
                    nb_components_per_pixel;
                Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> row_a(
                    a_data + offset, row_len);
                Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> row_b(
                    b_data + offset, row_len);
                if constexpr (std::is_same_v<Acc, T>) {
                    result += row_a.dot(row_b);
                } else {
                    result += row_a.template cast<Acc>().dot(
                        row_b.template cast<Acc>());
                }
            }
        }
        return static_cast<T>(result);
    }
    for (Index_t iz = start[2]; iz < end[2]; ++iz) {
        for (Index_t iy = start[1]; iy < end[1]; ++iy) {
            for (Index_t ix = start[0]; ix < end[0]; ++ix) {
                const Index_t offset =
                    (ix * stride[0] + iy * stride[1] + iz * stride[2]) *
                    nb_components_per_pixel;
                for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                    result += static_cast<Acc>(
                        conj_product(a_data[offset + c], b_data[offset + c]));
                }
            }
        }
    }
    return static_cast<T>(result);
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
    check_interior_reduction_layout(coll, nb_components_per_pixel,
                                    "interior_three_dots");

    Index_t start[3]{0, 0, 0};
    Index_t end[3]{1, 1, 1};
    Index_t stride[3]{0, 0, 0};
    for (Dim_t d = 0; d < spatial_dim; ++d) {
        start[d] = nb_ghosts_left[d];
        end[d] = nb_pts[d] - nb_ghosts_right[d];
        stride[d] = strides[d];
    }

    using Acc = promoted_t<T>;
    Acc ru{0}, wu{0}, rr{0};
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
                    ru += static_cast<Acc>(conj_product(rv, uv));
                    wu += static_cast<Acc>(conj_product(wv, uv));
                    rr += static_cast<Acc>(conj_product(rv, rv));
                }
            }
        }
    }
    return {static_cast<T>(ru), static_cast<T>(wu), static_cast<T>(rr)};
}

/**
 * Full contiguous-buffer reductions with promoted accumulation (see
 * promoted<T>): Eigen evaluates the cast lazily, so a single-precision
 * reduction still streams fp32 from memory but accumulates in double.
 */
template <typename T>
T full_vecdot(const TypedField<T, HostSpace>& a,
              const TypedField<T, HostSpace>& b) {
    using Acc = promoted_t<T>;
    if constexpr (std::is_same_v<Acc, T>) {
        return a.eigen_vec().dot(b.eigen_vec());
    } else {
        return static_cast<T>(a.eigen_vec().template cast<Acc>().dot(
            b.eigen_vec().template cast<Acc>()));
    }
}

template <typename T>
T full_norm_sq(const TypedField<T, HostSpace>& x) {
    using Acc = promoted_t<T>;
    if constexpr (std::is_same_v<Acc, T>) {
        return static_cast<T>(x.eigen_vec().squaredNorm());
    } else {
        return static_cast<T>(
            x.eigen_vec().template cast<Acc>().squaredNorm());
    }
}

/* ---------------------------------------------------------------------- */
/* Host operation bodies.                                                  */
/*                                                                         */
/* Each operation is written once as a function template over the scalar   */
/* type; the public <T, HostSpace> entry points below are thin delegators.  */
/* The Real and Complex paths differ only in the scalar type and the        */
/* sesquilinear product (interior_vecdot/conj_product handle the latter),   */
/* so a single body serves both — there is no per-type copy to keep in      */
/* sync. This mirrors the cross_host<T> pattern further down.               */
/* ---------------------------------------------------------------------- */

template <typename T>
T vecdot_host(const TypedField<T, HostSpace>& a,
              const TypedField<T, HostSpace>& b) {
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
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (!has_ghosts(global_coll)) {
            return full_vecdot(a, b);
        }
        const Index_t nb_components_per_pixel =
            a.get_nb_components() * a.get_nb_sub_pts();
        // Sum the interior directly; full-buffer-minus-ghosts cancels
        // catastrophically when the interior values are small
        return interior_vecdot(a.data(), b.data(), global_coll,
                               nb_components_per_pixel);
    }
    // LocalFieldCollection: no ghosts, use Eigen
    return full_vecdot(a, b);
}

template <typename T>
void axpy_host(T alpha, const TypedField<T, HostSpace>& x,
               TypedField<T, HostSpace>& y) {
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

template <typename T>
void scal_host(T alpha, TypedField<T, HostSpace>& x) {
    // Operate on full buffer using Eigen
    x.eigen_vec() *= alpha;
}

template <typename T>
void axpby_host(T alpha, const TypedField<T, HostSpace>& x, T beta,
                TypedField<T, HostSpace>& y) {
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpby: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpby: fields must have the same number of entries");
    }
    // Operate on full buffer using Eigen (contiguous, fast): y = a*x + b*y
    y.eigen_vec() = alpha * x.eigen_vec() + beta * y.eigen_vec();
}

template <typename T>
void copy_host(const TypedField<T, HostSpace>& src,
               TypedField<T, HostSpace>& dst) {
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

template <typename T>
T norm_sq_host(const TypedField<T, HostSpace>& x) {
    const auto& coll = x.get_collection();
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (!has_ghosts(global_coll)) {
            return full_norm_sq(x);
        }
        const Index_t nb_components_per_pixel =
            x.get_nb_components() * x.get_nb_sub_pts();
        // Sum the interior directly; full-buffer-minus-ghosts cancels
        // catastrophically when the interior values are small
        return interior_vecdot(x.data(), x.data(), global_coll,
                               nb_components_per_pixel);
    }
    // LocalFieldCollection: no ghosts, use Eigen
    return full_norm_sq(x);
}

template <typename T>
T axpy_norm_sq_host(T alpha, const TypedField<T, HostSpace>& x,
                    TypedField<T, HostSpace>& y) {
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
        if (!has_ghosts(global_coll)) {
            return full_norm_sq(y);
        }
        const Index_t nb_components_per_pixel =
            x.get_nb_components() * x.get_nb_sub_pts();
        return interior_vecdot(y.data(), y.data(), global_coll,
                               nb_components_per_pixel);
    }
    return full_norm_sq(y);
}

// Field-valued scal: x[c, i] *= alpha[c, i], templated on x's scalar type.
// alpha is always Real; a single-component alpha broadcasts over components.
template <typename TA, typename TX>
void scal_field_host(const TypedField<TA, HostSpace>& alpha,
                     TypedField<TX, HostSpace>& x) {
    check_field_alpha(alpha, x);

    const Index_t npix = x.get_nb_entries();
    const Index_t ncomp = x.get_nb_components();
    TX* xd = x.view().data();
    const TA* ad = alpha.view().data();

    if (alpha.get_nb_components() == ncomp) {
        // Elementwise: alpha and x share the same buffer layout
        for (Index_t i{0}; i < npix * ncomp; ++i) {
            xd[i] *= ad[i];
        }
    } else if (x.get_storage_order() == StorageOrder::StructureOfArrays) {
        for (Index_t c{0}; c < ncomp; ++c) {
            TX* xc = xd + c * npix;
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

}  // namespace internal

/* ---------------------------------------------------------------------- */
/* Public entry points: thin <T, HostSpace> delegators to the bodies above.*/
/* These specializations are the host ABI surface that linalg.hh declares. */
/* ---------------------------------------------------------------------- */

template <>
Real vecdot<Real, HostSpace>(const TypedField<Real, HostSpace>& a,
                             const TypedField<Real, HostSpace>& b) {
    return internal::vecdot_host(a, b);
}
template <>
Complex vecdot<Complex, HostSpace>(const TypedField<Complex, HostSpace>& a,
                                   const TypedField<Complex, HostSpace>& b) {
    return internal::vecdot_host(a, b);
}

template <>
void axpy<Real, HostSpace>(Real alpha, const TypedField<Real, HostSpace>& x,
                           TypedField<Real, HostSpace>& y) {
    internal::axpy_host(alpha, x, y);
}
template <>
void axpy<Complex, HostSpace>(Complex alpha,
                              const TypedField<Complex, HostSpace>& x,
                              TypedField<Complex, HostSpace>& y) {
    internal::axpy_host(alpha, x, y);
}

template <>
void scal<Real, HostSpace>(Real alpha, TypedField<Real, HostSpace>& x) {
    internal::scal_host(alpha, x);
}
template <>
void scal<Complex, HostSpace>(Complex alpha, TypedField<Complex, HostSpace>& x) {
    internal::scal_host(alpha, x);
}

template <>
void axpby<Real, HostSpace>(Real alpha, const TypedField<Real, HostSpace>& x,
                            Real beta, TypedField<Real, HostSpace>& y) {
    internal::axpby_host(alpha, x, beta, y);
}
template <>
void axpby<Complex, HostSpace>(Complex alpha,
                               const TypedField<Complex, HostSpace>& x,
                               Complex beta, TypedField<Complex, HostSpace>& y) {
    internal::axpby_host(alpha, x, beta, y);
}

template <>
void copy<Real, HostSpace>(const TypedField<Real, HostSpace>& src,
                           TypedField<Real, HostSpace>& dst) {
    internal::copy_host(src, dst);
}
template <>
void copy<Complex, HostSpace>(const TypedField<Complex, HostSpace>& src,
                              TypedField<Complex, HostSpace>& dst) {
    internal::copy_host(src, dst);
}

template <>
Real norm_sq<Real, HostSpace>(const TypedField<Real, HostSpace>& x) {
    return internal::norm_sq_host(x);
}
template <>
Complex norm_sq<Complex, HostSpace>(const TypedField<Complex, HostSpace>& x) {
    return internal::norm_sq_host(x);
}

template <>
std::array<Real, 3> pipelined_cg_dots<Real, HostSpace>(
    const TypedField<Real, HostSpace>& r, const TypedField<Real, HostSpace>& u,
    const TypedField<Real, HostSpace>& w) {
    for (const auto* other : {&u, &w}) {
        if (&other->get_collection() != &r.get_collection()) {
            throw FieldError(
                "pipelined_cg_dots: fields must belong to the same collection");
        }
        if (other->get_nb_components() != r.get_nb_components() ||
            other->get_nb_sub_pts() != r.get_nb_sub_pts()) {
            throw FieldError(
                "pipelined_cg_dots: fields must have the same number of "
                "components and sub-points");
        }
    }
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
Real axpy_norm_sq<Real, HostSpace>(Real alpha,
                                   const TypedField<Real, HostSpace>& x,
                                   TypedField<Real, HostSpace>& y) {
    return internal::axpy_norm_sq_host(alpha, x, y);
}
template <>
Complex axpy_norm_sq<Complex, HostSpace>(Complex alpha,
                                         const TypedField<Complex, HostSpace>& x,
                                         TypedField<Complex, HostSpace>& y) {
    return internal::axpy_norm_sq_host(alpha, x, y);
}

/* -- Single-precision (Real32 / Complex32) generic-op specializations ------ */
#define MUGRID_LINALG_HOST_SPECIALIZATIONS(T)                                  \
    template <>                                                                \
    T vecdot<T, HostSpace>(const TypedField<T, HostSpace>& a,                  \
                           const TypedField<T, HostSpace>& b) {                \
        return internal::vecdot_host(a, b);                                    \
    }                                                                          \
    template <>                                                                \
    T norm_sq<T, HostSpace>(const TypedField<T, HostSpace>& x) {               \
        return internal::norm_sq_host(x);                                      \
    }                                                                          \
    template <>                                                                \
    void axpy<T, HostSpace>(T alpha, const TypedField<T, HostSpace>& x,        \
                            TypedField<T, HostSpace>& y) {                     \
        internal::axpy_host(alpha, x, y);                                      \
    }                                                                          \
    template <>                                                                \
    void scal<T, HostSpace>(T alpha, TypedField<T, HostSpace>& x) {            \
        internal::scal_host(alpha, x);                                         \
    }                                                                          \
    template <>                                                                \
    void axpby<T, HostSpace>(T alpha, const TypedField<T, HostSpace>& x,       \
                             T beta, TypedField<T, HostSpace>& y) {            \
        internal::axpby_host(alpha, x, beta, y);                              \
    }                                                                          \
    template <>                                                                \
    void copy<T, HostSpace>(const TypedField<T, HostSpace>& src,               \
                            TypedField<T, HostSpace>& dst) {                   \
        internal::copy_host(src, dst);                                         \
    }                                                                          \
    template <>                                                                \
    T axpy_norm_sq<T, HostSpace>(T alpha, const TypedField<T, HostSpace>& x,   \
                                 TypedField<T, HostSpace>& y) {                \
        return internal::axpy_norm_sq_host(alpha, x, y);                       \
    }
MUGRID_LINALG_HOST_SPECIALIZATIONS(Real32)
MUGRID_LINALG_HOST_SPECIALIZATIONS(Complex32)
#undef MUGRID_LINALG_HOST_SPECIALIZATIONS

// pipelined_cg_dots: single-precision real path (mirrors the Real body).
template <>
std::array<Real32, 3> pipelined_cg_dots<Real32, HostSpace>(
    const TypedField<Real32, HostSpace>& r,
    const TypedField<Real32, HostSpace>& u,
    const TypedField<Real32, HostSpace>& w) {
    const auto& coll = r.get_collection();
    const Index_t nb_components_per_pixel =
        r.get_nb_components() * r.get_nb_sub_pts();
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        return internal::interior_three_dots(r.data(), u.data(), w.data(),
                                             global_coll,
                                             nb_components_per_pixel);
    }
    return {r.eigen_vec().dot(u.eigen_vec()), w.eigen_vec().dot(u.eigen_vec()),
            r.eigen_vec().squaredNorm()};
}

template <>
void scal<HostSpace>(const TypedField<Real, HostSpace>& alpha,
                     TypedField<Complex, HostSpace>& x) {
    internal::scal_field_host(alpha, x);
}
template <>
void scal<HostSpace>(const TypedField<Real, HostSpace>& alpha,
                     TypedField<Real, HostSpace>& x) {
    internal::scal_field_host(alpha, x);
}
template <>
void scal<HostSpace>(const TypedField<Real32, HostSpace>& alpha,
                     TypedField<Complex32, HostSpace>& x) {
    internal::scal_field_host(alpha, x);
}
template <>
void scal<HostSpace>(const TypedField<Real32, HostSpace>& alpha,
                     TypedField<Real32, HostSpace>& x) {
    internal::scal_field_host(alpha, x);
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
template <typename RT, typename CT>
void leray_buffers(const RT* k, const RT* invk, const CT* N,
                   CT* out, Index_t npix, bool soa) {
    if (soa) {
        const RT* k0 = k; const RT* k1 = k + npix; const RT* k2 = k + 2 * npix;
        const RT* i0 = invk; const RT* i1 = invk + npix; const RT* i2 = invk + 2 * npix;
        const CT* n0 = N; const CT* n1 = N + npix; const CT* n2 = N + 2 * npix;
        CT* o0 = out; CT* o1 = out + npix; CT* o2 = out + 2 * npix;
        for (Index_t i{0}; i < npix; ++i) {
            const CT s = i0[i] * n0[i] + i1[i] * n1[i] + i2[i] * n2[i];
            o0[i] -= k0[i] * s;
            o1[i] -= k1[i] * s;
            o2[i] -= k2[i] * s;
        }
    } else {
        for (Index_t i{0}; i < npix; ++i) {
            const RT* K = k + 3 * i; const RT* IK = invk + 3 * i;
            const CT* NN = N + 3 * i; CT* O = out + 3 * i;
            const CT s = IK[0] * NN[0] + IK[1] * NN[1] + IK[2] * NN[2];
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

template <>
void cross<Real32, HostSpace>(const TypedField<Real32, HostSpace>& a,
                              const TypedField<Real32, HostSpace>& b,
                              TypedField<Real32, HostSpace>& out) {
    cross_host(a, b, out);
}
template <>
void cross<Complex32, HostSpace>(const TypedField<Complex32, HostSpace>& a,
                                 const TypedField<Complex32, HostSpace>& b,
                                 TypedField<Complex32, HostSpace>& out) {
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

template <>
void leray_project<HostSpace>(const TypedField<Real32, HostSpace>& k,
                              const TypedField<Real32, HostSpace>& invk,
                              const TypedField<Complex32, HostSpace>& N,
                              TypedField<Complex32, HostSpace>& out) {
    const auto& coll = out.get_collection();
    internal::check_three_vector("leray_project", k, coll);
    internal::check_three_vector("leray_project", invk, coll);
    internal::check_three_vector("leray_project", N, coll);
    internal::check_three_vector("leray_project", out, coll);
    const Index_t npix = out.get_nb_entries();
    const bool soa =
        (out.get_storage_order() == StorageOrder::StructureOfArrays);
    internal::leray_buffers(k.view().data(), invk.view().data(),
                            N.view().data(), out.view().data(), npix, soa);
}

}  // namespace linalg
}  // namespace muGrid
