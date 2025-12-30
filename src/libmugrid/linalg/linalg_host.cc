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

namespace muGrid {
namespace linalg {

namespace internal {

/**
 * Helper to compute dot product of ghost region only.
 * We compute full buffer dot with Eigen, then subtract ghosts.
 */
template <typename T>
T ghost_vecdot(const T* a_data, const T* b_data,
               const GlobalFieldCollection& coll,
               Index_t nb_components_per_pixel) {
    const auto spatial_dim = coll.get_spatial_dim();
    const auto& nb_pts = coll.get_nb_subdomain_grid_pts_with_ghosts();
    const auto& nb_ghosts_left = coll.get_nb_ghosts_left();
    const auto& nb_ghosts_right = coll.get_nb_ghosts_right();
    const auto& strides = coll.get_pixels_with_ghosts().get_strides();

    T result = T{0};

    if (spatial_dim == 1) {
        const Index_t nx_total = nb_pts[0];
        const Index_t gx_left = nb_ghosts_left[0];
        const Index_t gx_right = nb_ghosts_right[0];
        const Index_t sx = strides[0];

        // Left ghost region
        for (Index_t ix = 0; ix < gx_left; ++ix) {
            const Index_t offset = ix * sx * nb_components_per_pixel;
            for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                result += a_data[offset + c] * b_data[offset + c];
            }
        }

        // Right ghost region
        for (Index_t ix = nx_total - gx_right; ix < nx_total; ++ix) {
            const Index_t offset = ix * sx * nb_components_per_pixel;
            for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                result += a_data[offset + c] * b_data[offset + c];
            }
        }
    } else if (spatial_dim == 2) {
        const Index_t nx_total = nb_pts[0];
        const Index_t ny_total = nb_pts[1];
        const Index_t gx_left = nb_ghosts_left[0];
        const Index_t gx_right = nb_ghosts_right[0];
        const Index_t gy_left = nb_ghosts_left[1];
        const Index_t gy_right = nb_ghosts_right[1];
        const Index_t sx = strides[0];
        const Index_t sy = strides[1];

        // Left ghost columns (full height)
        for (Index_t iy = 0; iy < ny_total; ++iy) {
            for (Index_t ix = 0; ix < gx_left; ++ix) {
                const Index_t offset = (ix * sx + iy * sy) * nb_components_per_pixel;
                for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                    result += a_data[offset + c] * b_data[offset + c];
                }
            }
        }

        // Right ghost columns (full height)
        for (Index_t iy = 0; iy < ny_total; ++iy) {
            for (Index_t ix = nx_total - gx_right; ix < nx_total; ++ix) {
                const Index_t offset = (ix * sx + iy * sy) * nb_components_per_pixel;
                for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                    result += a_data[offset + c] * b_data[offset + c];
                }
            }
        }

        // Top ghost rows (excluding corners already counted)
        for (Index_t iy = 0; iy < gy_left; ++iy) {
            for (Index_t ix = gx_left; ix < nx_total - gx_right; ++ix) {
                const Index_t offset = (ix * sx + iy * sy) * nb_components_per_pixel;
                for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                    result += a_data[offset + c] * b_data[offset + c];
                }
            }
        }

        // Bottom ghost rows (excluding corners already counted)
        for (Index_t iy = ny_total - gy_right; iy < ny_total; ++iy) {
            for (Index_t ix = gx_left; ix < nx_total - gx_right; ++ix) {
                const Index_t offset = (ix * sx + iy * sy) * nb_components_per_pixel;
                for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                    result += a_data[offset + c] * b_data[offset + c];
                }
            }
        }
    } else if (spatial_dim == 3) {
        const Index_t nx_total = nb_pts[0];
        const Index_t ny_total = nb_pts[1];
        const Index_t nz_total = nb_pts[2];
        const Index_t gx_left = nb_ghosts_left[0];
        const Index_t gx_right = nb_ghosts_right[0];
        const Index_t gy_left = nb_ghosts_left[1];
        const Index_t gy_right = nb_ghosts_right[1];
        const Index_t gz_left = nb_ghosts_left[2];
        const Index_t gz_right = nb_ghosts_right[2];
        const Index_t sx = strides[0];
        const Index_t sy = strides[1];
        const Index_t sz = strides[2];

        // Interior bounds
        const Index_t x_start = gx_left;
        const Index_t x_end = nx_total - gx_right;
        const Index_t y_start = gy_left;
        const Index_t y_end = ny_total - gy_right;
        const Index_t z_start = gz_left;
        const Index_t z_end = nz_total - gz_right;

        // Iterate over all ghost pixels
        for (Index_t iz = 0; iz < nz_total; ++iz) {
            for (Index_t iy = 0; iy < ny_total; ++iy) {
                for (Index_t ix = 0; ix < nx_total; ++ix) {
                    // Skip interior pixels
                    if (ix >= x_start && ix < x_end &&
                        iy >= y_start && iy < y_end &&
                        iz >= z_start && iz < z_end) {
                        continue;
                    }
                    const Index_t offset =
                        (ix * sx + iy * sy + iz * sz) * nb_components_per_pixel;
                    for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                        result += a_data[offset + c] * b_data[offset + c];
                    }
                }
            }
        }
    } else {
        throw FieldError("ghost_vecdot only supports 2D and 3D fields");
    }

    return result;
}

/**
 * Helper to compute squared norm of ghost region only.
 * We compute full buffer norm with Eigen, then subtract ghosts.
 */
template <typename T>
T ghost_norm_sq(const T* data, const GlobalFieldCollection& coll,
                Index_t nb_components_per_pixel) {
    const auto spatial_dim = coll.get_spatial_dim();
    const auto& nb_pts = coll.get_nb_subdomain_grid_pts_with_ghosts();
    const auto& nb_ghosts_left = coll.get_nb_ghosts_left();
    const auto& nb_ghosts_right = coll.get_nb_ghosts_right();
    const auto& strides = coll.get_pixels_with_ghosts().get_strides();

    T result = T{0};

    if (spatial_dim == 1) {
        const Index_t nx_total = nb_pts[0];
        const Index_t gx_left = nb_ghosts_left[0];
        const Index_t gx_right = nb_ghosts_right[0];
        const Index_t sx = strides[0];

        // Left ghost region
        for (Index_t ix = 0; ix < gx_left; ++ix) {
            const Index_t offset = ix * sx * nb_components_per_pixel;
            for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                const T val = data[offset + c];
                result += val * val;
            }
        }

        // Right ghost region
        for (Index_t ix = nx_total - gx_right; ix < nx_total; ++ix) {
            const Index_t offset = ix * sx * nb_components_per_pixel;
            for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                const T val = data[offset + c];
                result += val * val;
            }
        }
    } else if (spatial_dim == 2) {
        const Index_t nx_total = nb_pts[0];
        const Index_t ny_total = nb_pts[1];
        const Index_t gx_left = nb_ghosts_left[0];
        const Index_t gx_right = nb_ghosts_right[0];
        const Index_t gy_left = nb_ghosts_left[1];
        const Index_t gy_right = nb_ghosts_right[1];
        const Index_t sx = strides[0];
        const Index_t sy = strides[1];

        // Left ghost columns (full height)
        for (Index_t iy = 0; iy < ny_total; ++iy) {
            for (Index_t ix = 0; ix < gx_left; ++ix) {
                const Index_t offset = (ix * sx + iy * sy) * nb_components_per_pixel;
                for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                    const T val = data[offset + c];
                    result += val * val;
                }
            }
        }

        // Right ghost columns (full height)
        for (Index_t iy = 0; iy < ny_total; ++iy) {
            for (Index_t ix = nx_total - gx_right; ix < nx_total; ++ix) {
                const Index_t offset = (ix * sx + iy * sy) * nb_components_per_pixel;
                for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                    const T val = data[offset + c];
                    result += val * val;
                }
            }
        }

        // Top ghost rows (excluding corners already counted)
        for (Index_t iy = 0; iy < gy_left; ++iy) {
            for (Index_t ix = gx_left; ix < nx_total - gx_right; ++ix) {
                const Index_t offset = (ix * sx + iy * sy) * nb_components_per_pixel;
                for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                    const T val = data[offset + c];
                    result += val * val;
                }
            }
        }

        // Bottom ghost rows (excluding corners already counted)
        for (Index_t iy = ny_total - gy_right; iy < ny_total; ++iy) {
            for (Index_t ix = gx_left; ix < nx_total - gx_right; ++ix) {
                const Index_t offset = (ix * sx + iy * sy) * nb_components_per_pixel;
                for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                    const T val = data[offset + c];
                    result += val * val;
                }
            }
        }
    } else if (spatial_dim == 3) {
        const Index_t nx_total = nb_pts[0];
        const Index_t ny_total = nb_pts[1];
        const Index_t nz_total = nb_pts[2];
        const Index_t gx_left = nb_ghosts_left[0];
        const Index_t gx_right = nb_ghosts_right[0];
        const Index_t gy_left = nb_ghosts_left[1];
        const Index_t gy_right = nb_ghosts_right[1];
        const Index_t gz_left = nb_ghosts_left[2];
        const Index_t gz_right = nb_ghosts_right[2];
        const Index_t sx = strides[0];
        const Index_t sy = strides[1];
        const Index_t sz = strides[2];

        // Interior bounds
        const Index_t x_start = gx_left;
        const Index_t x_end = nx_total - gx_right;
        const Index_t y_start = gy_left;
        const Index_t y_end = ny_total - gy_right;
        const Index_t z_start = gz_left;
        const Index_t z_end = nz_total - gz_right;

        // Iterate over all ghost pixels (those outside the interior box)
        for (Index_t iz = 0; iz < nz_total; ++iz) {
            for (Index_t iy = 0; iy < ny_total; ++iy) {
                for (Index_t ix = 0; ix < nx_total; ++ix) {
                    // Skip interior pixels
                    if (ix >= x_start && ix < x_end &&
                        iy >= y_start && iy < y_end &&
                        iz >= z_start && iz < z_end) {
                        continue;
                    }
                    const Index_t offset =
                        (ix * sx + iy * sy + iz * sz) * nb_components_per_pixel;
                    for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                        const T val = data[offset + c];
                        result += val * val;
                    }
                }
            }
        }
    } else {
        throw FieldError("ghost_norm_sq only supports 2D and 3D fields");
    }

    return result;
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
        const Index_t nb_components_per_pixel =
            a.get_nb_components() * a.get_nb_sub_pts();
        // Compute full buffer dot with Eigen (fast), subtract ghost contributions
        Real full_dot = a.eigen_vec().dot(b.eigen_vec());
        Real ghost_dot = internal::ghost_vecdot(a.data(), b.data(), global_coll,
                                                nb_components_per_pixel);
        return full_dot - ghost_dot;
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
        const Index_t nb_components_per_pixel =
            a.get_nb_components() * a.get_nb_sub_pts();
        // Compute full buffer dot with Eigen (fast), subtract ghost contributions
        Complex full_dot = a.eigen_vec().dot(b.eigen_vec());
        Complex ghost_dot = internal::ghost_vecdot(a.data(), b.data(), global_coll,
                                                   nb_components_per_pixel);
        return full_dot - ghost_dot;
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
    if (x.get_nb_entries() != y.get_nb_entries()) {
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
    if (x.get_nb_entries() != y.get_nb_entries()) {
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
    if (x.get_nb_entries() != y.get_nb_entries()) {
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
    if (x.get_nb_entries() != y.get_nb_entries()) {
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
    if (src.get_nb_entries() != dst.get_nb_entries()) {
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
    if (src.get_nb_entries() != dst.get_nb_entries()) {
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
        const Index_t nb_components_per_pixel =
            x.get_nb_components() * x.get_nb_sub_pts();
        // Compute full buffer norm with Eigen (fast), subtract ghost contributions
        Real full_norm = x.eigen_vec().squaredNorm();
        Real ghost_norm = internal::ghost_norm_sq(x.data(), global_coll,
                                                  nb_components_per_pixel);
        return full_norm - ghost_norm;
    } else {
        // LocalFieldCollection: no ghosts, use Eigen
        return x.eigen_vec().squaredNorm();
    }
}

template <>
Complex norm_sq<Complex, HostSpace>(const TypedField<Complex, HostSpace>& x) {
    const auto& coll = x.get_collection();

    // Check if this is a GlobalFieldCollection (has ghost regions)
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        const Index_t nb_components_per_pixel =
            x.get_nb_components() * x.get_nb_sub_pts();
        // Compute full buffer norm with Eigen (fast), subtract ghost contributions
        Complex full_norm = x.eigen_vec().squaredNorm();
        Complex ghost_norm = internal::ghost_norm_sq(x.data(), global_coll,
                                                     nb_components_per_pixel);
        return full_norm - ghost_norm;
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
    if (x.get_nb_entries() != y.get_nb_entries()) {
        throw FieldError("axpy_norm_sq: fields must have the same number of entries");
    }

    // Use Eigen's optimized vectorized operations (two passes, but fast)
    // A true single-pass fusion would require breaking Eigen's vectorization
    y.eigen_vec() += alpha * x.eigen_vec();

    const auto& coll = x.get_collection();

    // For GlobalFieldCollection, subtract ghost contributions
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        const Index_t nb_components_per_pixel =
            x.get_nb_components() * x.get_nb_sub_pts();
        Real full_norm = y.eigen_vec().squaredNorm();
        Real ghost_norm = internal::ghost_norm_sq(y.data(), global_coll,
                                                  nb_components_per_pixel);
        return full_norm - ghost_norm;
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
    if (x.get_nb_entries() != y.get_nb_entries()) {
        throw FieldError("axpy_norm_sq: fields must have the same number of entries");
    }

    // Use Eigen's optimized vectorized operations (two passes, but fast)
    // A true single-pass fusion would require breaking Eigen's vectorization
    y.eigen_vec() += alpha * x.eigen_vec();

    const auto& coll = x.get_collection();

    // For GlobalFieldCollection, subtract ghost contributions
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        const Index_t nb_components_per_pixel =
            x.get_nb_components() * x.get_nb_sub_pts();
        Complex full_norm = y.eigen_vec().squaredNorm();
        Complex ghost_norm = internal::ghost_norm_sq(y.data(), global_coll,
                                                     nb_components_per_pixel);
        return full_norm - ghost_norm;
    }

    return y.eigen_vec().squaredNorm();
}

}  // namespace linalg
}  // namespace muGrid
