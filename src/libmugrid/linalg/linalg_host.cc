/**
 * @file   linalg_host.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2024
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
 * Helper function to compute dot product over interior region.
 * Handles 2D and 3D cases with proper stride computation.
 */
template <typename T>
T vecdot_interior(const T* a_data, const T* b_data,
                  const GlobalFieldCollection& coll,
                  Index_t nb_components_per_pixel) {
    const auto spatial_dim = coll.get_spatial_dim();
    const auto& nb_pts_with_ghosts = coll.get_nb_subdomain_grid_pts_with_ghosts();
    const auto& nb_ghosts_left = coll.get_nb_ghosts_left();
    const auto& nb_ghosts_right = coll.get_nb_ghosts_right();
    const auto& strides = coll.get_pixels_with_ghosts().get_strides();

    // Compute interior shape (without ghosts)
    DynGridIndex nb_interior(spatial_dim);
    for (Dim_t d = 0; d < spatial_dim; ++d) {
        nb_interior[d] = nb_pts_with_ghosts[d] - nb_ghosts_left[d] - nb_ghosts_right[d];
    }

    T result = T{0};

    if (spatial_dim == 2) {
        const Index_t nx = nb_interior[0];
        const Index_t ny = nb_interior[1];
        const Index_t gx = nb_ghosts_left[0];
        const Index_t gy = nb_ghosts_left[1];
        const Index_t sx = strides[0];
        const Index_t sy = strides[1];

        #pragma omp parallel for reduction(+:result) collapse(2)
        for (Index_t iy = 0; iy < ny; ++iy) {
            for (Index_t ix = 0; ix < nx; ++ix) {
                const Index_t pixel_offset =
                    ((gx + ix) * sx + (gy + iy) * sy) * nb_components_per_pixel;
                for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                    result += a_data[pixel_offset + c] * b_data[pixel_offset + c];
                }
            }
        }
    } else if (spatial_dim == 3) {
        const Index_t nx = nb_interior[0];
        const Index_t ny = nb_interior[1];
        const Index_t nz = nb_interior[2];
        const Index_t gx = nb_ghosts_left[0];
        const Index_t gy = nb_ghosts_left[1];
        const Index_t gz = nb_ghosts_left[2];
        const Index_t sx = strides[0];
        const Index_t sy = strides[1];
        const Index_t sz = strides[2];

        #pragma omp parallel for reduction(+:result) collapse(3)
        for (Index_t iz = 0; iz < nz; ++iz) {
            for (Index_t iy = 0; iy < ny; ++iy) {
                for (Index_t ix = 0; ix < nx; ++ix) {
                    const Index_t pixel_offset =
                        ((gx + ix) * sx + (gy + iy) * sy + (gz + iz) * sz) *
                        nb_components_per_pixel;
                    for (Index_t c = 0; c < nb_components_per_pixel; ++c) {
                        result += a_data[pixel_offset + c] * b_data[pixel_offset + c];
                    }
                }
            }
        }
    } else {
        throw FieldError("vecdot only supports 2D and 3D fields");
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
        return internal::vecdot_interior(a.data(), b.data(), global_coll,
                                         nb_components_per_pixel);
    } else {
        // LocalFieldCollection: no ghosts, use full buffer
        // Use Eigen for efficiency
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
        return internal::vecdot_interior(a.data(), b.data(), global_coll,
                                         nb_components_per_pixel);
    } else {
        // LocalFieldCollection: no ghosts, use full buffer
        // Use Eigen for efficiency
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
    return vecdot<Real, HostSpace>(x, x);
}

template <>
Complex norm_sq<Complex, HostSpace>(const TypedField<Complex, HostSpace>& x) {
    return vecdot<Complex, HostSpace>(x, x);
}

}  // namespace linalg
}  // namespace muGrid
