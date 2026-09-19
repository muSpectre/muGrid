/**
 * @file   transfer.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Sep 2026
 *
 * @brief  Multigrid grid-transfer operators: multilinear prolongation between
 *         nested nodal grids, and its exact adjoint
 *
 * Copyright © 2026 Lars Pastewka
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

#ifndef SRC_LIBMUGRID_OPERATORS_TRANSFER_HH_
#define SRC_LIBMUGRID_OPERATORS_TRANSFER_HH_

#include "core/types.hh"
#include "collection/field_collection_global.hh"
#include "field/field_typed.hh"
#include "memory/memory_space.hh"

#include <array>
#include <sstream>

namespace muGrid {

    // Kernel implementations (transfer.cc for the host, transfer_gpu.cc for the
    // device). Both take explicit strides, so the same kernel serves the host's
    // array-of-structures and the device's structure-of-arrays layout; the
    // caller supplies the matching strides. Pointers are pre-offset to the
    // first *interior* node of their grid.
    namespace transfer_kernels {

        /**
         * @brief Multilinear prolongation, 2D: coarse -> fine, 2:1 refinement.
         *
         * Iterates coarse nodes and writes the 2^Dim fine nodes of each coarse
         * cell, so every fine node is written exactly once (no read-modify-write
         * and no double counting). Reads one coarse node beyond the interior in
         * every direction, which is why the coarse grid needs a ghost layer.
         *
         * @param ncx, ncy  coarse *interior* extent
         * @param nb_dof    scalars per pixel (components x sub-points)
         */
        template <typename T>
        void prolong_2d_host(const T * MUGRID_RESTRICT coarse,
                             T * MUGRID_RESTRICT fine,
                             Index_t ncx, Index_t ncy, Index_t nb_dof,
                             Index_t csx, Index_t csy, Index_t csd,
                             Index_t fsx, Index_t fsy, Index_t fsd);

        //! Multilinear prolongation, 3D. See the 2D overload.
        template <typename T>
        void prolong_3d_host(const T * MUGRID_RESTRICT coarse,
                             T * MUGRID_RESTRICT fine,
                             Index_t ncx, Index_t ncy, Index_t ncz,
                             Index_t nb_dof,
                             Index_t csx, Index_t csy, Index_t csz, Index_t csd,
                             Index_t fsx, Index_t fsy, Index_t fsz,
                             Index_t fsd);

        /**
         * @brief Full-weighting restriction, 2D: fine -> coarse, the exact
         *        adjoint of `prolong_2d_host`.
         *
         * Gathers the 3^Dim fine nodes around each coarse node with weights
         * `(1/2)^(number of non-zero offsets)` — the tensor product of
         * `[1/2, 1, 1/2]`. This is `Pᵀ`, *not* the `1/2^Dim`-normalised full
         * weighting: the residual is a force (a functional), so restricting it
         * carries no measure factor, and `R = Pᵀ` is what keeps the resulting
         * two-grid operator — and hence the preconditioner — symmetric.
         *
         * Reads one fine node beyond the interior in every direction, which is
         * why the fine grid needs a ghost layer (the stiffness stencil already
         * requires one).
         */
        template <typename T>
        void restrict_2d_host(const T * MUGRID_RESTRICT fine,
                              T * MUGRID_RESTRICT coarse,
                              Index_t ncx, Index_t ncy, Index_t nb_dof,
                              Index_t fsx, Index_t fsy, Index_t fsd,
                              Index_t csx, Index_t csy, Index_t csd);

        //! Full-weighting restriction, 3D. See the 2D overload.
        template <typename T>
        void restrict_3d_host(const T * MUGRID_RESTRICT fine,
                              T * MUGRID_RESTRICT coarse,
                              Index_t ncx, Index_t ncy, Index_t ncz,
                              Index_t nb_dof,
                              Index_t fsx, Index_t fsy, Index_t fsz,
                              Index_t fsd,
                              Index_t csx, Index_t csy, Index_t csz,
                              Index_t csd);

    }  // namespace transfer_kernels

    namespace internal {

        //! Geometry of one side of a transfer, resolved from a collection.
        template <Dim_t Dim>
        struct TransferGridInfo {
            std::array<Index_t, Dim> nb_interior{};  //!< interior extent
            std::array<Index_t, Dim> strides{};      //!< pixel strides (AoS)
            Index_t stride_dof{};                    //!< scalar stride
            Index_t offset{};                        //!< to first interior node
            Index_t nb_dof{};                        //!< scalars per pixel
        };

    }  // namespace internal

    /**
     * @class GridTransfer
     * @brief Multilinear prolongation and its adjoint between two nested nodal
     *        grids differing by a factor of two in every direction.
     *
     * Both operations act **component-wise**: they never mix the components of
     * a vector field. For displacement fields that is exactly what is wanted —
     * multilinear interpolation then reproduces linear displacement fields
     * exactly, so its range contains every rigid-body mode and every constant
     * strain. That is the near-nullspace property an algebraic multigrid has to
     * be told explicitly; a geometric hierarchy gets it for free.
     *
     * The two grids must be **nested**: the fine subdomain must be exactly twice
     * the coarse one in extent *and* start at exactly twice its global location,
     * so that coarse node `c` of this rank coincides with fine node `2c` of this
     * rank and no transfer crosses a rank boundary. Constructing both
     * decompositions with the same (power-of-two) `nb_subdivisions` on a
     * power-of-two grid satisfies this; anything else is rejected with an
     * explicit error rather than silently producing garbage at the seams.
     *
     * The caller is responsible for the ghost exchange: `prolong` reads the
     * coarse field's ghosts, `restrict` reads the fine field's, so
     * `communicate_ghosts` must have been called on the *input* field. Neither
     * operation writes ghosts, so neither needs `reduce_ghosts`.
     */
    template <Dim_t Dim>
    class GridTransfer {
        static_assert(Dim == 2 || Dim == 3,
                      "GridTransfer is only implemented for 2D and 3D");

    public:
        GridTransfer() = default;
        GridTransfer(const GridTransfer & other) = delete;
        GridTransfer(GridTransfer && other) = default;
        ~GridTransfer() = default;
        GridTransfer & operator=(const GridTransfer & other) = delete;
        GridTransfer & operator=(GridTransfer && other) = default;

        //! Spatial dimension.
        Dim_t get_spatial_dim() const { return Dim; }

        //! `fine = P coarse`. Reads the coarse field's ghost layer.
        void prolong(const TypedFieldBase<Real> & coarse,
                     TypedFieldBase<Real> & fine) const {
            this->prolong_impl<Real>(coarse, fine);
        }
        //! Single-precision (Real32) overload of prolong().
        void prolong(const TypedFieldBase<Real32> & coarse,
                     TypedFieldBase<Real32> & fine) const {
            this->prolong_impl<Real32>(coarse, fine);
        }

        //! `coarse = Pᵀ fine`. Reads the fine field's ghost layer.
        void restrict(const TypedFieldBase<Real> & fine,
                      TypedFieldBase<Real> & coarse) const {
            this->restrict_impl<Real>(fine, coarse);
        }
        //! Single-precision (Real32) overload of restrict().
        void restrict(const TypedFieldBase<Real32> & fine,
                      TypedFieldBase<Real32> & coarse) const {
            this->restrict_impl<Real32>(fine, coarse);
        }

    private:
        static const char * operator_name() {
            return Dim == 2 ? "GridTransfer2D" : "GridTransfer3D";
        }

        /**
         * @brief Validate the pair and resolve both grids' geometry.
         *
         * Checked here rather than at construction so the object stays
         * stateless and cannot outlive the collections it describes; the cost
         * is a handful of integer comparisons against a kernel that touches
         * every node.
         */
        static std::array<internal::TransferGridInfo<Dim>, 2>
        validate(const Field & coarse_field, const Field & fine_field) {
            const std::string op{operator_name()};

            auto as_global = [&op](const Field & field, const char * which) {
                const auto * fc = dynamic_cast<const GlobalFieldCollection *>(
                    &field.get_collection());
                if (fc == nullptr) {
                    throw RuntimeError{op + ": the " + which +
                                       " field must live on a "
                                       "GlobalFieldCollection"};
                }
                if (fc->get_spatial_dim() != Dim) {
                    std::stringstream err{};
                    err << op << ": the " << which << " field is "
                        << fc->get_spatial_dim() << "-dimensional";
                    throw RuntimeError{err.str()};
                }
                return fc;
            };
            const auto * coarse_fc = as_global(coarse_field, "coarse");
            const auto * fine_fc = as_global(fine_field, "fine");

            const Index_t nb_dof{coarse_field.get_nb_components() *
                                 coarse_field.get_nb_sub_pts()};
            const Index_t fine_nb_dof{fine_field.get_nb_components() *
                                      fine_field.get_nb_sub_pts()};
            if (nb_dof != fine_nb_dof) {
                std::stringstream err{};
                err << op << ": the coarse field carries " << nb_dof
                    << " scalar(s) per pixel but the fine field carries "
                    << fine_nb_dof
                    << "; a grid transfer acts component-wise and cannot "
                       "change the number of degrees of freedom";
                throw RuntimeError{err.str()};
            }

            std::array<internal::TransferGridInfo<Dim>, 2> info{};
            const std::array<const GlobalFieldCollection *, 2> fcs{coarse_fc,
                                                                   fine_fc};
            const std::array<const char *, 2> names{"coarse", "fine"};
            for (std::size_t side{0}; side < 2; ++side) {
                const auto & fc = *fcs[side];
                const auto & pixel_strides{
                    fc.get_pixels_with_ghosts().get_strides()};
                const auto & with_ghosts{
                    fc.get_nb_subdomain_grid_pts_with_ghosts()};
                const auto & ghosts_left{fc.get_nb_ghosts_left()};
                const auto & ghosts_right{fc.get_nb_ghosts_right()};
                const auto interior{fc.get_nb_subdomain_grid_pts_without_ghosts()};

                auto & side_info = info[side];
                side_info.nb_dof = nb_dof;
                side_info.stride_dof = 1;  // host: array of structures
                side_info.offset = 0;
                Index_t expected_pixel_stride{1};
                for (Dim_t dir{0}; dir < Dim; ++dir) {
                    if (pixel_strides[dir] != expected_pixel_stride) {
                        std::stringstream err{};
                        err << op << ": the " << names[side]
                            << " collection must use dense column-major pixel "
                               "storage, but has pixel stride "
                            << pixel_strides[dir] << " on axis " << dir
                            << " where " << expected_pixel_stride
                            << " was expected";
                        throw RuntimeError{err.str()};
                    }
                    if (ghosts_left[dir] < 1 or ghosts_right[dir] < 1) {
                        std::stringstream err{};
                        err << op << ": the " << names[side]
                            << " collection needs at least one ghost layer on "
                               "each side of axis "
                            << dir << " (has " << ghosts_left[dir] << " left, "
                            << ghosts_right[dir] << " right)";
                        throw RuntimeError{err.str()};
                    }
                    side_info.nb_interior[dir] = interior[dir];
                    side_info.strides[dir] = expected_pixel_stride * nb_dof;
                    side_info.offset +=
                        ghosts_left[dir] * side_info.strides[dir];
                    expected_pixel_stride *= with_ghosts[dir];
                }
            }

            // Nesting: same rank must own coarse node c and fine node 2c.
            const auto coarse_loc{
                coarse_fc->get_subdomain_locations_without_ghosts()};
            const auto fine_loc{
                fine_fc->get_subdomain_locations_without_ghosts()};
            for (Dim_t dir{0}; dir < Dim; ++dir) {
                if (info[1].nb_interior[dir] != 2 * info[0].nb_interior[dir]) {
                    std::stringstream err{};
                    err << op << ": grids are not nested on axis " << dir
                        << " — the fine subdomain has "
                        << info[1].nb_interior[dir]
                        << " points where twice the coarse subdomain's "
                        << info[0].nb_interior[dir] << " is "
                        << 2 * info[0].nb_interior[dir]
                        << ". Build every level with the same power-of-two "
                           "nb_subdivisions on a grid whose extent stays "
                           "divisible by two at each level.";
                    throw RuntimeError{err.str()};
                }
                if (fine_loc[dir] != 2 * coarse_loc[dir]) {
                    std::stringstream err{};
                    err << op << ": grids are not nested on axis " << dir
                        << " — the fine subdomain starts at global index "
                        << fine_loc[dir] << " where twice the coarse "
                        << "subdomain's start " << coarse_loc[dir] << " is "
                        << 2 * coarse_loc[dir]
                        << ". The two decompositions disagree about which rank "
                           "owns which region.";
                    throw RuntimeError{err.str()};
                }
            }
            return info;
        }

        template <typename T>
        void prolong_impl(const TypedFieldBase<T> & coarse_field,
                          TypedFieldBase<T> & fine_field) const {
            const auto info = validate(coarse_field, fine_field);
            const auto & c = info[0];
            const auto & f = info[1];
            const T * coarse = coarse_field.data() + c.offset;
            T * fine = fine_field.data() + f.offset;
            if constexpr (Dim == 2) {
                transfer_kernels::prolong_2d_host<T>(
                    coarse, fine, c.nb_interior[0], c.nb_interior[1], c.nb_dof,
                    c.strides[0], c.strides[1], c.stride_dof, f.strides[0],
                    f.strides[1], f.stride_dof);
            } else {
                transfer_kernels::prolong_3d_host<T>(
                    coarse, fine, c.nb_interior[0], c.nb_interior[1],
                    c.nb_interior[2], c.nb_dof, c.strides[0], c.strides[1],
                    c.strides[2], c.stride_dof, f.strides[0], f.strides[1],
                    f.strides[2], f.stride_dof);
            }
        }

        template <typename T>
        void restrict_impl(const TypedFieldBase<T> & fine_field,
                           TypedFieldBase<T> & coarse_field) const {
            const auto info = validate(coarse_field, fine_field);
            const auto & c = info[0];
            const auto & f = info[1];
            const T * fine = fine_field.data() + f.offset;
            T * coarse = coarse_field.data() + c.offset;
            if constexpr (Dim == 2) {
                transfer_kernels::restrict_2d_host<T>(
                    fine, coarse, c.nb_interior[0], c.nb_interior[1], c.nb_dof,
                    f.strides[0], f.strides[1], f.stride_dof, c.strides[0],
                    c.strides[1], c.stride_dof);
            } else {
                transfer_kernels::restrict_3d_host<T>(
                    fine, coarse, c.nb_interior[0], c.nb_interior[1],
                    c.nb_interior[2], c.nb_dof, f.strides[0], f.strides[1],
                    f.strides[2], f.stride_dof, c.strides[0], c.strides[1],
                    c.strides[2], c.stride_dof);
            }
        }
    };

    //! 2D grid transfer (bilinear prolongation).
    using GridTransfer2D = GridTransfer<2>;
    //! 3D grid transfer (trilinear prolongation).
    using GridTransfer3D = GridTransfer<3>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_TRANSFER_HH_
