/**
 * @file   isotropic_stiffness.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   26 Jun 2026
 *
 * @brief  Dimension-templated fused isotropic linear-elastic stiffness operator
 *
 * Computes K @ u = B^T C B @ u for isotropic linear elastic materials,
 * exploiting K = 2μ G + λ V where G and V are geometry-only matrices shared by
 * all voxels (so spatially-varying materials cost O(N × 2) rather than
 * O(N × NB_ELEMENT_DOFS²)). The 2D (linear triangles) and 3D (linear
 * tetrahedra) operators shared their entire interface, validation and device
 * dispatch, differing only in the dimension, the element counts and the
 * geometry of G/V. They are unified here into a single
 * `template <Dim_t Dim> IsotropicStiffnessOperator`, which inherits the
 * `MaterialOperator` interface (Phase 5c) so the λ/μ-carrying signature and the
 * stencil/ghost metadata are declared once. The two genuinely dimension-
 * specific operations — `precompute_matrices` and `apply_impl` (host + device)
 * — remain as explicit specializations in isotropic_stiffness_2d.cc /
 * isotropic_stiffness_3d.cc / isotropic_stiffness_gpu.cc. The historical class
 * names remain as type aliases so no binding or downstream code changes.
 *
 * Copyright © 2025 Lars Pastewka
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

#ifndef SRC_LIBMUGRID_OPERATORS_SOLIDS_ISOTROPIC_STIFFNESS_HH_
#define SRC_LIBMUGRID_OPERATORS_SOLIDS_ISOTROPIC_STIFFNESS_HH_

#include "core/types.hh"
#include "core/exception.hh"
#include "field/field_typed.hh"
#include "memory/memory_space.hh"
#include "operators/linear.hh"
#include "collection/field_collection_global.hh"

#include <array>
#include <sstream>
#include <string>
#include <vector>

namespace muGrid {

    /**
     * @class MaterialOperator
     * @brief Interface for elliptic operators parameterised by per-pixel Lamé
     *        fields, i.e. `force = K(λ, μ) @ displacement`.
     *
     * This is the material-operator sibling of LinearOperator: where
     * LinearOperator carries the `apply(in, out)` signature, MaterialOperator
     * carries the `apply(displacement, λ, μ, force)` signature of a stiffness
     * operator whose coefficients vary in space. Declaring it once means the
     * stencil/ghost metadata and the λ/μ apply interface are shared by every
     * concrete material operator instead of being re-declared per dimension.
     */
    class MaterialOperator {
       public:
        MaterialOperator() = default;
        virtual ~MaterialOperator() = default;
        MaterialOperator(const MaterialOperator &) = delete;
        MaterialOperator(MaterialOperator &&) = default;
        MaterialOperator & operator=(const MaterialOperator &) = delete;
        MaterialOperator & operator=(MaterialOperator &&) = default;

        //! force = K(λ, μ) @ displacement
        virtual void apply(const TypedFieldBase<Real> & displacement,
                           const TypedFieldBase<Real> & lambda,
                           const TypedFieldBase<Real> & mu,
                           TypedFieldBase<Real> & force) const = 0;

        //! force += alpha * K(λ, μ) @ displacement
        virtual void
        apply_increment(const TypedFieldBase<Real> & displacement,
                        const TypedFieldBase<Real> & lambda,
                        const TypedFieldBase<Real> & mu, Real alpha,
                        TypedFieldBase<Real> & force) const = 0;

        //! Spatial dimension of the operator.
        virtual Dim_t get_spatial_dim() const = 0;
        //! Stencil offset in pixels.
        virtual Shape_t get_offset() const = 0;
        //! Stencil shape in pixels.
        virtual Shape_t get_stencil_shape() const = 0;
        //! Ghost layers required by apply().
        virtual GhostRequirement get_apply_ghost_requirement() const = 0;

        //! Ghost layers sufficient for all operations (apply is the only one).
        GhostRequirement get_ghost_requirement() const {
            return this->get_apply_ghost_requirement();
        }
    };

    /**
     * @struct IsotropicStiffnessTraits
     * @brief Dimension-varying element counts of the stiffness operator.
     */
    template <Dim_t Dim>
    struct IsotropicStiffnessTraits;
    template <>
    struct IsotropicStiffnessTraits<2> {
        static constexpr Index_t nb_nodes = 4;          //!< 4 corner nodes
        static constexpr Index_t nb_dofs_per_node = 2;  //!< ux, uy
        static constexpr Index_t nb_quad = 2;           //!< 2 triangles
    };
    template <>
    struct IsotropicStiffnessTraits<3> {
        static constexpr Index_t nb_nodes = 8;          //!< 8 corner nodes
        static constexpr Index_t nb_dofs_per_node = 3;  //!< ux, uy, uz
        static constexpr Index_t nb_quad = 5;           //!< 5 tetrahedra
    };

    // Kernel declarations. The host kernels are defined in
    // isotropic_stiffness_2d.cc / isotropic_stiffness_3d.cc, the device kernels
    // in isotropic_stiffness_gpu.cc.
    namespace isotropic_stiffness_kernels {

        void isotropic_stiffness_2d_host(
            const Real * MUGRID_RESTRICT displacement,
            const Real * MUGRID_RESTRICT lambda,
            const Real * MUGRID_RESTRICT mu, Real * MUGRID_RESTRICT force,
            Index_t nnx, Index_t nny, Index_t nelx, Index_t nely,
            Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_d,
            Index_t mat_stride_x, Index_t mat_stride_y, Index_t force_stride_x,
            Index_t force_stride_y, Index_t force_stride_d, const Real * G,
            const Real * V, Real alpha, bool increment);

        void isotropic_stiffness_3d_host(
            const Real * MUGRID_RESTRICT displacement,
            const Real * MUGRID_RESTRICT lambda,
            const Real * MUGRID_RESTRICT mu, Real * MUGRID_RESTRICT force,
            Index_t nnx, Index_t nny, Index_t nnz, Index_t nelx, Index_t nely,
            Index_t nelz, Index_t disp_stride_x, Index_t disp_stride_y,
            Index_t disp_stride_z, Index_t disp_stride_d, Index_t mat_stride_x,
            Index_t mat_stride_y, Index_t mat_stride_z, Index_t force_stride_x,
            Index_t force_stride_y, Index_t force_stride_z,
            Index_t force_stride_d, const Real * G, const Real * V, Real alpha,
            bool increment);

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        void isotropic_stiffness_2d_gpu(
            const Real * displacement, const Real * lambda, const Real * mu,
            Real * force, Index_t nnx, Index_t nny, Index_t nelx, Index_t nely,
            Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_d,
            Index_t mat_stride_x, Index_t mat_stride_y, Index_t force_stride_x,
            Index_t force_stride_y, Index_t force_stride_d, const Real * G,
            const Real * V, Real alpha, bool increment);

        void isotropic_stiffness_3d_gpu(
            const Real * displacement, const Real * lambda, const Real * mu,
            Real * force, Index_t nnx, Index_t nny, Index_t nnz, Index_t nelx,
            Index_t nely, Index_t nelz, Index_t disp_stride_x,
            Index_t disp_stride_y, Index_t disp_stride_z, Index_t disp_stride_d,
            Index_t mat_stride_x, Index_t mat_stride_y, Index_t mat_stride_z,
            Index_t force_stride_x, Index_t force_stride_y,
            Index_t force_stride_z, Index_t force_stride_d, const Real * G,
            const Real * V, Real alpha, bool increment);
#endif

    }  // namespace isotropic_stiffness_kernels

    namespace internal {

        /**
         * @brief Result of validate_stiffness_fields.
         *
         * Holds the two collections (already verified to be global) and the
         * per-axis computable region (grid-with-ghosts minus the one-cell
         * stencil on each side).
         */
        template <Dim_t Dim>
        struct StiffnessFieldInfo {
            const GlobalFieldCollection * disp_fc;
            const GlobalFieldCollection * mat_fc;
            std::array<Index_t, Dim> nb_computable;
        };

        /**
         * @brief Shared validation for the isotropic stiffness operators.
         *
         * Used by the host and device, 2D and 3D apply implementations, which
         * differ only in the stride bookkeeping that follows. Checks that both
         * the displacement/force field and the material (λ, μ) field live on a
         * GlobalFieldCollection with at least one ghost layer on every side,
         * and that their computable regions agree. The dimension is handled by
         * a loop over the @p Dim axes rather than hard-coded `[0], [1], [2]`
         * index checks, so the rule lives in one place.
         *
         * @param disp_coll collection of the displacement/force field
         * @param mat_coll  collection of the material (λ, μ) fields
         * @throws RuntimeError if any check fails
         */
        template <Dim_t Dim>
        StiffnessFieldInfo<Dim>
        validate_stiffness_fields(const FieldCollection & disp_coll,
                                  const FieldCollection & mat_coll) {
            // The kernel gathers neighbouring elements at offset -1 and their
            // nodes at offset +1, i.e. one ghost cell on each side per axis.
            constexpr Index_t STENCIL = 1;
            const std::string op{"IsotropicStiffnessOperator" +
                                 std::to_string(Dim) + "D"};

            auto * disp_fc =
                dynamic_cast<const GlobalFieldCollection *>(&disp_coll);
            if (!disp_fc) {
                throw RuntimeError(op + " requires GlobalFieldCollection");
            }
            auto * mat_fc =
                dynamic_cast<const GlobalFieldCollection *>(&mat_coll);
            if (!mat_fc) {
                throw RuntimeError(
                    op + " material fields require GlobalFieldCollection");
            }

            // Both fields need at least one ghost on each side of every axis.
            auto check_ghosts = [&op](const GlobalFieldCollection & fc,
                                      const std::string & what) {
                auto left = fc.get_nb_ghosts_left();
                auto right = fc.get_nb_ghosts_right();
                for (Dim_t i = 0; i < Dim; ++i) {
                    if (left[i] < 1) {
                        throw RuntimeError(op +
                                           " requires at least 1 ghost cell on "
                                           "the left side of " +
                                           what);
                    }
                    if (right[i] < 1) {
                        throw RuntimeError(op +
                                           " requires at least 1 ghost cell on "
                                           "the right side of " +
                                           what);
                    }
                }
            };
            check_ghosts(*disp_fc, "displacement/force fields");

            // Computable region = grid-with-ghosts minus the stencil per side.
            auto disp_with_ghosts =
                disp_fc->get_nb_subdomain_grid_pts_with_ghosts();
            auto mat_with_ghosts =
                mat_fc->get_nb_subdomain_grid_pts_with_ghosts();
            StiffnessFieldInfo<Dim> info{disp_fc, mat_fc, {}};
            std::array<Index_t, Dim> nb_mat{};
            for (Dim_t i = 0; i < Dim; ++i) {
                info.nb_computable[i] = disp_with_ghosts[i] - 2 * STENCIL;
                nb_mat[i] = mat_with_ghosts[i] - 2 * STENCIL;
            }

            // Material and node computable regions must coincide.
            if (nb_mat != info.nb_computable) {
                std::stringstream err{};
                err << op << ": material field computable region (";
                for (Dim_t i = 0; i < Dim; ++i) {
                    err << (i ? ", " : "") << nb_mat[i];
                }
                err << ") must match node field computable region (";
                for (Dim_t i = 0; i < Dim; ++i) {
                    err << (i ? ", " : "") << info.nb_computable[i];
                }
                err << ")";
                throw RuntimeError(err.str());
            }

            check_ghosts(*mat_fc, "material fields (lambda, mu)");

            return info;
        }

    }  // namespace internal

    /**
     * @class IsotropicStiffnessOperator
     * @brief Fused stiffness operator for isotropic linear elastic materials
     *        (2D linear triangles / 3D linear tetrahedra).
     *
     * Computes force = K @ displacement with K = 2μ G + λ V. G and V are
     * precomputed once at construction from the grid spacing.
     */
    template <Dim_t Dim>
    class IsotropicStiffnessOperator : public MaterialOperator {
        static_assert(Dim == 2 || Dim == 3,
                      "IsotropicStiffnessOperator is only implemented for 2D "
                      "and 3D");

       public:
        //! Number of nodes per pixel/voxel (4 in 2D, 8 in 3D)
        static constexpr Index_t NB_NODES =
            IsotropicStiffnessTraits<Dim>::nb_nodes;
        //! Number of DOFs per node (= spatial dimension)
        static constexpr Index_t NB_DOFS_PER_NODE =
            IsotropicStiffnessTraits<Dim>::nb_dofs_per_node;
        //! Total DOFs per element
        static constexpr Index_t NB_ELEMENT_DOFS = NB_NODES * NB_DOFS_PER_NODE;
        //! Number of quadrature points (2 triangles in 2D, 5 tetrahedra in 3D)
        static constexpr Index_t NB_QUAD =
            IsotropicStiffnessTraits<Dim>::nb_quad;

        //! Flat element-matrix storage type.
        using ElementMatrix =
            std::array<Real, NB_ELEMENT_DOFS * NB_ELEMENT_DOFS>;

        /**
         * @brief Construct with grid spacing (length Dim); precomputes G and V.
         */
        explicit IsotropicStiffnessOperator(
            const std::vector<Real> & grid_spacing)
            : grid_spacing{grid_spacing} {
            if (static_cast<Index_t>(this->grid_spacing.size()) != Dim) {
                throw RuntimeError(std::to_string(Dim) +
                                   "D operator requires " + std::to_string(Dim) +
                                   "D grid spacing");
            }
            this->precompute_matrices();
        }

        //! Default constructor is deleted
        IsotropicStiffnessOperator() = delete;
        //! Destructor
        ~IsotropicStiffnessOperator() override = default;

        // ---- Host interface (overrides MaterialOperator) ----
        void apply(const TypedFieldBase<Real> & displacement,
                   const TypedFieldBase<Real> & lambda,
                   const TypedFieldBase<Real> & mu,
                   TypedFieldBase<Real> & force) const override {
            this->apply_impl(displacement, lambda, mu, 1.0, force, false);
        }
        void apply_increment(const TypedFieldBase<Real> & displacement,
                             const TypedFieldBase<Real> & lambda,
                             const TypedFieldBase<Real> & mu, Real alpha,
                             TypedFieldBase<Real> & force) const override {
            this->apply_impl(displacement, lambda, mu, alpha, force, true);
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        // ---- Device interface ----
        void apply(const TypedFieldBase<Real, DefaultDeviceSpace> & displacement,
                   const TypedFieldBase<Real, DefaultDeviceSpace> & lambda,
                   const TypedFieldBase<Real, DefaultDeviceSpace> & mu,
                   TypedFieldBase<Real, DefaultDeviceSpace> & force) const {
            this->apply_impl(displacement, lambda, mu, 1.0, force, false);
        }
        void apply_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & displacement,
            const TypedFieldBase<Real, DefaultDeviceSpace> & lambda,
            const TypedFieldBase<Real, DefaultDeviceSpace> & mu, Real alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & force) const {
            this->apply_impl(displacement, lambda, mu, alpha, force, true);
        }
#endif

        //! Precomputed G = Σ_q w_q B_q^T B_q (shear stiffness geometry).
        const ElementMatrix & get_G() const { return G_matrix; }
        //! Precomputed V = Σ_q w_q (B_q^T m)(m^T B_q) (volumetric geometry).
        const ElementMatrix & get_V() const { return V_matrix; }

        //! Spatial dimension.
        Dim_t get_spatial_dim() const override { return Dim; }

        //! Stencil offset in pixels (gathers neighbours in [-1, +1]).
        Shape_t get_offset() const override {
            if constexpr (Dim == 2) {
                return Shape_t{-1, -1};
            } else {
                return Shape_t{-1, -1, -1};
            }
        }

        //! Stencil shape in pixels (3 in every direction).
        Shape_t get_stencil_shape() const override {
            if constexpr (Dim == 2) {
                return Shape_t{3, 3};
            } else {
                return Shape_t{3, 3, 3};
            }
        }

        //! Ghost layers required by apply() (one on each side, every axis).
        GhostRequirement get_apply_ghost_requirement() const override {
            if constexpr (Dim == 2) {
                return GhostRequirement{Shape_t{1, 1}, Shape_t{1, 1}};
            } else {
                return GhostRequirement{Shape_t{1, 1, 1}, Shape_t{1, 1, 1}};
            }
        }

       private:
        std::vector<Real> grid_spacing;
        ElementMatrix G_matrix;  //!< 2μ coefficient geometry
        ElementMatrix V_matrix;  //!< λ coefficient geometry

        //! Compute the geometry matrices G and V (dimension-specific; explicit
        //! specialization in isotropic_stiffness_{2,3}d.cc).
        void precompute_matrices();

        //! Host apply with optional increment (dimension-specific; explicit
        //! specialization in isotropic_stiffness_{2,3}d.cc).
        void apply_impl(const TypedFieldBase<Real> & displacement,
                        const TypedFieldBase<Real> & lambda,
                        const TypedFieldBase<Real> & mu, Real alpha,
                        TypedFieldBase<Real> & force, bool increment) const;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        //! Device apply with optional increment (dimension-specific; explicit
        //! specialization in isotropic_stiffness_gpu.cc).
        void apply_impl(
            const TypedFieldBase<Real, DefaultDeviceSpace> & displacement,
            const TypedFieldBase<Real, DefaultDeviceSpace> & lambda,
            const TypedFieldBase<Real, DefaultDeviceSpace> & mu, Real alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & force,
            bool increment) const;
#endif
    };

    // Explicit specialization declarations: the bodies live in the .cc files,
    // so every translation unit that instantiates the operator must see that
    // these members are specialized elsewhere (and not implicitly instantiated
    // from the — intentionally undefined — primary template).
    template <>
    void IsotropicStiffnessOperator<2>::precompute_matrices();
    template <>
    void IsotropicStiffnessOperator<2>::apply_impl(
        const TypedFieldBase<Real> &, const TypedFieldBase<Real> &,
        const TypedFieldBase<Real> &, Real, TypedFieldBase<Real> &, bool) const;
    template <>
    void IsotropicStiffnessOperator<3>::precompute_matrices();
    template <>
    void IsotropicStiffnessOperator<3>::apply_impl(
        const TypedFieldBase<Real> &, const TypedFieldBase<Real> &,
        const TypedFieldBase<Real> &, Real, TypedFieldBase<Real> &, bool) const;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    template <>
    void IsotropicStiffnessOperator<2>::apply_impl(
        const TypedFieldBase<Real, DefaultDeviceSpace> &,
        const TypedFieldBase<Real, DefaultDeviceSpace> &,
        const TypedFieldBase<Real, DefaultDeviceSpace> &, Real,
        TypedFieldBase<Real, DefaultDeviceSpace> &, bool) const;
    template <>
    void IsotropicStiffnessOperator<3>::apply_impl(
        const TypedFieldBase<Real, DefaultDeviceSpace> &,
        const TypedFieldBase<Real, DefaultDeviceSpace> &,
        const TypedFieldBase<Real, DefaultDeviceSpace> &, Real,
        TypedFieldBase<Real, DefaultDeviceSpace> &, bool) const;
#endif

    //! 2D fused isotropic stiffness operator. Preserves the historical name.
    using IsotropicStiffnessOperator2D = IsotropicStiffnessOperator<2>;
    //! 3D fused isotropic stiffness operator. Preserves the historical name.
    using IsotropicStiffnessOperator3D = IsotropicStiffnessOperator<3>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_SOLIDS_ISOTROPIC_STIFFNESS_HH_
