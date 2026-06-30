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
#include "operators/fem_element.hh"
#include "collection/field_collection_global.hh"

#include <array>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace muGrid {

    //! Return a `const T*` view of a double-valued geometry array (G, V, Dbar
    //! or E_macro): for T == Real this is `src.data()` with no copy; otherwise
    //! the values are converted to T once into the caller-provided `scratch`
    //! and its data() returned. Keeps the double-precision path zero-overhead
    //! while letting the single-precision kernels run monomorphically in T.
    template <typename T, typename SrcArray, typename ScratchArray>
    const T * geometry_as(const SrcArray & src,
                          [[maybe_unused]] ScratchArray & scratch) {
        if constexpr (std::is_same_v<T, Real>) {
            return src.data();
        } else {
            for (std::size_t i = 0; i < src.size(); ++i) {
                scratch[i] = static_cast<T>(src[i]);
            }
            return scratch.data();
        }
    }

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
                           TypedFieldBase<Real> & force) const {
            precision_not_implemented();
        }
        //! Single-precision (Real32) overload of apply().
        virtual void apply(const TypedFieldBase<Real32> & displacement,
                           const TypedFieldBase<Real32> & lambda,
                           const TypedFieldBase<Real32> & mu,
                           TypedFieldBase<Real32> & force) const {
            precision_not_implemented();
        }

        //! force += alpha * K(λ, μ) @ displacement
        virtual void
        apply_increment(const TypedFieldBase<Real> & displacement,
                        const TypedFieldBase<Real> & lambda,
                        const TypedFieldBase<Real> & mu, Real alpha,
                        TypedFieldBase<Real> & force) const {
            precision_not_implemented();
        }
        //! Single-precision (Real32) overload of apply_increment().
        virtual void
        apply_increment(const TypedFieldBase<Real32> & displacement,
                        const TypedFieldBase<Real32> & lambda,
                        const TypedFieldBase<Real32> & mu, Real32 alpha,
                        TypedFieldBase<Real32> & force) const {
            precision_not_implemented();
        }

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

       protected:
        //! Default body for the precision overloads an operator does not
        //! implement (operators are monomorphic in their scalar type; see
        //! LinearOperator::precision_not_implemented).
        [[noreturn]] static void precision_not_implemented() {
            throw RuntimeError(
                "This material operator was not instantiated for the scalar "
                "precision of the supplied fields. Construct the operator for "
                "the matching type (e.g. the Real32 variant for "
                "single-precision fields).");
        }
    };

    /**
     * @enum FEMElementKind
     * @brief Finite element used by the fused stiffness operator. The element
     *        only affects the one-time precomputation of the G/V/D̄ matrices
     *        from its reference shape-function gradients (fem_element.hh); the
     *        per-call kernels are element-agnostic, so it is selected at
     *        construction rather than as a template parameter.
     */
    enum class FEMElementKind { P1, Q1 };

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
            const std::vector<Real> & grid_spacing,
            FEMElementKind element = FEMElementKind::Q1)
            : grid_spacing{grid_spacing} {
            if (static_cast<Index_t>(this->grid_spacing.size()) != Dim) {
                throw RuntimeError(std::to_string(Dim) +
                                   "D operator requires " + std::to_string(Dim) +
                                   "D grid spacing");
            }
            this->load_element(element);
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
            this->apply_impl<Real>(displacement, lambda, mu, 1.0, force, false);
        }
        void apply_increment(const TypedFieldBase<Real> & displacement,
                             const TypedFieldBase<Real> & lambda,
                             const TypedFieldBase<Real> & mu, Real alpha,
                             TypedFieldBase<Real> & force) const override {
            this->apply_impl<Real>(displacement, lambda, mu, alpha, force, true);
        }
        //! Single-precision (Real32) overloads.
        void apply(const TypedFieldBase<Real32> & displacement,
                   const TypedFieldBase<Real32> & lambda,
                   const TypedFieldBase<Real32> & mu,
                   TypedFieldBase<Real32> & force) const override {
            this->apply_impl<Real32>(displacement, lambda, mu, 1.0f, force,
                                     false);
        }
        void apply_increment(const TypedFieldBase<Real32> & displacement,
                             const TypedFieldBase<Real32> & lambda,
                             const TypedFieldBase<Real32> & mu, Real32 alpha,
                             TypedFieldBase<Real32> & force) const override {
            this->apply_impl<Real32>(displacement, lambda, mu, alpha, force,
                                     true);
        }

        // ---- Uniform-coefficient host interface ----
        //! force = K(λ, μ) @ displacement for spatially uniform Lamé scalars.
        //! Needs no material fields — the reference stiffness Kʳᵉᶠ of the
        //! Green's-function preconditioner is exactly this operator.
        void apply_uniform(const TypedFieldBase<Real> & displacement,
                           Real lambda, Real mu,
                           TypedFieldBase<Real> & force) const {
            this->apply_uniform_impl<Real>(displacement, lambda, mu, 1.0, force,
                                           false);
        }
        //! force += alpha * K(λ, μ) @ displacement, uniform λ, μ.
        void apply_uniform_increment(const TypedFieldBase<Real> & displacement,
                                     Real lambda, Real mu, Real alpha,
                                     TypedFieldBase<Real> & force) const {
            this->apply_uniform_impl<Real>(displacement, lambda, mu, alpha,
                                           force, true);
        }
        //! Single-precision (Real32) uniform-coefficient overloads.
        void apply_uniform(const TypedFieldBase<Real32> & displacement,
                           Real32 lambda, Real32 mu,
                           TypedFieldBase<Real32> & force) const {
            this->apply_uniform_impl<Real32>(displacement, lambda, mu, 1.0f,
                                             force, false);
        }
        void apply_uniform_increment(
            const TypedFieldBase<Real32> & displacement, Real32 lambda,
            Real32 mu, Real32 alpha, TypedFieldBase<Real32> & force) const {
            this->apply_uniform_impl<Real32>(displacement, lambda, mu, alpha,
                                             force, true);
        }

        // ---- Streaming homogenization helpers (host) ----
        //! Assemble force = Bᵀ C(λ, μ) E_macro (the divergence of the stress of
        //! a uniform macroscopic strain E_macro), with no global strain/stress
        //! field. The homogenization right-hand side is the negative of this.
        //! E_macro is the (symmetric) Dim×Dim macro strain, row-major.
        void apply_macro_rhs(const TypedFieldBase<Real> & lambda,
                             const TypedFieldBase<Real> & mu,
                             const std::array<Real, Dim * Dim> & E_macro,
                             TypedFieldBase<Real> & force) const {
            this->apply_macro_rhs_impl<Real>(lambda, mu, E_macro, force);
        }
        //! Single-precision (Real32) overload of apply_macro_rhs (E_macro stays
        //! double).
        void apply_macro_rhs(const TypedFieldBase<Real32> & lambda,
                             const TypedFieldBase<Real32> & mu,
                             const std::array<Real, Dim * Dim> & E_macro,
                             TypedFieldBase<Real32> & force) const {
            this->apply_macro_rhs_impl<Real32>(lambda, mu, E_macro, force);
        }

        //! Volume-averaged stress integral ∫ σ dV over the local subdomain,
        //! where σ = C(λ, μ):(E_macro + sym(∇u)). Returns the Dim×Dim tensor
        //! (row-major) as a local, un-reduced volume integral: the caller sums
        //! across MPI ranks and divides by the total volume. Computes the
        //! element strain locally, so no global strain/stress field is needed.
        std::array<Real, Dim * Dim>
        average_stress(const TypedFieldBase<Real> & displacement,
                       const TypedFieldBase<Real> & lambda,
                       const TypedFieldBase<Real> & mu,
                       const std::array<Real, Dim * Dim> & E_macro) const {
            return this->average_stress_impl<Real>(displacement, lambda, mu,
                                                   E_macro);
        }
        //! Single-precision (Real32) overload of average_stress. The volume
        //! integral is still returned in double for the cross-rank reduction.
        std::array<Real, Dim * Dim>
        average_stress(const TypedFieldBase<Real32> & displacement,
                       const TypedFieldBase<Real32> & lambda,
                       const TypedFieldBase<Real32> & mu,
                       const std::array<Real, Dim * Dim> & E_macro) const {
            return this->average_stress_impl<Real32>(displacement, lambda, mu,
                                                     E_macro);
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        // ---- Device interface (host- and single-precision) ----
        // DSpace shorthand for the device-field overloads below.
        template <typename U>
        using DField = TypedFieldBase<U, DefaultDeviceSpace>;

        void apply(const DField<Real> & displacement,
                   const DField<Real> & lambda, const DField<Real> & mu,
                   DField<Real> & force) const {
            this->apply_impl<Real>(displacement, lambda, mu, 1.0, force, false);
        }
        void apply(const DField<Real32> & displacement,
                   const DField<Real32> & lambda, const DField<Real32> & mu,
                   DField<Real32> & force) const {
            this->apply_impl<Real32>(displacement, lambda, mu, 1.0f, force,
                                     false);
        }
        void apply_increment(const DField<Real> & displacement,
                             const DField<Real> & lambda,
                             const DField<Real> & mu, Real alpha,
                             DField<Real> & force) const {
            this->apply_impl<Real>(displacement, lambda, mu, alpha, force, true);
        }
        void apply_increment(const DField<Real32> & displacement,
                             const DField<Real32> & lambda,
                             const DField<Real32> & mu, Real32 alpha,
                             DField<Real32> & force) const {
            this->apply_impl<Real32>(displacement, lambda, mu, alpha, force,
                                     true);
        }

        // ---- Uniform-coefficient device interface ----
        void apply_uniform(const DField<Real> & displacement, Real lambda,
                           Real mu, DField<Real> & force) const {
            this->apply_uniform_impl<Real>(displacement, lambda, mu, 1.0, force,
                                           false);
        }
        void apply_uniform(const DField<Real32> & displacement, Real32 lambda,
                           Real32 mu, DField<Real32> & force) const {
            this->apply_uniform_impl<Real32>(displacement, lambda, mu, 1.0f,
                                             force, false);
        }
        void apply_uniform_increment(const DField<Real> & displacement,
                                     Real lambda, Real mu, Real alpha,
                                     DField<Real> & force) const {
            this->apply_uniform_impl<Real>(displacement, lambda, mu, alpha,
                                           force, true);
        }
        void apply_uniform_increment(const DField<Real32> & displacement,
                                     Real32 lambda, Real32 mu, Real32 alpha,
                                     DField<Real32> & force) const {
            this->apply_uniform_impl<Real32>(displacement, lambda, mu, alpha,
                                             force, true);
        }

        // ---- Streaming homogenization helpers (device) ----
        //! Device counterpart of apply_macro_rhs (see the host overload).
        void apply_macro_rhs(const DField<Real> & lambda,
                             const DField<Real> & mu,
                             const std::array<Real, Dim * Dim> & E_macro,
                             DField<Real> & force) const {
            this->apply_macro_rhs_impl<Real>(lambda, mu, E_macro, force);
        }
        void apply_macro_rhs(const DField<Real32> & lambda,
                             const DField<Real32> & mu,
                             const std::array<Real, Dim * Dim> & E_macro,
                             DField<Real32> & force) const {
            this->apply_macro_rhs_impl<Real32>(lambda, mu, E_macro, force);
        }

        //! Device counterpart of average_stress (see the host overload).
        std::array<Real, Dim * Dim>
        average_stress(const DField<Real> & displacement,
                       const DField<Real> & lambda, const DField<Real> & mu,
                       const std::array<Real, Dim * Dim> & E_macro) const {
            return this->average_stress_impl<Real>(displacement, lambda, mu,
                                                   E_macro);
        }
        std::array<Real, Dim * Dim>
        average_stress(const DField<Real32> & displacement,
                       const DField<Real32> & lambda,
                       const DField<Real32> & mu,
                       const std::array<Real, Dim * Dim> & E_macro) const {
            return this->average_stress_impl<Real32>(displacement, lambda, mu,
                                                     E_macro);
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

       public:
        //! Element-averaged gradient operator D̄[j*NB_NODES+n] = scale[j]
        //! Σ_q w_q B[q][j][n], so ḡ_ij = Σ_n D̄[j][n] u[n,i]. Used by
        //! average_stress.
        using GradientAverage = std::array<Real, Dim * NB_NODES>;
        const GradientAverage & get_Dbar() const { return Dbar_matrix; }

       private:
        std::vector<Real> grid_spacing;
        ElementMatrix G_matrix;  //!< 2μ coefficient geometry
        ElementMatrix V_matrix;  //!< λ coefficient geometry
        GradientAverage Dbar_matrix;  //!< element-averaged gradient operator

        //! Selected element's reference data, copied from the traits at
        //! construction (fem_element.hh). B is flattened [q][d][n]. Only
        //! precompute_matrices() consumes these, so runtime storage is free.
        std::vector<Real> elem_B;
        std::vector<Real> elem_Wfrac;
        Index_t elem_nb_quad{0};

        //! Copy the chosen element's B / Wfrac / NbQuad out of its traits.
        void load_element(FEMElementKind element) {
            auto load = [this](auto trait) {
                using T = decltype(trait);
                static_assert(T::SpatialDim == Dim,
                              "element/operator dimension mismatch");
                this->elem_nb_quad = T::NbQuad;
                this->elem_Wfrac.assign(T::Wfrac, T::Wfrac + T::NbQuad);
                this->elem_B.resize(T::NbQuad * Dim * NB_NODES);
                for (Index_t q = 0; q < T::NbQuad; ++q) {
                    for (Index_t d = 0; d < Dim; ++d) {
                        for (Index_t n = 0; n < NB_NODES; ++n) {
                            this->elem_B[(q * Dim + d) * NB_NODES + n] =
                                T::B[q][d][n];
                        }
                    }
                }
            };
            if constexpr (Dim == 2) {
                if (element == FEMElementKind::Q1) {
                    load(Q1Quad2D{});
                } else {
                    load(P1Tri2D{});
                }
            } else {
                if (element == FEMElementKind::Q1) {
                    load(Q1Hex3D{});
                } else {
                    load(P1Tet3D{});
                }
            }
        }

        //! Compute the geometry matrices G, V and Dbar from the selected
        //! element's reference data (dimension-specific Voigt assembly; explicit
        //! specialization in isotropic_stiffness_{2,3}d.cc).
        void precompute_matrices();

        //! Build the constant per-element vectors Gu = G u* and Vu = V u* of
        //! the affine displacement u* with ∇u* = E_macro (used by macro RHS).
        void macro_rhs_vectors(const std::array<Real, Dim * Dim> & E_macro,
                               ElementMatrix & Gu, ElementMatrix & Vu) const {
            std::array<Real, NB_ELEMENT_DOFS> u_star{};
            constexpr auto NODE_OFFSET = node_offset_table();
            for (Index_t n = 0; n < NB_NODES; ++n) {
                for (Index_t i = 0; i < Dim; ++i) {
                    Real ui = 0.0;
                    for (Index_t j = 0; j < Dim; ++j) {
                        ui += E_macro[i * Dim + j] *
                              static_cast<Real>(NODE_OFFSET[n][j]) *
                              this->grid_spacing[j];
                    }
                    u_star[n * Dim + i] = ui;
                }
            }
            for (Index_t r = 0; r < NB_ELEMENT_DOFS; ++r) {
                Real gsum = 0.0, vsum = 0.0;
                for (Index_t c = 0; c < NB_ELEMENT_DOFS; ++c) {
                    gsum += G_matrix[r * NB_ELEMENT_DOFS + c] * u_star[c];
                    vsum += V_matrix[r * NB_ELEMENT_DOFS + c] * u_star[c];
                }
                Gu[r] = gsum;
                Vu[r] = vsum;
            }
        }

        //! Node offsets within an element [node][dim] (binary indexing); shared
        //! by the affine-displacement construction above.
        static constexpr std::array<std::array<Index_t, Dim>, NB_NODES>
        node_offset_table() {
            std::array<std::array<Index_t, Dim>, NB_NODES> tbl{};
            for (Index_t n = 0; n < NB_NODES; ++n) {
                for (Index_t d = 0; d < Dim; ++d) {
                    tbl[n][d] = (n >> d) & 1;
                }
            }
            return tbl;
        }

        //! Host apply with optional increment (dimension-specific; explicit
        //! member-template specialization in isotropic_stiffness_{2,3}d.cc,
        //! instantiated for T ∈ {Real, Real32}).
        template <typename T>
        void apply_impl(const TypedFieldBase<T> & displacement,
                        const TypedFieldBase<T> & lambda,
                        const TypedFieldBase<T> & mu, T alpha,
                        TypedFieldBase<T> & force, bool increment) const;

        //! Host apply for uniform Lamé scalars (dimension-specific; explicit
        //! specialization in isotropic_stiffness_{2,3}d.cc).
        template <typename T>
        void apply_uniform_impl(const TypedFieldBase<T> & displacement,
                                T lambda, T mu, T alpha,
                                TypedFieldBase<T> & force,
                                bool increment) const;

        //! Host macro-RHS / stress-average (dimension-specific; explicit
        //! specialization in isotropic_stiffness_{2,3}d.cc). E_macro stays in
        //! double; the kernel-side per-element vectors are converted to T.
        template <typename T>
        void apply_macro_rhs_impl(const TypedFieldBase<T> & lambda,
                                  const TypedFieldBase<T> & mu,
                                  const std::array<Real, Dim * Dim> & E_macro,
                                  TypedFieldBase<T> & force) const;
        template <typename T>
        std::array<Real, Dim * Dim>
        average_stress_impl(const TypedFieldBase<T> & displacement,
                            const TypedFieldBase<T> & lambda,
                            const TypedFieldBase<T> & mu,
                            const std::array<Real, Dim * Dim> & E_macro) const;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        // The geometry matrices (G/V/Dbar/Gu/Vu/E_macro) live in double
        // __constant__ memory and stay double; the kernels cast them to T at
        // the point of load so the per-element arithmetic runs in T. Hence the
        // device impls are member templates on T like their host counterparts.

        //! Device apply with optional increment (dimension-specific; explicit
        //! specialization in isotropic_stiffness_gpu.cc).
        template <typename T>
        void apply_impl(
            const TypedFieldBase<T, DefaultDeviceSpace> & displacement,
            const TypedFieldBase<T, DefaultDeviceSpace> & lambda,
            const TypedFieldBase<T, DefaultDeviceSpace> & mu, T alpha,
            TypedFieldBase<T, DefaultDeviceSpace> & force,
            bool increment) const;

        //! Device apply for uniform Lamé scalars (explicit specialization in
        //! isotropic_stiffness_gpu.cc).
        template <typename T>
        void apply_uniform_impl(
            const TypedFieldBase<T, DefaultDeviceSpace> & displacement,
            T lambda, T mu, T alpha,
            TypedFieldBase<T, DefaultDeviceSpace> & force,
            bool increment) const;

        //! Device macro-RHS / stress-average (explicit specialization in
        //! isotropic_stiffness_gpu.cc).
        template <typename T>
        void apply_macro_rhs_impl(
            const TypedFieldBase<T, DefaultDeviceSpace> & lambda,
            const TypedFieldBase<T, DefaultDeviceSpace> & mu,
            const std::array<Real, Dim * Dim> & E_macro,
            TypedFieldBase<T, DefaultDeviceSpace> & force) const;
        template <typename T>
        std::array<Real, Dim * Dim> average_stress_impl(
            const TypedFieldBase<T, DefaultDeviceSpace> & displacement,
            const TypedFieldBase<T, DefaultDeviceSpace> & lambda,
            const TypedFieldBase<T, DefaultDeviceSpace> & mu,
            const std::array<Real, Dim * Dim> & E_macro) const;
#endif
    };

    // Specialization declarations: the bodies live in the .cc files, so every
    // translation unit that instantiates the operator must see that these
    // members are specialized elsewhere (and not implicitly instantiated from
    // the — intentionally undefined — primary template). precompute_matrices is
    // a plain member specialization; the apply/transpose impls are member
    // templates on the scalar type T (instantiated for Real and Real32 in the
    // .cc files). One declaration per (Dim, space) covers both precisions.
    // The optional trailing argument is the memory space; when present it is
    // spliced in as `TypedFieldBase<T, DefaultDeviceSpace>` via __VA_OPT__.
#define MUGRID_TFB(...) TypedFieldBase<T __VA_OPT__(, ) __VA_ARGS__>
#define MUGRID_DECLARE_STIFFNESS_IMPLS(D, ...)                                 \
    template <>                                                                \
    template <typename T>                                                      \
    void IsotropicStiffnessOperator<D>::apply_impl(                            \
        const MUGRID_TFB(__VA_ARGS__) &, const MUGRID_TFB(__VA_ARGS__) &,      \
        const MUGRID_TFB(__VA_ARGS__) &, T, MUGRID_TFB(__VA_ARGS__) &, bool)   \
        const;                                                                 \
    template <>                                                                \
    template <typename T>                                                      \
    void IsotropicStiffnessOperator<D>::apply_uniform_impl(                    \
        const MUGRID_TFB(__VA_ARGS__) &, T, T, T, MUGRID_TFB(__VA_ARGS__) &,   \
        bool) const;                                                           \
    template <>                                                                \
    template <typename T>                                                      \
    void IsotropicStiffnessOperator<D>::apply_macro_rhs_impl(                  \
        const MUGRID_TFB(__VA_ARGS__) &, const MUGRID_TFB(__VA_ARGS__) &,      \
        const std::array<Real, D * D> &, MUGRID_TFB(__VA_ARGS__) &) const;     \
    template <>                                                                \
    template <typename T>                                                      \
    std::array<Real, D * D>                                                    \
    IsotropicStiffnessOperator<D>::average_stress_impl(                        \
        const MUGRID_TFB(__VA_ARGS__) &, const MUGRID_TFB(__VA_ARGS__) &,      \
        const MUGRID_TFB(__VA_ARGS__) &, const std::array<Real, D * D> &)      \
        const;

    template <>
    void IsotropicStiffnessOperator<2>::precompute_matrices();
    template <>
    void IsotropicStiffnessOperator<3>::precompute_matrices();

    MUGRID_DECLARE_STIFFNESS_IMPLS(2)
    MUGRID_DECLARE_STIFFNESS_IMPLS(3)

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    // Device impls are member templates on T too (the double __constant__
    // geometry is cast to T inside the kernels). One declaration per dim covers
    // both precisions.
    MUGRID_DECLARE_STIFFNESS_IMPLS(2, DefaultDeviceSpace)
    MUGRID_DECLARE_STIFFNESS_IMPLS(3, DefaultDeviceSpace)
#endif
#undef MUGRID_DECLARE_STIFFNESS_IMPLS
#undef MUGRID_TFB

    //! 2D fused isotropic stiffness operator. Preserves the historical name.
    using IsotropicStiffnessOperator2D = IsotropicStiffnessOperator<2>;
    //! 3D fused isotropic stiffness operator. Preserves the historical name.
    using IsotropicStiffnessOperator3D = IsotropicStiffnessOperator<3>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_SOLIDS_ISOTROPIC_STIFFNESS_HH_
