/**
 * @file   isotropic_stiffness_operator.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   31 Dec 2025
 *
 * @brief  Fused elliptic operator for isotropic linear elastic materials
 *
 * This class provides optimized implementations of the stiffness operator
 * K = B^T C B for isotropic linear elastic materials. Instead of storing
 * the full K matrix, it exploits the isotropic structure:
 *
 *   K = 2μ G + λ V
 *
 * where:
 * - G = Σ_q w_q B_q^T B_q is a geometry-only matrix (same for all voxels)
 * - V = Σ_q w_q (B_q^T m)(m^T B_q) is a volumetric coupling matrix
 * - λ, μ are Lamé parameters (can vary spatially)
 * - m = [1, 1, 1, 0, 0, 0]^T (Voigt notation trace selector)
 *
 * This reduces memory from O(N × 24²) for full K storage to O(N × 2) for
 * spatially-varying isotropic materials, plus O(1) for the shared G and V.
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

#ifndef SRC_LIBMUGRID_OPERATORS_SOLIDS_ISOTROPIC_STIFFNESS_2D_HH_
#define SRC_LIBMUGRID_OPERATORS_SOLIDS_ISOTROPIC_STIFFNESS_2D_HH_

#include "core/types.hh"
#include "field/field_typed.hh"
#include "memory/memory_space.hh"

#include <array>
#include <vector>

namespace muGrid {

    // Forward declaration
    class GlobalFieldCollection;

    /**
     * @class IsotropicStiffnessOperator2D
     * @brief Fused stiffness operator for 2D isotropic linear elastic
     * materials.
     *
     * Computes K @ u = B^T C B @ u for 2D linear triangular elements.
     * Uses the decomposition K = 2μ G + λ V where G and V are precomputed
     * geometry matrices shared by all pixels.
     *
     * Memory layout:
     * - Displacement field: [3, nx, ny] (3 DOFs per node: ux, uy for 2D +
     * ghost)
     * - Material field: [2, nx-1, ny-1] (λ, μ per pixel)
     * - Force field: [3, nx, ny] (same as displacement)
     */
    class IsotropicStiffnessOperator2D {
       public:
        //! Number of nodes per pixel (4 corners)
        static constexpr Index_t NB_NODES = 4;
        //! Number of DOFs per node (2 for 2D)
        static constexpr Index_t NB_DOFS_PER_NODE = 2;
        //! Total DOFs per element
        static constexpr Index_t NB_ELEMENT_DOFS = NB_NODES * NB_DOFS_PER_NODE;
        //! Number of quadrature points (2 triangles)
        static constexpr Index_t NB_QUAD = 2;

        /**
         * @brief Construct the operator with given grid spacing.
         * @param grid_spacing Grid spacing [hx, hy]
         */
        explicit IsotropicStiffnessOperator2D(
            const std::vector<Real> & grid_spacing);

        //! Default constructor is deleted
        IsotropicStiffnessOperator2D() = delete;

        //! Destructor
        ~IsotropicStiffnessOperator2D() = default;

        /**
         * @brief Apply the stiffness operator: force = K @ displacement
         *
         * @param displacement Input displacement field [2, nx, ny]
         * @param lambda Lamé first parameter field [nx-1, ny-1]
         * @param mu Lamé second parameter (shear modulus) field [nx-1, ny-1]
         * @param force Output force field [2, nx, ny]
         */
        void apply(const TypedFieldBase<Real> & displacement,
                   const TypedFieldBase<Real> & lambda,
                   const TypedFieldBase<Real> & mu,
                   TypedFieldBase<Real> & force) const;

        /**
         * @brief Apply with increment: force += alpha * K @ displacement
         */
        void apply_increment(const TypedFieldBase<Real> & displacement,
                             const TypedFieldBase<Real> & lambda,
                             const TypedFieldBase<Real> & mu, Real alpha,
                             TypedFieldBase<Real> & force) const;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        void
        apply(const TypedFieldBase<Real, DefaultDeviceSpace> & displacement,
              const TypedFieldBase<Real, DefaultDeviceSpace> & lambda,
              const TypedFieldBase<Real, DefaultDeviceSpace> & mu,
              TypedFieldBase<Real, DefaultDeviceSpace> & force) const;

        void apply_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & displacement,
            const TypedFieldBase<Real, DefaultDeviceSpace> & lambda,
            const TypedFieldBase<Real, DefaultDeviceSpace> & mu, Real alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & force) const;
#endif

        /**
         * @brief Get the precomputed G matrix (geometry-only).
         * @return G matrix as flat array [NB_ELEMENT_DOFS × NB_ELEMENT_DOFS]
         */
        const std::array<Real, NB_ELEMENT_DOFS * NB_ELEMENT_DOFS> &
        get_G() const {
            return G_matrix;
        }

        /**
         * @brief Get the precomputed V matrix (volumetric coupling).
         * @return V matrix as flat array [NB_ELEMENT_DOFS × NB_ELEMENT_DOFS]
         */
        const std::array<Real, NB_ELEMENT_DOFS * NB_ELEMENT_DOFS> &
        get_V() const {
            return V_matrix;
        }

       private:
        std::vector<Real> grid_spacing;

        //! Precomputed G = Σ_q w_q B_q^T B_q (shear stiffness)
        std::array<Real, NB_ELEMENT_DOFS * NB_ELEMENT_DOFS> G_matrix;

        //! Precomputed V = Σ_q w_q (B_q^T m)(m^T B_q) (volumetric stiffness)
        std::array<Real, NB_ELEMENT_DOFS * NB_ELEMENT_DOFS> V_matrix;

        //! Compute the geometry matrices G and V
        void precompute_matrices();

        //! Internal implementation
        void apply_impl(const TypedFieldBase<Real> & displacement,
                        const TypedFieldBase<Real> & lambda,
                        const TypedFieldBase<Real> & mu, Real alpha,
                        TypedFieldBase<Real> & force, bool increment) const;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        void apply_impl(
            const TypedFieldBase<Real, DefaultDeviceSpace> & displacement,
            const TypedFieldBase<Real, DefaultDeviceSpace> & lambda,
            const TypedFieldBase<Real, DefaultDeviceSpace> & mu, Real alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & force,
            bool increment) const;
#endif
    };

    // Kernel declarations
    namespace isotropic_stiffness_kernels {

        /**
         * @brief 2D host kernel for isotropic stiffness operator.
         *
         * Uses gather pattern: iterates over interior nodes, gathers from
         * neighboring elements via ghost cells. Ghost communication handles
         * periodicity and MPI boundaries.
         *
         * @param nnx, nny Number of interior nodes
         * @param nelx, nely Number of elements
         */
        void isotropic_stiffness_2d_host(
            const Real * MUGRID_RESTRICT displacement,
            const Real * MUGRID_RESTRICT lambda,
            const Real * MUGRID_RESTRICT mu, Real * MUGRID_RESTRICT force,
            Index_t nnx, Index_t nny, Index_t nelx, Index_t nely,
            Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_d,
            Index_t mat_stride_x, Index_t mat_stride_y, Index_t force_stride_x,
            Index_t force_stride_y, Index_t force_stride_d, const Real * G,
            const Real * V, Real alpha, bool increment);

#if defined(MUGRID_ENABLE_CUDA)
        void isotropic_stiffness_2d_cuda(
            const Real * displacement, const Real * lambda, const Real * mu,
            Real * force, Index_t nnx, Index_t nny, Index_t nelx, Index_t nely,
            Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_d,
            Index_t mat_stride_x, Index_t mat_stride_y, Index_t force_stride_x,
            Index_t force_stride_y, Index_t force_stride_d, const Real * G,
            const Real * V, Real alpha, bool increment);
#endif

#if defined(MUGRID_ENABLE_HIP)
        void isotropic_stiffness_2d_hip(
            const Real * displacement, const Real * lambda, const Real * mu,
            Real * force, Index_t nnx, Index_t nny, Index_t nelx, Index_t nely,
            Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_d,
            Index_t mat_stride_x, Index_t mat_stride_y, Index_t force_stride_x,
            Index_t force_stride_y, Index_t force_stride_d, const Real * G,
            const Real * V, Real alpha, bool increment);
#endif

    }  // namespace isotropic_stiffness_kernels

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_SOLIDS_ISOTROPIC_STIFFNESS_2D_HH_
