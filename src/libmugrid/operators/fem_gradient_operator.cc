/**
 * @file   fem_gradient_operator.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Dec 2024
 *
 * @brief  Host implementation of hard-coded linear FEM gradient operator
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

#include "fem_gradient_operator.hh"
#include "collection/field_collection_global.hh"
#include "core/exception.hh"

#if defined(MUGRID_ENABLE_CUDA)
    #include <cuda_runtime.h>
#elif defined(MUGRID_ENABLE_HIP)
    #include <hip/hip_runtime.h>
#endif

namespace muGrid {

    // =========================================================================
    // 3D Shape Function Gradients Definition
    // =========================================================================
    // These are computed for the 5-tet Kuhn triangulation of a unit cube.
    // For each tetrahedron, the gradient of the shape function N_i at node i
    // is constant within the tetrahedron.
    //
    // For a tetrahedron with vertices v0, v1, v2, v3:
    // grad(N0) points from face (v1,v2,v3) towards v0, scaled by 1/(3*V)
    // where V is the tetrahedron volume.
    //
    // For a unit cube, each tetrahedron has volume 1/6.
    // The shape function gradients are ±1 in each direction, depending on
    // which face of the tetrahedron the node is opposite to.
    // =========================================================================

    namespace fem_gradient_kernels {

        // 3D shape function gradients [dim][quad][node]
        // Pre-computed for Kuhn triangulation
        const Real B_3D_REF[DIM_3D][NB_QUAD_3D][NB_NODES_3D] = {
            // d/dx gradients (scaled by 1/hx at runtime)
            {
                // Tet 0: nodes 0,1,2,4 → active nodes contribute
                {-1.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0},
                // Tet 1: nodes 1,2,4,5
                { 0.0, -1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0},
                // Tet 2: nodes 2,4,5,6
                { 0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  0.0},
                // Tet 3: nodes 1,2,3,5
                { 0.0, -1.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0},
                // Tet 4: nodes 2,3,5,7
                { 0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  1.0}
            },
            // d/dy gradients (scaled by 1/hy at runtime)
            {
                // Tet 0: nodes 0,1,2,4
                {-1.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0},
                // Tet 1: nodes 1,2,4,5
                { 0.0,  0.0,  1.0,  0.0, -1.0,  0.0,  0.0,  0.0},
                // Tet 2: nodes 2,4,5,6
                { 0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  1.0,  0.0},
                // Tet 3: nodes 1,2,3,5
                { 0.0,  0.0, -1.0,  1.0,  0.0,  0.0,  0.0,  0.0},
                // Tet 4: nodes 2,3,5,7
                { 0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0,  1.0}
            },
            // d/dz gradients (scaled by 1/hz at runtime)
            {
                // Tet 0: nodes 0,1,2,4
                {-1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0},
                // Tet 1: nodes 1,2,4,5
                { 0.0, -1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0},
                // Tet 2: nodes 2,4,5,6
                { 0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  1.0,  0.0},
                // Tet 3: nodes 1,2,3,5
                { 0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  0.0},
                // Tet 4: nodes 2,3,5,7
                { 0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  1.0}
            }
        };

    }  // namespace fem_gradient_kernels

    // =========================================================================
    // FEMGradientOperator Implementation
    // =========================================================================

    FEMGradientOperator::FEMGradientOperator(Index_t spatial_dim,
                                              std::vector<Real> grid_spacing)
        : Parent{}, spatial_dim{spatial_dim}, grid_spacing{std::move(grid_spacing)} {
        if (spatial_dim != 2 && spatial_dim != 3) {
            throw RuntimeError("FEMGradientOperator only supports 2D and 3D grids");
        }
        // Default grid spacing is 1.0 in each direction
        if (this->grid_spacing.empty()) {
            this->grid_spacing.resize(spatial_dim, 1.0);
        }
        if (static_cast<Index_t>(this->grid_spacing.size()) != spatial_dim) {
            throw RuntimeError("Grid spacing must have " +
                std::to_string(spatial_dim) + " components");
        }
    }

    const GlobalFieldCollection& FEMGradientOperator::validate_fields(
        const Field &nodal_field,
        const Field &gradient_field,
        Index_t &nb_components) const {

        // Get field collections
        auto& nodal_collection = nodal_field.get_collection();
        auto& gradient_collection = gradient_field.get_collection();

        // Must be the same collection
        if (&nodal_collection != &gradient_collection) {
            throw RuntimeError("Nodal and gradient fields must belong to the "
                               "same field collection");
        }

        // Must be global field collection
        auto* global_fc = dynamic_cast<const GlobalFieldCollection*>(
            &nodal_collection);
        if (!global_fc) {
            throw RuntimeError("FEMGradientOperator requires GlobalFieldCollection");
        }

        // Check dimension matches
        if (global_fc->get_spatial_dim() != this->spatial_dim) {
            throw RuntimeError("Field collection dimension (" +
                std::to_string(global_fc->get_spatial_dim()) +
                ") does not match operator dimension (" +
                std::to_string(this->spatial_dim) + ")");
        }

        // Get and validate component counts
        // Output should have nb_operators * nb_nodal_components components
        Index_t nb_nodal_components = nodal_field.get_nb_components();
        Index_t nb_grad_components = gradient_field.get_nb_components();
        Index_t expected_grad_components = this->get_nb_operators() * nb_nodal_components;

        if (nb_grad_components != expected_grad_components) {
            std::stringstream err_msg;
            err_msg << "Component mismatch: Expected gradient field with "
                    << expected_grad_components << " components ("
                    << this->get_nb_operators() << " operators × "
                    << nb_nodal_components << " nodal components) but got "
                    << nb_grad_components << " components.";
            throw RuntimeError(err_msg.str());
        }

        nb_components = nb_nodal_components;
        return *global_fc;
    }

    std::vector<Real> FEMGradientOperator::get_quadrature_weights() const {
        if (this->spatial_dim == 2) {
            // Each triangle has area = 0.5 * hx * hy
            // Total pixel area = hx * hy
            // Weight = triangle_area / pixel_area = 0.5
            Real pixel_area = this->grid_spacing[0] * this->grid_spacing[1];
            return {0.5 * pixel_area, 0.5 * pixel_area};
        } else {
            // 5 tetrahedra, each with volume = 1/5 of voxel
            Real voxel_volume = this->grid_spacing[0] * this->grid_spacing[1] *
                                this->grid_spacing[2];
            return {0.2 * voxel_volume, 0.2 * voxel_volume, 0.2 * voxel_volume,
                    0.2 * voxel_volume, 0.2 * voxel_volume};
        }
    }

    void FEMGradientOperator::apply_impl(const TypedFieldBase<Real> &nodal_field,
                                          TypedFieldBase<Real> &gradient_field,
                                          Real alpha,
                                          bool increment) const {
        Index_t nb_components;
        const auto& collection = this->validate_fields(nodal_field, gradient_field,
                                                        nb_components);

        // Get grid dimensions (with ghosts)
        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        // Get raw data pointers
        const Real* nodal = nodal_field.data();
        Real* gradient = gradient_field.data();

        // Grid spacing
        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];

        if (this->spatial_dim == 2) {
            Index_t nx = nb_grid_pts[0];
            Index_t ny = nb_grid_pts[1];

            // For continuous FEM, the nodal field has shape [components, sub_pts, nx, ny]
            // Field layout (column-major/AoS): components vary FASTEST, then sub_pts,
            // then spatial dimensions.
            Index_t nb_sub = this->get_nb_nodal_pts();  // 1 (one scalar per grid point)
            Index_t nodal_stride_c = 1;  // components are innermost
            Index_t nodal_stride_n = nb_components;
            Index_t nodal_stride_x = nb_components * nb_sub;
            Index_t nodal_stride_y = nb_components * nb_sub * nx;

            // For gradient field [components, operators, nb_quad, x, y]:
            // Column-major (AoS): components vary fastest, then operators, then quad,
            // then spatial dimensions.
            Index_t dim = this->spatial_dim;  // 2 (operators = derivative directions)
            Index_t nb_quad = this->get_nb_quad_pts();  // 2
            Index_t grad_stride_c = 1;  // components are innermost
            Index_t grad_stride_d = nb_components;  // operators after components
            Index_t grad_stride_q = nb_components * dim;  // quad after components*operators
            Index_t grad_stride_x = nb_components * dim * nb_quad;
            Index_t grad_stride_y = nb_components * dim * nb_quad * nx;

            // Process each component independently
            // Each component's data is interleaved, offset by just comp (not comp * stride)
            for (Index_t comp = 0; comp < nb_components; ++comp) {
                const Real* nodal_comp = nodal + comp * nodal_stride_c;
                Real* gradient_comp = gradient + comp * grad_stride_c;

                fem_gradient_kernels::fem_gradient_2d_host(
                    nodal_comp, gradient_comp, nx, ny,
                    nodal_stride_x, nodal_stride_y, nodal_stride_n,
                    grad_stride_x, grad_stride_y, grad_stride_q, grad_stride_d,
                    hx, hy, alpha, increment);
            }
        } else {
            Index_t nx = nb_grid_pts[0];
            Index_t ny = nb_grid_pts[1];
            Index_t nz = nb_grid_pts[2];
            Real hz = this->grid_spacing[2];

            // Nodal field strides [components, sub_pts, x, y, z]
            // Column-major (AoS): components vary fastest
            Index_t nb_sub = this->get_nb_nodal_pts();  // 1
            Index_t nodal_stride_c = 1;
            Index_t nodal_stride_n = nb_components;
            Index_t nodal_stride_x = nb_components * nb_sub;
            Index_t nodal_stride_y = nb_components * nb_sub * nx;
            Index_t nodal_stride_z = nb_components * nb_sub * nx * ny;

            // Gradient field strides [components, operators, nb_quad, x, y, z]
            // Column-major (AoS): components vary fastest
            Index_t dim = this->spatial_dim;  // 3
            Index_t nb_quad = this->get_nb_quad_pts();  // 5
            Index_t grad_stride_c = 1;
            Index_t grad_stride_d = nb_components;
            Index_t grad_stride_q = nb_components * dim;
            Index_t grad_stride_x = nb_components * dim * nb_quad;
            Index_t grad_stride_y = nb_components * dim * nb_quad * nx;
            Index_t grad_stride_z = nb_components * dim * nb_quad * nx * ny;

            // Process each component independently
            for (Index_t comp = 0; comp < nb_components; ++comp) {
                const Real* nodal_comp = nodal + comp * nodal_stride_c;
                Real* gradient_comp = gradient + comp * grad_stride_c;

                fem_gradient_kernels::fem_gradient_3d_host(
                    nodal_comp, gradient_comp, nx, ny, nz,
                    nodal_stride_x, nodal_stride_y, nodal_stride_z, nodal_stride_n,
                    grad_stride_x, grad_stride_y, grad_stride_z,
                    grad_stride_q, grad_stride_d,
                    hx, hy, hz, alpha, increment);
            }
        }
    }

    void FEMGradientOperator::transpose_impl(
        const TypedFieldBase<Real> &gradient_field,
        TypedFieldBase<Real> &nodal_field,
        Real alpha,
        bool increment,
        const std::vector<Real> &weights) const {

        Index_t nb_components;
        const auto& collection = this->validate_fields(nodal_field, gradient_field,
                                                        nb_components);

        // Get grid dimensions
        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        // Get raw data pointers
        const Real* gradient = gradient_field.data();
        Real* nodal = nodal_field.data();

        // Grid spacing
        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];

        // Get quadrature weights
        std::vector<Real> quad_weights = weights.empty() ?
            this->get_quadrature_weights() : weights;

        if (this->spatial_dim == 2) {
            Index_t nx = nb_grid_pts[0];
            Index_t ny = nb_grid_pts[1];

            // Nodal field strides [components, sub_pts, x, y]
            // Column-major (AoS): components vary fastest
            Index_t nb_nodes = this->get_nb_nodal_pts();
            Index_t nodal_stride_c = 1;
            Index_t nodal_stride_n = nb_components;
            Index_t nodal_stride_x = nb_components * nb_nodes;
            Index_t nodal_stride_y = nb_components * nb_nodes * nx;

            // Gradient field strides [components, operators, nb_quad, x, y]
            // Column-major (AoS): components vary fastest
            Index_t dim = this->spatial_dim;
            Index_t nb_quad = this->get_nb_quad_pts();
            Index_t grad_stride_c = 1;
            Index_t grad_stride_d = nb_components;
            Index_t grad_stride_q = nb_components * dim;
            Index_t grad_stride_x = nb_components * dim * nb_quad;
            Index_t grad_stride_y = nb_components * dim * nb_quad * nx;

            // Process each component independently
            for (Index_t comp = 0; comp < nb_components; ++comp) {
                const Real* gradient_comp = gradient + comp * grad_stride_c;
                Real* nodal_comp = nodal + comp * nodal_stride_c;

                fem_gradient_kernels::fem_divergence_2d_host(
                    gradient_comp, nodal_comp, nx, ny,
                    grad_stride_x, grad_stride_y, grad_stride_q, grad_stride_d,
                    nodal_stride_x, nodal_stride_y, nodal_stride_n,
                    hx, hy, quad_weights.data(), alpha, increment);
            }
        } else {
            Index_t nx = nb_grid_pts[0];
            Index_t ny = nb_grid_pts[1];
            Index_t nz = nb_grid_pts[2];
            Real hz = this->grid_spacing[2];

            // Nodal field strides [components, sub_pts, x, y, z]
            // Column-major (AoS): components vary fastest
            Index_t nb_nodes = this->get_nb_nodal_pts();
            Index_t nodal_stride_c = 1;
            Index_t nodal_stride_n = nb_components;
            Index_t nodal_stride_x = nb_components * nb_nodes;
            Index_t nodal_stride_y = nb_components * nb_nodes * nx;
            Index_t nodal_stride_z = nb_components * nb_nodes * nx * ny;

            // Gradient field strides [components, operators, nb_quad, x, y, z]
            // Column-major (AoS): components vary fastest
            Index_t dim = this->spatial_dim;
            Index_t nb_quad = this->get_nb_quad_pts();
            Index_t grad_stride_c = 1;
            Index_t grad_stride_d = nb_components;
            Index_t grad_stride_q = nb_components * dim;
            Index_t grad_stride_x = nb_components * dim * nb_quad;
            Index_t grad_stride_y = nb_components * dim * nb_quad * nx;
            Index_t grad_stride_z = nb_components * dim * nb_quad * nx * ny;

            // Process each component independently
            for (Index_t comp = 0; comp < nb_components; ++comp) {
                const Real* gradient_comp = gradient + comp * grad_stride_c;
                Real* nodal_comp = nodal + comp * nodal_stride_c;

                fem_gradient_kernels::fem_divergence_3d_host(
                    gradient_comp, nodal_comp, nx, ny, nz,
                    grad_stride_x, grad_stride_y, grad_stride_z,
                    grad_stride_q, grad_stride_d,
                    nodal_stride_x, nodal_stride_y, nodal_stride_z, nodal_stride_n,
                    hx, hy, hz, quad_weights.data(), alpha, increment);
            }
        }
    }

    void FEMGradientOperator::apply(const TypedFieldBase<Real> &nodal_field,
                                     TypedFieldBase<Real> &gradient_field) const {
        this->apply_impl(nodal_field, gradient_field, 1.0, false);
    }

    void FEMGradientOperator::apply_increment(
        const TypedFieldBase<Real> &nodal_field,
        const Real &alpha,
        TypedFieldBase<Real> &gradient_field) const {
        this->apply_impl(nodal_field, gradient_field, alpha, true);
    }

    void FEMGradientOperator::transpose(const TypedFieldBase<Real> &gradient_field,
                                         TypedFieldBase<Real> &nodal_field,
                                         const std::vector<Real> &weights) const {
        this->transpose_impl(gradient_field, nodal_field, 1.0, false, weights);
    }

    void FEMGradientOperator::transpose_increment(
        const TypedFieldBase<Real> &gradient_field,
        const Real &alpha,
        TypedFieldBase<Real> &nodal_field,
        const std::vector<Real> &weights) const {
        this->transpose_impl(gradient_field, nodal_field, alpha, true, weights);
    }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    void FEMGradientOperator::apply_impl(
        const TypedFieldBase<Real, DefaultDeviceSpace> &nodal_field,
        TypedFieldBase<Real, DefaultDeviceSpace> &gradient_field,
        Real alpha,
        bool increment) const {
        Index_t nb_components;
        const auto& collection = this->validate_fields(nodal_field, gradient_field,
                                                        nb_components);

        // Get grid dimensions (with ghosts)
        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        // Get raw device data pointers via view()
        const Real* nodal = nodal_field.view().data();
        Real* gradient = gradient_field.view().data();

        // Grid spacing
        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];

        // Device uses StructureOfArrays (SoA) layout:
        // - spatial indices are fastest varying
        // - component/quadrature indices are slowest varying
        if (this->spatial_dim == 2) {
            Index_t nx = nb_grid_pts[0];
            Index_t ny = nb_grid_pts[1];

            // For SoA: nodal field [components, sub_pts, x, y]
            // Memory: x fastest, then y, then sub_pts, then components
            Index_t nodal_stride_x = 1;
            Index_t nodal_stride_y = nx;
            Index_t nodal_stride_n = nx * ny;
            Index_t nodal_stride_c = nx * ny * this->get_nb_nodal_pts();

            // For SoA: gradient field [components, operators, nb_quad, x, y]
            // Memory: x fastest, then y, then q, then operators, then components
            Index_t dim = this->spatial_dim;
            Index_t nb_quad = this->get_nb_quad_pts();
            Index_t grad_stride_x = 1;
            Index_t grad_stride_y = nx;
            Index_t grad_stride_q = nx * ny;
            Index_t grad_stride_d = nx * ny * nb_quad;
            Index_t grad_stride_c = nx * ny * nb_quad * dim;

            // Process each component independently
            for (Index_t comp = 0; comp < nb_components; ++comp) {
                const Real* nodal_comp = nodal + comp * nodal_stride_c;
                Real* gradient_comp = gradient + comp * grad_stride_c;

#if defined(MUGRID_ENABLE_CUDA)
                fem_gradient_kernels::fem_gradient_2d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
                fem_gradient_kernels::fem_gradient_2d_hip(
#endif
                    nodal_comp, gradient_comp, nx, ny,
                    nodal_stride_x, nodal_stride_y, nodal_stride_n,
                    grad_stride_x, grad_stride_y, grad_stride_q, grad_stride_d,
                    hx, hy, alpha, increment);
            }
        } else {
            Index_t nx = nb_grid_pts[0];
            Index_t ny = nb_grid_pts[1];
            Index_t nz = nb_grid_pts[2];
            Real hz = this->grid_spacing[2];

            // For SoA: nodal field [components, sub_pts, x, y, z]
            Index_t nodal_stride_x = 1;
            Index_t nodal_stride_y = nx;
            Index_t nodal_stride_z = nx * ny;
            Index_t nodal_stride_n = nx * ny * nz;
            Index_t nodal_stride_c = nx * ny * nz * this->get_nb_nodal_pts();

            // For SoA: gradient field [components, operators, nb_quad, x, y, z]
            Index_t dim = this->spatial_dim;
            Index_t nb_quad = this->get_nb_quad_pts();
            Index_t grad_stride_x = 1;
            Index_t grad_stride_y = nx;
            Index_t grad_stride_z = nx * ny;
            Index_t grad_stride_q = nx * ny * nz;
            Index_t grad_stride_d = nx * ny * nz * nb_quad;
            Index_t grad_stride_c = nx * ny * nz * nb_quad * dim;

            // Process each component independently
            for (Index_t comp = 0; comp < nb_components; ++comp) {
                const Real* nodal_comp = nodal + comp * nodal_stride_c;
                Real* gradient_comp = gradient + comp * grad_stride_c;

#if defined(MUGRID_ENABLE_CUDA)
                fem_gradient_kernels::fem_gradient_3d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
                fem_gradient_kernels::fem_gradient_3d_hip(
#endif
                    nodal_comp, gradient_comp, nx, ny, nz,
                    nodal_stride_x, nodal_stride_y, nodal_stride_z, nodal_stride_n,
                    grad_stride_x, grad_stride_y, grad_stride_z,
                    grad_stride_q, grad_stride_d,
                    hx, hy, hz, alpha, increment);
            }
        }
    }

    void FEMGradientOperator::transpose_impl(
        const TypedFieldBase<Real, DefaultDeviceSpace> &gradient_field,
        TypedFieldBase<Real, DefaultDeviceSpace> &nodal_field,
        Real alpha,
        bool increment,
        const std::vector<Real> &weights) const {
        Index_t nb_components;
        const auto& collection = this->validate_fields(nodal_field, gradient_field,
                                                        nb_components);

        // Get grid dimensions
        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        // Get raw device data pointers via view()
        const Real* gradient = gradient_field.view().data();
        Real* nodal = nodal_field.view().data();

        // Grid spacing
        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];

        // Get quadrature weights (host memory)
        std::vector<Real> quad_weights = weights.empty() ?
            this->get_quadrature_weights() : weights;

        // Device uses StructureOfArrays (SoA) layout:
        // - spatial indices are fastest varying
        // - component/quadrature indices are slowest varying
        if (this->spatial_dim == 2) {
            Index_t nx = nb_grid_pts[0];
            Index_t ny = nb_grid_pts[1];

            // For SoA: nodal field [components, sub_pts, x, y]
            Index_t nodal_stride_x = 1;
            Index_t nodal_stride_y = nx;
            Index_t nodal_stride_n = nx * ny;
            Index_t nodal_stride_c = nx * ny * this->get_nb_nodal_pts();

            // For SoA: gradient field [components, operators, nb_quad, x, y]
            Index_t dim = this->spatial_dim;
            Index_t nb_quad = this->get_nb_quad_pts();
            Index_t grad_stride_x = 1;
            Index_t grad_stride_y = nx;
            Index_t grad_stride_q = nx * ny;
            Index_t grad_stride_d = nx * ny * nb_quad;
            Index_t grad_stride_c = nx * ny * nb_quad * dim;

            // Process each component independently
            for (Index_t comp = 0; comp < nb_components; ++comp) {
                const Real* gradient_comp = gradient + comp * grad_stride_c;
                Real* nodal_comp = nodal + comp * nodal_stride_c;

#if defined(MUGRID_ENABLE_CUDA)
                fem_gradient_kernels::fem_divergence_2d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
                fem_gradient_kernels::fem_divergence_2d_hip(
#endif
                    gradient_comp, nodal_comp, nx, ny,
                    grad_stride_x, grad_stride_y, grad_stride_q, grad_stride_d,
                    nodal_stride_x, nodal_stride_y, nodal_stride_n,
                    hx, hy, quad_weights.data(), alpha, increment);
            }
        } else {
            Index_t nx = nb_grid_pts[0];
            Index_t ny = nb_grid_pts[1];
            Index_t nz = nb_grid_pts[2];
            Real hz = this->grid_spacing[2];

            // For SoA: nodal field [components, sub_pts, x, y, z]
            Index_t nodal_stride_x = 1;
            Index_t nodal_stride_y = nx;
            Index_t nodal_stride_z = nx * ny;
            Index_t nodal_stride_n = nx * ny * nz;
            Index_t nodal_stride_c = nx * ny * nz * this->get_nb_nodal_pts();

            // For SoA: gradient field [components, operators, nb_quad, x, y, z]
            Index_t dim = this->spatial_dim;
            Index_t nb_quad = this->get_nb_quad_pts();
            Index_t grad_stride_x = 1;
            Index_t grad_stride_y = nx;
            Index_t grad_stride_z = nx * ny;
            Index_t grad_stride_q = nx * ny * nz;
            Index_t grad_stride_d = nx * ny * nz * nb_quad;
            Index_t grad_stride_c = nx * ny * nz * nb_quad * dim;

            // Allocate device memory for weights (once, shared by all components)
            Real* d_quad_weights = nullptr;
#if defined(MUGRID_ENABLE_CUDA)
            cudaMalloc(&d_quad_weights, quad_weights.size() * sizeof(Real));
            cudaMemcpy(d_quad_weights, quad_weights.data(),
                       quad_weights.size() * sizeof(Real), cudaMemcpyHostToDevice);
#elif defined(MUGRID_ENABLE_HIP)
            hipMalloc(&d_quad_weights, quad_weights.size() * sizeof(Real));
            hipMemcpy(d_quad_weights, quad_weights.data(),
                      quad_weights.size() * sizeof(Real), hipMemcpyHostToDevice);
#endif

            // Process each component independently
            for (Index_t comp = 0; comp < nb_components; ++comp) {
                const Real* gradient_comp = gradient + comp * grad_stride_c;
                Real* nodal_comp = nodal + comp * nodal_stride_c;

#if defined(MUGRID_ENABLE_CUDA)
                fem_gradient_kernels::fem_divergence_3d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
                fem_gradient_kernels::fem_divergence_3d_hip(
#endif
                    gradient_comp, nodal_comp, nx, ny, nz,
                    grad_stride_x, grad_stride_y, grad_stride_z,
                    grad_stride_q, grad_stride_d,
                    nodal_stride_x, nodal_stride_y, nodal_stride_z, nodal_stride_n,
                    hx, hy, hz, d_quad_weights, alpha, increment);
            }

            // Free device weights
#if defined(MUGRID_ENABLE_CUDA)
            cudaFree(d_quad_weights);
#elif defined(MUGRID_ENABLE_HIP)
            hipFree(d_quad_weights);
#endif
        }
    }

    void FEMGradientOperator::apply(
        const TypedFieldBase<Real, DefaultDeviceSpace> &nodal_field,
        TypedFieldBase<Real, DefaultDeviceSpace> &gradient_field) const {
        this->apply_impl(nodal_field, gradient_field, 1.0, false);
    }

    void FEMGradientOperator::apply_increment(
        const TypedFieldBase<Real, DefaultDeviceSpace> &nodal_field,
        const Real &alpha,
        TypedFieldBase<Real, DefaultDeviceSpace> &gradient_field) const {
        this->apply_impl(nodal_field, gradient_field, alpha, true);
    }

    void FEMGradientOperator::transpose(
        const TypedFieldBase<Real, DefaultDeviceSpace> &gradient_field,
        TypedFieldBase<Real, DefaultDeviceSpace> &nodal_field,
        const std::vector<Real> &weights) const {
        this->transpose_impl(gradient_field, nodal_field, 1.0, false, weights);
    }

    void FEMGradientOperator::transpose_increment(
        const TypedFieldBase<Real, DefaultDeviceSpace> &gradient_field,
        const Real &alpha,
        TypedFieldBase<Real, DefaultDeviceSpace> &nodal_field,
        const std::vector<Real> &weights) const {
        this->transpose_impl(gradient_field, nodal_field, alpha, true, weights);
    }
#endif

    // =========================================================================
    // Kernel Implementations
    // =========================================================================

    namespace fem_gradient_kernels {

        void fem_gradient_2d_host(
            const Real* MUGRID_RESTRICT nodal_input,
            Real* MUGRID_RESTRICT gradient_output,
            Index_t nx, Index_t ny,
            Index_t nodal_stride_x, Index_t nodal_stride_y, Index_t nodal_stride_n,
            Index_t grad_stride_x, Index_t grad_stride_y,
            Index_t grad_stride_q, Index_t grad_stride_d,
            Real hx, Real hy,
            Real alpha,
            bool increment) {

            // Scale factors for shape function gradients
            const Real inv_hx = alpha / hx;
            const Real inv_hy = alpha / hy;

            // Process interior points (excluding last row/column which need
            // nodes from the next pixel)
            for (Index_t iy = 0; iy < ny - 1; ++iy) {
                #if defined(_MSC_VER)
                #pragma loop(ivdep)
                #elif defined(__clang__)
                #pragma clang loop vectorize(enable) interleave(enable)
                #elif defined(__GNUC__)
                #pragma GCC ivdep
                #endif
                for (Index_t ix = 0; ix < nx - 1; ++ix) {
                    // Base indices for this pixel
                    Index_t nodal_base = ix * nodal_stride_x + iy * nodal_stride_y;
                    Index_t grad_base = ix * grad_stride_x + iy * grad_stride_y;

                    // Get nodal values at pixel corners
                    // Node 0: (ix, iy), Node 1: (ix+1, iy)
                    // Node 2: (ix, iy+1), Node 3: (ix+1, iy+1)
                    Real n0 = nodal_input[nodal_base + 0 * nodal_stride_n];
                    Real n1 = nodal_input[nodal_base + nodal_stride_x + 0 * nodal_stride_n];
                    Real n2 = nodal_input[nodal_base + nodal_stride_y + 0 * nodal_stride_n];
                    Real n3 = nodal_input[nodal_base + nodal_stride_x + nodal_stride_y + 0 * nodal_stride_n];

                    // Triangle 0 (lower-left): nodes 0, 1, 2
                    // grad_x = (-n0 + n1) / hx
                    // grad_y = (-n0 + n2) / hy
                    Real grad_x_t0 = inv_hx * (-n0 + n1);
                    Real grad_y_t0 = inv_hy * (-n0 + n2);

                    // Triangle 1 (upper-right): nodes 1, 3, 2
                    // grad_x = (-n2 + n3) / hx
                    // grad_y = (-n1 + n3) / hy
                    Real grad_x_t1 = inv_hx * (-n2 + n3);
                    Real grad_y_t1 = inv_hy * (-n1 + n3);

                    // Store gradients
                    // Layout: [dim, quad, x, y] with strides
                    if (increment) {
                        // Quad 0 (Triangle 0)
                        gradient_output[grad_base + 0 * grad_stride_d + 0 * grad_stride_q] += grad_x_t0;
                        gradient_output[grad_base + 1 * grad_stride_d + 0 * grad_stride_q] += grad_y_t0;
                        // Quad 1 (Triangle 1)
                        gradient_output[grad_base + 0 * grad_stride_d + 1 * grad_stride_q] += grad_x_t1;
                        gradient_output[grad_base + 1 * grad_stride_d + 1 * grad_stride_q] += grad_y_t1;
                    } else {
                        gradient_output[grad_base + 0 * grad_stride_d + 0 * grad_stride_q] = grad_x_t0;
                        gradient_output[grad_base + 1 * grad_stride_d + 0 * grad_stride_q] = grad_y_t0;
                        gradient_output[grad_base + 0 * grad_stride_d + 1 * grad_stride_q] = grad_x_t1;
                        gradient_output[grad_base + 1 * grad_stride_d + 1 * grad_stride_q] = grad_y_t1;
                    }
                }
            }
        }

        void fem_divergence_2d_host(
            const Real* MUGRID_RESTRICT gradient_input,
            Real* MUGRID_RESTRICT nodal_output,
            Index_t nx, Index_t ny,
            Index_t grad_stride_x, Index_t grad_stride_y,
            Index_t grad_stride_q, Index_t grad_stride_d,
            Index_t nodal_stride_x, Index_t nodal_stride_y, Index_t nodal_stride_n,
            Real hx, Real hy,
            const Real* quad_weights,
            Real alpha,
            bool increment) {

            // Scale factors including quadrature weights
            // For B^T B to be positive semi-definite, we need the true adjoint B^T
            // (not the negative divergence). The adjoint is computed by distributing
            // quadrature point contributions to nodes using the same shape function
            // gradients that were used in the forward pass.
            const Real w0_inv_hx = alpha * quad_weights[0] / hx;
            const Real w0_inv_hy = alpha * quad_weights[0] / hy;
            const Real w1_inv_hx = alpha * quad_weights[1] / hx;
            const Real w1_inv_hy = alpha * quad_weights[1] / hy;

            // Initialize output if not incrementing
            // Output has shape [nb_sub, nx, ny] where nb_sub = 1 for continuous FEM
            if (!increment) {
                Index_t total_size = nx * ny;  // One scalar per grid point
                for (Index_t i = 0; i < total_size; ++i) {
                    nodal_output[i] = 0.0;
                }
            }

            // The transpose accumulates contributions from all quadrature points
            // to the nodal points. Each quadrature point contributes to all
            // nodes that have non-zero shape functions at that point.
            for (Index_t iy = 0; iy < ny - 1; ++iy) {
                for (Index_t ix = 0; ix < nx - 1; ++ix) {
                    Index_t grad_base = ix * grad_stride_x + iy * grad_stride_y;
                    Index_t nodal_base = ix * nodal_stride_x + iy * nodal_stride_y;

                    // Get gradient values at quadrature points
                    Real gx_t0 = gradient_input[grad_base + 0 * grad_stride_d + 0 * grad_stride_q];
                    Real gy_t0 = gradient_input[grad_base + 1 * grad_stride_d + 0 * grad_stride_q];
                    Real gx_t1 = gradient_input[grad_base + 0 * grad_stride_d + 1 * grad_stride_q];
                    Real gy_t1 = gradient_input[grad_base + 1 * grad_stride_d + 1 * grad_stride_q];

                    // Triangle 0 contributions: B^T * sigma
                    // Node 0: dN0/dx * gx + dN0/dy * gy = -1/hx * gx - 1/hy * gy
                    // Node 1: dN1/dx * gx + dN1/dy * gy = +1/hx * gx + 0
                    // Node 2: dN2/dx * gx + dN2/dy * gy = 0 + 1/hy * gy
                    // Node 3: 0
                    Real contrib_n0_t0 = w0_inv_hx * (-gx_t0) + w0_inv_hy * (-gy_t0);
                    Real contrib_n1_t0 = w0_inv_hx * (gx_t0);
                    Real contrib_n2_t0 = w0_inv_hy * (gy_t0);

                    // Triangle 1 contributions:
                    // Node 0: 0
                    // Node 1: dN1/dx * gx + dN1/dy * gy = 0 - 1/hy * gy
                    // Node 2: dN2/dx * gx + dN2/dy * gy = -1/hx * gx + 0
                    // Node 3: dN3/dx * gx + dN3/dy * gy = 1/hx * gx + 1/hy * gy
                    Real contrib_n1_t1 = w1_inv_hy * (-gy_t1);
                    Real contrib_n2_t1 = w1_inv_hx * (-gx_t1);
                    Real contrib_n3_t1 = w1_inv_hx * (gx_t1) + w1_inv_hy * (gy_t1);

                    // Accumulate to nodal points (single DOF per node for scalar)
                    nodal_output[nodal_base] += contrib_n0_t0;
                    nodal_output[nodal_base + nodal_stride_x] += contrib_n1_t0 + contrib_n1_t1;
                    nodal_output[nodal_base + nodal_stride_y] += contrib_n2_t0 + contrib_n2_t1;
                    nodal_output[nodal_base + nodal_stride_x + nodal_stride_y] += contrib_n3_t1;
                }
            }
        }

        void fem_gradient_3d_host(
            const Real* MUGRID_RESTRICT nodal_input,
            Real* MUGRID_RESTRICT gradient_output,
            Index_t nx, Index_t ny, Index_t nz,
            Index_t nodal_stride_x, Index_t nodal_stride_y, Index_t nodal_stride_z,
            Index_t nodal_stride_n,
            Index_t grad_stride_x, Index_t grad_stride_y, Index_t grad_stride_z,
            Index_t grad_stride_q, Index_t grad_stride_d,
            Real hx, Real hy, Real hz,
            Real alpha,
            bool increment) {

            const Real inv_hx = alpha / hx;
            const Real inv_hy = alpha / hy;
            const Real inv_hz = alpha / hz;

            for (Index_t iz = 0; iz < nz - 1; ++iz) {
                for (Index_t iy = 0; iy < ny - 1; ++iy) {
                    #if defined(_MSC_VER)
                    #pragma loop(ivdep)
                    #elif defined(__clang__)
                    #pragma clang loop vectorize(enable) interleave(enable)
                    #elif defined(__GNUC__)
                    #pragma GCC ivdep
                    #endif
                    for (Index_t ix = 0; ix < nx - 1; ++ix) {
                        Index_t nodal_base = ix * nodal_stride_x +
                                             iy * nodal_stride_y +
                                             iz * nodal_stride_z;
                        Index_t grad_base = ix * grad_stride_x +
                                            iy * grad_stride_y +
                                            iz * grad_stride_z;

                        // Get all 8 nodal values
                        Real n[8];
                        for (Index_t node = 0; node < 8; ++node) {
                            Index_t ox = NODE_OFFSET_3D[node][0];
                            Index_t oy = NODE_OFFSET_3D[node][1];
                            Index_t oz = NODE_OFFSET_3D[node][2];
                            n[node] = nodal_input[nodal_base +
                                                  ox * nodal_stride_x +
                                                  oy * nodal_stride_y +
                                                  oz * nodal_stride_z];
                        }

                        // Compute gradients for each tetrahedron
                        for (Index_t q = 0; q < NB_QUAD_3D; ++q) {
                            Real grad_x = 0.0, grad_y = 0.0, grad_z = 0.0;
                            for (Index_t node = 0; node < 8; ++node) {
                                grad_x += B_3D_REF[0][q][node] * n[node];
                                grad_y += B_3D_REF[1][q][node] * n[node];
                                grad_z += B_3D_REF[2][q][node] * n[node];
                            }
                            grad_x *= inv_hx;
                            grad_y *= inv_hy;
                            grad_z *= inv_hz;

                            Index_t grad_idx = grad_base + q * grad_stride_q;
                            if (increment) {
                                gradient_output[grad_idx + 0 * grad_stride_d] += grad_x;
                                gradient_output[grad_idx + 1 * grad_stride_d] += grad_y;
                                gradient_output[grad_idx + 2 * grad_stride_d] += grad_z;
                            } else {
                                gradient_output[grad_idx + 0 * grad_stride_d] = grad_x;
                                gradient_output[grad_idx + 1 * grad_stride_d] = grad_y;
                                gradient_output[grad_idx + 2 * grad_stride_d] = grad_z;
                            }
                        }
                    }
                }
            }
        }

        void fem_divergence_3d_host(
            const Real* MUGRID_RESTRICT gradient_input,
            Real* MUGRID_RESTRICT nodal_output,
            Index_t nx, Index_t ny, Index_t nz,
            Index_t grad_stride_x, Index_t grad_stride_y, Index_t grad_stride_z,
            Index_t grad_stride_q, Index_t grad_stride_d,
            Index_t nodal_stride_x, Index_t nodal_stride_y, Index_t nodal_stride_z,
            Index_t nodal_stride_n,
            Real hx, Real hy, Real hz,
            const Real* quad_weights,
            Real alpha,
            bool increment) {

            // Initialize output if not incrementing
            if (!increment) {
                Index_t total_size = nx * ny * nz;
                for (Index_t i = 0; i < total_size; ++i) {
                    nodal_output[i] = 0.0;
                }
            }

            // For B^T B to be positive semi-definite, use positive signs
            const Real inv_hx = alpha / hx;
            const Real inv_hy = alpha / hy;
            const Real inv_hz = alpha / hz;

            for (Index_t iz = 0; iz < nz - 1; ++iz) {
                for (Index_t iy = 0; iy < ny - 1; ++iy) {
                    for (Index_t ix = 0; ix < nx - 1; ++ix) {
                        Index_t grad_base = ix * grad_stride_x +
                                            iy * grad_stride_y +
                                            iz * grad_stride_z;
                        Index_t nodal_base = ix * nodal_stride_x +
                                             iy * nodal_stride_y +
                                             iz * nodal_stride_z;

                        // For each quadrature point
                        for (Index_t q = 0; q < NB_QUAD_3D; ++q) {
                            Real w = quad_weights[q];
                            Index_t grad_idx = grad_base + q * grad_stride_q;
                            Real gx = gradient_input[grad_idx + 0 * grad_stride_d];
                            Real gy = gradient_input[grad_idx + 1 * grad_stride_d];
                            Real gz = gradient_input[grad_idx + 2 * grad_stride_d];

                            // Accumulate B^T * g to each node
                            for (Index_t node = 0; node < 8; ++node) {
                                Real contrib = w * (B_3D_REF[0][q][node] * inv_hx * gx +
                                                    B_3D_REF[1][q][node] * inv_hy * gy +
                                                    B_3D_REF[2][q][node] * inv_hz * gz);
                                Index_t ox = NODE_OFFSET_3D[node][0];
                                Index_t oy = NODE_OFFSET_3D[node][1];
                                Index_t oz = NODE_OFFSET_3D[node][2];
                                nodal_output[nodal_base +
                                             ox * nodal_stride_x +
                                             oy * nodal_stride_y +
                                             oz * nodal_stride_z] += contrib;
                            }
                        }
                    }
                }
            }
        }

    }  // namespace fem_gradient_kernels

}  // namespace muGrid
