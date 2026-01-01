/**
 * @file   isotropic_stiffness_operator_gpu.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   31 Dec 2025
 *
 * @brief  GPU (CUDA/HIP) implementation of fused isotropic stiffness operator
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

#include "isotropic_stiffness_2d.hh"
#include "isotropic_stiffness_3d.hh"
#include "collection/field_collection_global.hh"
#include "core/exception.hh"

#if defined(MUGRID_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(MUGRID_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)

namespace muGrid {

// ============================================================================
// GPU Constants and Kernels
// ============================================================================

// 2D: 8x8 matrices (4 nodes × 2 DOFs each)
__constant__ Real d_G_2D[64];
__constant__ Real d_V_2D[64];

// 3D: 24x24 matrices (8 nodes × 3 DOFs each)
__constant__ Real d_G_3D[576];
__constant__ Real d_V_3D[576];

// Node offsets for element local numbering
__constant__ int d_NODE_OFFSET_2D[4][2] = {
    {0, 0}, {1, 0}, {0, 1}, {1, 1}
};

__constant__ int d_NODE_OFFSET_3D[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
};

// ============================================================================
// 2D GPU Kernel (Gather Pattern - No Atomics)
// ============================================================================

/**
 * @brief 2D isotropic stiffness kernel using gather pattern
 *
 * Each thread computes the force at one NODE by gathering contributions from
 * all neighboring elements (up to 4 pixels share each node). This avoids
 * atomic operations entirely.
 *
 * For a node at (ix, iy), the neighboring elements are:
 *   - Element (ix-1, iy-1): this node is local node 3 (corner 1,1)
 *   - Element (ix,   iy-1): this node is local node 2 (corner 0,1)
 *   - Element (ix-1, iy  ): this node is local node 1 (corner 1,0)
 *   - Element (ix,   iy  ): this node is local node 0 (corner 0,0)
 */
__global__ void isotropic_stiffness_2d_kernel(
    const Real* __restrict__ displacement,
    const Real* __restrict__ lambda,
    const Real* __restrict__ mu,
    Real* __restrict__ force,
    Index_t nnx, Index_t nny,
    Index_t nelx, Index_t nely,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_d,
    Index_t mat_stride_x, Index_t mat_stride_y,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_d,
    Real alpha, bool increment, bool periodic) {

    constexpr int NB_NODES = 4;
    constexpr int NB_DOFS = 2;
    constexpr int NB_ELEM_DOFS = NB_NODES * NB_DOFS;

    // Thread indexing for NODES
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds (loop over interior nodes)
    if (ix >= nnx || iy >= nny) return;

    // Neighboring element offsets and corresponding local node index
    // Element at (ix + eox, iy + eoy) has this node as local node `local_node`
    const int ELEM_OFFSETS[4][3] = {
        {-1, -1, 3},  // Element (ix-1, iy-1): this node is local node 3
        { 0, -1, 2},  // Element (ix,   iy-1): this node is local node 2
        {-1,  0, 1},  // Element (ix-1, iy  ): this node is local node 1
        { 0,  0, 0}   // Element (ix,   iy  ): this node is local node 0
    };

    // Accumulate force for this node
    Real f[NB_DOFS] = {0.0, 0.0};

    // Loop over neighboring elements
    #pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
        int ex = ix + ELEM_OFFSETS[elem][0];
        int ey = iy + ELEM_OFFSETS[elem][1];
        int local_node = ELEM_OFFSETS[elem][2];

        // Handle periodic wrapping or skip out-of-bounds
        if (periodic) {
            ex = (ex + nelx) % nelx;
            ey = (ey + nely) % nely;
        } else {
            if (ex < 0 || ex >= nelx || ey < 0 || ey >= nely) continue;
        }

        // Get material parameters for this element
        Real lam = lambda[ex * mat_stride_x + ey * mat_stride_y];
        Real mu_val = mu[ex * mat_stride_x + ey * mat_stride_y];

        // Gather displacements from all 4 nodes of this element
        Real u[NB_ELEM_DOFS];
        #pragma unroll
        for (int node = 0; node < NB_NODES; ++node) {
            int nix = ex + d_NODE_OFFSET_2D[node][0];
            int niy = ey + d_NODE_OFFSET_2D[node][1];
            Index_t base = nix * disp_stride_x + niy * disp_stride_y;
            #pragma unroll
            for (int d = 0; d < NB_DOFS; ++d) {
                u[node * NB_DOFS + d] = displacement[base + d * disp_stride_d];
            }
        }

        // Compute only the rows of K @ u that correspond to this node
        // f_local = K[local_node*NB_DOFS:(local_node+1)*NB_DOFS, :] @ u
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d) {
            int row = local_node * NB_DOFS + d;
            Real contrib = 0.0;
            #pragma unroll
            for (int j = 0; j < NB_ELEM_DOFS; ++j) {
                contrib += (2.0 * mu_val * d_G_2D[row * NB_ELEM_DOFS + j] +
                            lam * d_V_2D[row * NB_ELEM_DOFS + j]) * u[j];
            }
            f[d] += contrib;
        }
    }

    // Write force to output (no atomics needed!)
    Index_t base = ix * force_stride_x + iy * force_stride_y;
    if (increment) {
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d) {
            force[base + d * force_stride_d] += alpha * f[d];
        }
    } else {
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d) {
            force[base + d * force_stride_d] = alpha * f[d];
        }
    }
}

// ============================================================================
// 3D GPU Kernel (Gather Pattern - No Atomics)
// ============================================================================

/**
 * @brief 3D isotropic stiffness kernel using gather pattern
 *
 * Each thread computes the force at one NODE by gathering contributions from
 * all neighboring elements (up to 8 voxels share each node). This avoids
 * atomic operations entirely.
 *
 * For a node at (ix, iy, iz), the neighboring elements are the 8 voxels
 * at positions (ix+ox, iy+oy, iz+oz) where ox, oy, oz ∈ {-1, 0}.
 * The local node index within each element depends on which corner this
 * node occupies.
 */
__global__ void isotropic_stiffness_3d_kernel(
    const Real* __restrict__ displacement,
    const Real* __restrict__ lambda,
    const Real* __restrict__ mu,
    Real* __restrict__ force,
    Index_t nnx, Index_t nny, Index_t nnz,
    Index_t nelx, Index_t nely, Index_t nelz,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_z,
    Index_t disp_stride_d,
    Index_t mat_stride_x, Index_t mat_stride_y, Index_t mat_stride_z,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_z,
    Index_t force_stride_d,
    Real alpha, bool increment, bool periodic) {

    constexpr int NB_NODES = 8;
    constexpr int NB_DOFS = 3;
    constexpr int NB_ELEM_DOFS = NB_NODES * NB_DOFS;

    // Thread indexing for NODES
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    Index_t iz = blockIdx.z * blockDim.z + threadIdx.z;

    // Check bounds (loop over interior nodes)
    if (ix >= nnx || iy >= nny || iz >= nnz) return;

    // Neighboring element offsets and corresponding local node index
    // Element at (ix + eox, iy + eoy, iz + eoz) has this node as local node `local_node`
    // Local node index = (1-eox-1) + 2*(1-eoy-1) + 4*(1-eoz-1) = -eox + 2*(-eoy) + 4*(-eoz)
    // When eox=-1: contributes 1, eox=0: contributes 0
    const int ELEM_OFFSETS[8][4] = {
        {-1, -1, -1, 7},  // Element (ix-1, iy-1, iz-1): local node 7 (corner 1,1,1)
        { 0, -1, -1, 6},  // Element (ix,   iy-1, iz-1): local node 6 (corner 0,1,1)
        {-1,  0, -1, 5},  // Element (ix-1, iy,   iz-1): local node 5 (corner 1,0,1)
        { 0,  0, -1, 4},  // Element (ix,   iy,   iz-1): local node 4 (corner 0,0,1)
        {-1, -1,  0, 3},  // Element (ix-1, iy-1, iz  ): local node 3 (corner 1,1,0)
        { 0, -1,  0, 2},  // Element (ix,   iy-1, iz  ): local node 2 (corner 0,1,0)
        {-1,  0,  0, 1},  // Element (ix-1, iy,   iz  ): local node 1 (corner 1,0,0)
        { 0,  0,  0, 0}   // Element (ix,   iy,   iz  ): local node 0 (corner 0,0,0)
    };

    // Accumulate force for this node
    Real f[NB_DOFS] = {0.0, 0.0, 0.0};

    // Loop over neighboring elements
    #pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
        int ex = ix + ELEM_OFFSETS[elem][0];
        int ey = iy + ELEM_OFFSETS[elem][1];
        int ez = iz + ELEM_OFFSETS[elem][2];
        int local_node = ELEM_OFFSETS[elem][3];

        // Handle periodic wrapping or skip out-of-bounds
        if (periodic) {
            ex = (ex + nelx) % nelx;
            ey = (ey + nely) % nely;
            ez = (ez + nelz) % nelz;
        } else {
            if (ex < 0 || ex >= nelx ||
                ey < 0 || ey >= nely ||
                ez < 0 || ez >= nelz) continue;
        }

        // Get material parameters for this element
        Index_t mat_idx = ex * mat_stride_x + ey * mat_stride_y + ez * mat_stride_z;
        Real lam = lambda[mat_idx];
        Real mu_val = mu[mat_idx];

        // Gather displacements from all 8 nodes of this element
        Real u[NB_ELEM_DOFS];
        #pragma unroll
        for (int node = 0; node < NB_NODES; ++node) {
            int nix = ex + d_NODE_OFFSET_3D[node][0];
            int niy = ey + d_NODE_OFFSET_3D[node][1];
            int niz = ez + d_NODE_OFFSET_3D[node][2];
            Index_t base = nix * disp_stride_x + niy * disp_stride_y +
                           niz * disp_stride_z;
            #pragma unroll
            for (int d = 0; d < NB_DOFS; ++d) {
                u[node * NB_DOFS + d] = displacement[base + d * disp_stride_d];
            }
        }

        // Compute only the rows of K @ u that correspond to this node
        // f_local = K[local_node*NB_DOFS:(local_node+1)*NB_DOFS, :] @ u
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d) {
            int row = local_node * NB_DOFS + d;
            Real contrib = 0.0;
            #pragma unroll
            for (int j = 0; j < NB_ELEM_DOFS; ++j) {
                contrib += (2.0 * mu_val * d_G_3D[row * NB_ELEM_DOFS + j] +
                            lam * d_V_3D[row * NB_ELEM_DOFS + j]) * u[j];
            }
            f[d] += contrib;
        }
    }

    // Write force to output (no atomics needed!)
    Index_t base = ix * force_stride_x + iy * force_stride_y + iz * force_stride_z;
    if (increment) {
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d) {
            force[base + d * force_stride_d] += alpha * f[d];
        }
    } else {
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d) {
            force[base + d * force_stride_d] = alpha * f[d];
        }
    }
}

// ============================================================================
// Kernel Wrapper Functions
// ============================================================================

namespace isotropic_stiffness_kernels {

#if defined(MUGRID_ENABLE_CUDA)

static bool g_2d_constants_initialized = false;
static bool g_3d_constants_initialized = false;

void isotropic_stiffness_2d_cuda(
    const Real* displacement, const Real* lambda, const Real* mu,
    Real* force,
    Index_t nnx, Index_t nny,
    Index_t nelx, Index_t nely,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_d,
    Index_t mat_stride_x, Index_t mat_stride_y,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_d,
    const Real* G, const Real* V,
    Real alpha, bool increment, bool periodic) {

    // Copy G and V to constant memory if not already done
    if (!g_2d_constants_initialized) {
        cudaMemcpyToSymbol(d_G_2D, G, 64 * sizeof(Real));
        cudaMemcpyToSymbol(d_V_2D, V, 64 * sizeof(Real));
        g_2d_constants_initialized = true;
    }

    // Launch stiffness kernel - one thread per NODE (gather pattern writes directly)
    dim3 block(16, 16);
    dim3 grid((nnx + block.x - 1) / block.x, (nny + block.y - 1) / block.y);

    isotropic_stiffness_2d_kernel<<<grid, block>>>(
        displacement, lambda, mu, force,
        nnx, nny, nelx, nely,
        disp_stride_x, disp_stride_y, disp_stride_d,
        mat_stride_x, mat_stride_y,
        force_stride_x, force_stride_y, force_stride_d,
        alpha, increment, periodic);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw RuntimeError("CUDA kernel launch failed: " +
                          std::string(cudaGetErrorString(err)));
    }
}

void isotropic_stiffness_3d_cuda(
    const Real* displacement, const Real* lambda, const Real* mu,
    Real* force,
    Index_t nnx, Index_t nny, Index_t nnz,
    Index_t nelx, Index_t nely, Index_t nelz,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_z,
    Index_t disp_stride_d,
    Index_t mat_stride_x, Index_t mat_stride_y, Index_t mat_stride_z,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_z,
    Index_t force_stride_d,
    const Real* G, const Real* V,
    Real alpha, bool increment, bool periodic) {

    // Copy G and V to constant memory if not already done
    if (!g_3d_constants_initialized) {
        cudaMemcpyToSymbol(d_G_3D, G, 576 * sizeof(Real));
        cudaMemcpyToSymbol(d_V_3D, V, 576 * sizeof(Real));
        g_3d_constants_initialized = true;
    }

    // Launch stiffness kernel - one thread per NODE (gather pattern writes directly)
    dim3 block(8, 8, 4);
    dim3 grid((nnx + block.x - 1) / block.x,
              (nny + block.y - 1) / block.y,
              (nnz + block.z - 1) / block.z);

    isotropic_stiffness_3d_kernel<<<grid, block>>>(
        displacement, lambda, mu, force,
        nnx, nny, nnz, nelx, nely, nelz,
        disp_stride_x, disp_stride_y, disp_stride_z, disp_stride_d,
        mat_stride_x, mat_stride_y, mat_stride_z,
        force_stride_x, force_stride_y, force_stride_z, force_stride_d,
        alpha, increment, periodic);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw RuntimeError("CUDA kernel launch failed: " +
                          std::string(cudaGetErrorString(err)));
    }
}

#endif  // MUGRID_ENABLE_CUDA

#if defined(MUGRID_ENABLE_HIP)

static bool g_2d_constants_initialized = false;
static bool g_3d_constants_initialized = false;

void isotropic_stiffness_2d_hip(
    const Real* displacement, const Real* lambda, const Real* mu,
    Real* force,
    Index_t nnx, Index_t nny,
    Index_t nelx, Index_t nely,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_d,
    Index_t mat_stride_x, Index_t mat_stride_y,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_d,
    const Real* G, const Real* V,
    Real alpha, bool increment, bool periodic) {

    // Copy G and V to constant memory if not already done
    if (!g_2d_constants_initialized) {
        hipMemcpyToSymbol(d_G_2D, G, 64 * sizeof(Real));
        hipMemcpyToSymbol(d_V_2D, V, 64 * sizeof(Real));
        g_2d_constants_initialized = true;
    }

    // Launch stiffness kernel - one thread per NODE (gather pattern writes directly)
    dim3 block(16, 16);
    dim3 grid((nnx + block.x - 1) / block.x, (nny + block.y - 1) / block.y);

    hipLaunchKernelGGL(isotropic_stiffness_2d_kernel, grid, block, 0, 0,
        displacement, lambda, mu, force,
        nnx, nny, nelx, nely,
        disp_stride_x, disp_stride_y, disp_stride_d,
        mat_stride_x, mat_stride_y,
        force_stride_x, force_stride_y, force_stride_d,
        alpha, increment, periodic);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw RuntimeError("HIP kernel launch failed: " +
                          std::string(hipGetErrorString(err)));
    }
}

void isotropic_stiffness_3d_hip(
    const Real* displacement, const Real* lambda, const Real* mu,
    Real* force,
    Index_t nnx, Index_t nny, Index_t nnz,
    Index_t nelx, Index_t nely, Index_t nelz,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_z,
    Index_t disp_stride_d,
    Index_t mat_stride_x, Index_t mat_stride_y, Index_t mat_stride_z,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_z,
    Index_t force_stride_d,
    const Real* G, const Real* V,
    Real alpha, bool increment, bool periodic) {

    // Copy G and V to constant memory if not already done
    if (!g_3d_constants_initialized) {
        hipMemcpyToSymbol(d_G_3D, G, 576 * sizeof(Real));
        hipMemcpyToSymbol(d_V_3D, V, 576 * sizeof(Real));
        g_3d_constants_initialized = true;
    }

    // Launch stiffness kernel - one thread per NODE (gather pattern writes directly)
    dim3 block(8, 8, 4);
    dim3 grid((nnx + block.x - 1) / block.x,
              (nny + block.y - 1) / block.y,
              (nnz + block.z - 1) / block.z);

    hipLaunchKernelGGL(isotropic_stiffness_3d_kernel, grid, block, 0, 0,
        displacement, lambda, mu, force,
        nnx, nny, nnz, nelx, nely, nelz,
        disp_stride_x, disp_stride_y, disp_stride_z, disp_stride_d,
        mat_stride_x, mat_stride_y, mat_stride_z,
        force_stride_x, force_stride_y, force_stride_z, force_stride_d,
        alpha, increment, periodic);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw RuntimeError("HIP kernel launch failed: " +
                          std::string(hipGetErrorString(err)));
    }
}

#endif  // MUGRID_ENABLE_HIP

}  // namespace isotropic_stiffness_kernels

// ============================================================================
// Class Method Implementations for GPU
// ============================================================================

#if defined(MUGRID_ENABLE_CUDA)

void IsotropicStiffnessOperator2D::apply(
    const TypedFieldBase<Real, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<Real, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<Real, DefaultDeviceSpace>& mu,
    TypedFieldBase<Real, DefaultDeviceSpace>& force) const {
    apply_impl(displacement, lambda, mu, 1.0, force, false);
}

void IsotropicStiffnessOperator2D::apply_increment(
    const TypedFieldBase<Real, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<Real, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<Real, DefaultDeviceSpace>& mu,
    Real alpha,
    TypedFieldBase<Real, DefaultDeviceSpace>& force) const {
    apply_impl(displacement, lambda, mu, alpha, force, true);
}

void IsotropicStiffnessOperator2D::apply_impl(
    const TypedFieldBase<Real, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<Real, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<Real, DefaultDeviceSpace>& mu,
    Real alpha,
    TypedFieldBase<Real, DefaultDeviceSpace>& force,
    bool increment) const {

    auto& disp_coll = displacement.get_collection();
    auto* disp_global_fc = dynamic_cast<const GlobalFieldCollection*>(&disp_coll);
    if (!disp_global_fc) {
        throw RuntimeError("IsotropicStiffnessOperator2D requires GlobalFieldCollection");
    }

    // Get dimensions from material field collection (defines number of elements)
    auto& mat_coll = lambda.get_collection();
    auto* mat_global_fc = dynamic_cast<const GlobalFieldCollection*>(&mat_coll);
    if (!mat_global_fc) {
        throw RuntimeError("IsotropicStiffnessOperator2D material fields require GlobalFieldCollection");
    }

    // Validate ghost configuration for displacement/force fields
    auto nb_ghosts_left = disp_global_fc->get_nb_ghosts_left();
    auto nb_ghosts_right = disp_global_fc->get_nb_ghosts_right();
    if (nb_ghosts_right[0] < 1 || nb_ghosts_right[1] < 1) {
        throw RuntimeError(
            "IsotropicStiffnessOperator2D requires at least 1 ghost cell on the "
            "right side of displacement/force fields (nb_ghosts_right >= (1, 1))");
    }

    // Material field dimensions = number of elements (no ghosts on material field)
    auto nb_elements = mat_global_fc->get_nb_subdomain_grid_pts_without_ghosts();
    Index_t nelx = nb_elements[0];
    Index_t nely = nb_elements[1];

    // Get number of interior nodes
    auto nb_interior = disp_global_fc->get_nb_subdomain_grid_pts_without_ghosts();
    Index_t nnx = nb_interior[0];
    Index_t nny = nb_interior[1];

    // Determine if periodic BC based on material field size
    bool periodic = (nelx == nnx) && (nely == nny);
    bool non_periodic = (nelx == nnx - 1) && (nely == nny - 1);

    if (!periodic && !non_periodic) {
        throw RuntimeError(
            "Material field dimensions (" + std::to_string(nelx) + ", " +
            std::to_string(nely) + ") must equal interior nodes (" +
            std::to_string(nnx) + ", " + std::to_string(nny) +
            ") for periodic BC, or interior nodes - 1 for non-periodic BC");
    }

    if (periodic && (nb_ghosts_left[0] < 1 || nb_ghosts_left[1] < 1)) {
        throw RuntimeError(
            "IsotropicStiffnessOperator2D with periodic BC requires at least 1 "
            "ghost cell on the left side of displacement/force fields");
    }

    // Node dimensions (for displacement/force fields with ghosts)
    auto nb_nodes = disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t nx = nb_nodes[0];
    Index_t ny = nb_nodes[1];

    // Ghost offsets (interior starts at this offset in the ghosted array)
    Index_t ghost_offset_x = nb_ghosts_left[0];
    Index_t ghost_offset_y = nb_ghosts_left[1];

    // GPU uses SoA layout: [d, x, y]
    Index_t disp_stride_d = nx * ny;
    Index_t disp_stride_x = 1;
    Index_t disp_stride_y = nx;

    Index_t mat_stride_x = 1;
    Index_t mat_stride_y = nelx;

    Index_t force_stride_d = nx * ny;
    Index_t force_stride_x = 1;
    Index_t force_stride_y = nx;

    // Offset data pointers to account for left ghosts
    Index_t disp_offset = ghost_offset_x * disp_stride_x +
                          ghost_offset_y * disp_stride_y;
    Index_t force_offset = ghost_offset_x * force_stride_x +
                           ghost_offset_y * force_stride_y;

    isotropic_stiffness_kernels::isotropic_stiffness_2d_cuda(
        displacement.view().data() + disp_offset, lambda.view().data(), mu.view().data(),
        force.view().data() + force_offset,
        nnx, nny,  // Number of interior nodes
        nelx, nely,  // Number of elements
        disp_stride_x, disp_stride_y, disp_stride_d,
        mat_stride_x, mat_stride_y,
        force_stride_x, force_stride_y, force_stride_d,
        G_matrix.data(), V_matrix.data(),
        alpha, increment, periodic);
}

void IsotropicStiffnessOperator3D::apply(
    const TypedFieldBase<Real, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<Real, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<Real, DefaultDeviceSpace>& mu,
    TypedFieldBase<Real, DefaultDeviceSpace>& force) const {
    apply_impl(displacement, lambda, mu, 1.0, force, false);
}

void IsotropicStiffnessOperator3D::apply_increment(
    const TypedFieldBase<Real, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<Real, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<Real, DefaultDeviceSpace>& mu,
    Real alpha,
    TypedFieldBase<Real, DefaultDeviceSpace>& force) const {
    apply_impl(displacement, lambda, mu, alpha, force, true);
}

void IsotropicStiffnessOperator3D::apply_impl(
    const TypedFieldBase<Real, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<Real, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<Real, DefaultDeviceSpace>& mu,
    Real alpha,
    TypedFieldBase<Real, DefaultDeviceSpace>& force,
    bool increment) const {

    auto& disp_coll = displacement.get_collection();
    auto* disp_global_fc = dynamic_cast<const GlobalFieldCollection*>(&disp_coll);
    if (!disp_global_fc) {
        throw RuntimeError("IsotropicStiffnessOperator3D requires GlobalFieldCollection");
    }

    // Get dimensions from material field collection (defines number of elements)
    auto& mat_coll = lambda.get_collection();
    auto* mat_global_fc = dynamic_cast<const GlobalFieldCollection*>(&mat_coll);
    if (!mat_global_fc) {
        throw RuntimeError("IsotropicStiffnessOperator3D material fields require GlobalFieldCollection");
    }

    // Validate ghost configuration for displacement/force fields
    auto nb_ghosts_left = disp_global_fc->get_nb_ghosts_left();
    auto nb_ghosts_right = disp_global_fc->get_nb_ghosts_right();
    if (nb_ghosts_right[0] < 1 || nb_ghosts_right[1] < 1 || nb_ghosts_right[2] < 1) {
        throw RuntimeError(
            "IsotropicStiffnessOperator3D requires at least 1 ghost cell on the "
            "right side of displacement/force fields (nb_ghosts_right >= (1, 1, 1))");
    }

    // Material field dimensions = number of elements
    auto nb_elements = mat_global_fc->get_nb_subdomain_grid_pts_without_ghosts();
    Index_t nelx = nb_elements[0];
    Index_t nely = nb_elements[1];
    Index_t nelz = nb_elements[2];

    // Get number of interior nodes
    auto nb_interior = disp_global_fc->get_nb_subdomain_grid_pts_without_ghosts();
    Index_t nnx = nb_interior[0];
    Index_t nny = nb_interior[1];
    Index_t nnz = nb_interior[2];

    // Determine if periodic BC based on material field size
    bool periodic = (nelx == nnx) && (nely == nny) && (nelz == nnz);
    bool non_periodic = (nelx == nnx - 1) && (nely == nny - 1) && (nelz == nnz - 1);

    if (!periodic && !non_periodic) {
        throw RuntimeError(
            "Material field dimensions (" + std::to_string(nelx) + ", " +
            std::to_string(nely) + ", " + std::to_string(nelz) +
            ") must equal interior nodes (" + std::to_string(nnx) + ", " +
            std::to_string(nny) + ", " + std::to_string(nnz) +
            ") for periodic BC, or interior nodes - 1 for non-periodic BC");
    }

    if (periodic && (nb_ghosts_left[0] < 1 || nb_ghosts_left[1] < 1 || nb_ghosts_left[2] < 1)) {
        throw RuntimeError(
            "IsotropicStiffnessOperator3D with periodic BC requires at least 1 "
            "ghost cell on the left side of displacement/force fields");
    }

    // Node dimensions (for displacement/force fields with ghosts)
    auto nb_nodes = disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t nx = nb_nodes[0];
    Index_t ny = nb_nodes[1];
    Index_t nz = nb_nodes[2];

    // Ghost offsets (interior starts at this offset in the ghosted array)
    Index_t ghost_offset_x = nb_ghosts_left[0];
    Index_t ghost_offset_y = nb_ghosts_left[1];
    Index_t ghost_offset_z = nb_ghosts_left[2];

    // GPU uses SoA layout: [d, x, y, z]
    Index_t disp_stride_d = nx * ny * nz;
    Index_t disp_stride_x = 1;
    Index_t disp_stride_y = nx;
    Index_t disp_stride_z = nx * ny;

    Index_t mat_stride_x = 1;
    Index_t mat_stride_y = nelx;
    Index_t mat_stride_z = nelx * nely;

    Index_t force_stride_d = nx * ny * nz;
    Index_t force_stride_x = 1;
    Index_t force_stride_y = nx;
    Index_t force_stride_z = nx * ny;

    // Offset data pointers to account for left ghosts
    Index_t disp_offset = ghost_offset_x * disp_stride_x +
                          ghost_offset_y * disp_stride_y +
                          ghost_offset_z * disp_stride_z;
    Index_t force_offset = ghost_offset_x * force_stride_x +
                           ghost_offset_y * force_stride_y +
                           ghost_offset_z * force_stride_z;

    isotropic_stiffness_kernels::isotropic_stiffness_3d_cuda(
        displacement.view().data() + disp_offset, lambda.view().data(), mu.view().data(),
        force.view().data() + force_offset,
        nnx, nny, nnz,  // Number of interior nodes
        nelx, nely, nelz,  // Number of elements
        disp_stride_x, disp_stride_y, disp_stride_z, disp_stride_d,
        mat_stride_x, mat_stride_y, mat_stride_z,
        force_stride_x, force_stride_y, force_stride_z, force_stride_d,
        G_matrix.data(), V_matrix.data(),
        alpha, increment, periodic);
}

#elif defined(MUGRID_ENABLE_HIP)

void IsotropicStiffnessOperator2D::apply(
    const TypedFieldBase<Real, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<Real, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<Real, DefaultDeviceSpace>& mu,
    TypedFieldBase<Real, DefaultDeviceSpace>& force) const {
    apply_impl(displacement, lambda, mu, 1.0, force, false);
}

void IsotropicStiffnessOperator2D::apply_increment(
    const TypedFieldBase<Real, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<Real, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<Real, DefaultDeviceSpace>& mu,
    Real alpha,
    TypedFieldBase<Real, DefaultDeviceSpace>& force) const {
    apply_impl(displacement, lambda, mu, alpha, force, true);
}

void IsotropicStiffnessOperator2D::apply_impl(
    const TypedFieldBase<Real, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<Real, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<Real, DefaultDeviceSpace>& mu,
    Real alpha,
    TypedFieldBase<Real, DefaultDeviceSpace>& force,
    bool increment) const {

    auto& disp_coll = displacement.get_collection();
    auto* disp_global_fc = dynamic_cast<const GlobalFieldCollection*>(&disp_coll);
    if (!disp_global_fc) {
        throw RuntimeError("IsotropicStiffnessOperator2D requires GlobalFieldCollection");
    }

    // Get dimensions from material field collection (defines number of elements)
    auto& mat_coll = lambda.get_collection();
    auto* mat_global_fc = dynamic_cast<const GlobalFieldCollection*>(&mat_coll);
    if (!mat_global_fc) {
        throw RuntimeError("IsotropicStiffnessOperator2D material fields require GlobalFieldCollection");
    }

    // Validate ghost configuration for displacement/force fields
    auto nb_ghosts_left = disp_global_fc->get_nb_ghosts_left();
    auto nb_ghosts_right = disp_global_fc->get_nb_ghosts_right();
    if (nb_ghosts_right[0] < 1 || nb_ghosts_right[1] < 1) {
        throw RuntimeError(
            "IsotropicStiffnessOperator2D requires at least 1 ghost cell on the "
            "right side of displacement/force fields (nb_ghosts_right >= (1, 1))");
    }

    // Material field dimensions = number of elements
    auto nb_elements = mat_global_fc->get_nb_subdomain_grid_pts_without_ghosts();
    Index_t nelx = nb_elements[0];
    Index_t nely = nb_elements[1];

    // Get number of interior nodes
    auto nb_interior = disp_global_fc->get_nb_subdomain_grid_pts_without_ghosts();
    Index_t nnx = nb_interior[0];
    Index_t nny = nb_interior[1];

    // Determine if periodic BC based on material field size
    bool periodic = (nelx == nnx) && (nely == nny);
    bool non_periodic = (nelx == nnx - 1) && (nely == nny - 1);

    if (!periodic && !non_periodic) {
        throw RuntimeError(
            "Material field dimensions (" + std::to_string(nelx) + ", " +
            std::to_string(nely) + ") must equal interior nodes (" +
            std::to_string(nnx) + ", " + std::to_string(nny) +
            ") for periodic BC, or interior nodes - 1 for non-periodic BC");
    }

    if (periodic && (nb_ghosts_left[0] < 1 || nb_ghosts_left[1] < 1)) {
        throw RuntimeError(
            "IsotropicStiffnessOperator2D with periodic BC requires at least 1 "
            "ghost cell on the left side of displacement/force fields");
    }

    // Node dimensions (for displacement/force fields with ghosts)
    auto nb_nodes = disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t nx = nb_nodes[0];
    Index_t ny = nb_nodes[1];

    // Ghost offsets (interior starts at this offset in the ghosted array)
    Index_t ghost_offset_x = nb_ghosts_left[0];
    Index_t ghost_offset_y = nb_ghosts_left[1];

    // GPU uses SoA layout: [d, x, y]
    Index_t disp_stride_d = nx * ny;
    Index_t disp_stride_x = 1;
    Index_t disp_stride_y = nx;

    Index_t mat_stride_x = 1;
    Index_t mat_stride_y = nelx;

    Index_t force_stride_d = nx * ny;
    Index_t force_stride_x = 1;
    Index_t force_stride_y = nx;

    // Offset data pointers to account for left ghosts
    Index_t disp_offset = ghost_offset_x * disp_stride_x +
                          ghost_offset_y * disp_stride_y;
    Index_t force_offset = ghost_offset_x * force_stride_x +
                           ghost_offset_y * force_stride_y;

    isotropic_stiffness_kernels::isotropic_stiffness_2d_hip(
        displacement.view().data() + disp_offset, lambda.view().data(), mu.view().data(),
        force.view().data() + force_offset,
        nnx, nny,  // Number of interior nodes
        nelx, nely,  // Number of elements
        disp_stride_x, disp_stride_y, disp_stride_d,
        mat_stride_x, mat_stride_y,
        force_stride_x, force_stride_y, force_stride_d,
        G_matrix.data(), V_matrix.data(),
        alpha, increment, periodic);
}

void IsotropicStiffnessOperator3D::apply(
    const TypedFieldBase<Real, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<Real, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<Real, DefaultDeviceSpace>& mu,
    TypedFieldBase<Real, DefaultDeviceSpace>& force) const {
    apply_impl(displacement, lambda, mu, 1.0, force, false);
}

void IsotropicStiffnessOperator3D::apply_increment(
    const TypedFieldBase<Real, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<Real, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<Real, DefaultDeviceSpace>& mu,
    Real alpha,
    TypedFieldBase<Real, DefaultDeviceSpace>& force) const {
    apply_impl(displacement, lambda, mu, alpha, force, true);
}

void IsotropicStiffnessOperator3D::apply_impl(
    const TypedFieldBase<Real, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<Real, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<Real, DefaultDeviceSpace>& mu,
    Real alpha,
    TypedFieldBase<Real, DefaultDeviceSpace>& force,
    bool increment) const {

    auto& disp_coll = displacement.get_collection();
    auto* disp_global_fc = dynamic_cast<const GlobalFieldCollection*>(&disp_coll);
    if (!disp_global_fc) {
        throw RuntimeError("IsotropicStiffnessOperator3D requires GlobalFieldCollection");
    }

    // Get dimensions from material field collection (defines number of elements)
    auto& mat_coll = lambda.get_collection();
    auto* mat_global_fc = dynamic_cast<const GlobalFieldCollection*>(&mat_coll);
    if (!mat_global_fc) {
        throw RuntimeError("IsotropicStiffnessOperator3D material fields require GlobalFieldCollection");
    }

    // Validate ghost configuration for displacement/force fields
    auto nb_ghosts_left = disp_global_fc->get_nb_ghosts_left();
    auto nb_ghosts_right = disp_global_fc->get_nb_ghosts_right();
    if (nb_ghosts_right[0] < 1 || nb_ghosts_right[1] < 1 || nb_ghosts_right[2] < 1) {
        throw RuntimeError(
            "IsotropicStiffnessOperator3D requires at least 1 ghost cell on the "
            "right side of displacement/force fields (nb_ghosts_right >= (1, 1, 1))");
    }

    // Material field dimensions = number of elements
    auto nb_elements = mat_global_fc->get_nb_subdomain_grid_pts_without_ghosts();
    Index_t nelx = nb_elements[0];
    Index_t nely = nb_elements[1];
    Index_t nelz = nb_elements[2];

    // Get number of interior nodes
    auto nb_interior = disp_global_fc->get_nb_subdomain_grid_pts_without_ghosts();
    Index_t nnx = nb_interior[0];
    Index_t nny = nb_interior[1];
    Index_t nnz = nb_interior[2];

    // Determine if periodic BC based on material field size
    bool periodic = (nelx == nnx) && (nely == nny) && (nelz == nnz);
    bool non_periodic = (nelx == nnx - 1) && (nely == nny - 1) && (nelz == nnz - 1);

    if (!periodic && !non_periodic) {
        throw RuntimeError(
            "Material field dimensions (" + std::to_string(nelx) + ", " +
            std::to_string(nely) + ", " + std::to_string(nelz) +
            ") must equal interior nodes (" + std::to_string(nnx) + ", " +
            std::to_string(nny) + ", " + std::to_string(nnz) +
            ") for periodic BC, or interior nodes - 1 for non-periodic BC");
    }

    if (periodic && (nb_ghosts_left[0] < 1 || nb_ghosts_left[1] < 1 || nb_ghosts_left[2] < 1)) {
        throw RuntimeError(
            "IsotropicStiffnessOperator3D with periodic BC requires at least 1 "
            "ghost cell on the left side of displacement/force fields");
    }

    // Node dimensions (for displacement/force fields with ghosts)
    auto nb_nodes = disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t nx = nb_nodes[0];
    Index_t ny = nb_nodes[1];
    Index_t nz = nb_nodes[2];

    // Ghost offsets (interior starts at this offset in the ghosted array)
    Index_t ghost_offset_x = nb_ghosts_left[0];
    Index_t ghost_offset_y = nb_ghosts_left[1];
    Index_t ghost_offset_z = nb_ghosts_left[2];

    // GPU uses SoA layout: [d, x, y, z]
    Index_t disp_stride_d = nx * ny * nz;
    Index_t disp_stride_x = 1;
    Index_t disp_stride_y = nx;
    Index_t disp_stride_z = nx * ny;

    Index_t mat_stride_x = 1;
    Index_t mat_stride_y = nelx;
    Index_t mat_stride_z = nelx * nely;

    Index_t force_stride_d = nx * ny * nz;
    Index_t force_stride_x = 1;
    Index_t force_stride_y = nx;
    Index_t force_stride_z = nx * ny;

    // Offset data pointers to account for left ghosts
    Index_t disp_offset = ghost_offset_x * disp_stride_x +
                          ghost_offset_y * disp_stride_y +
                          ghost_offset_z * disp_stride_z;
    Index_t force_offset = ghost_offset_x * force_stride_x +
                           ghost_offset_y * force_stride_y +
                           ghost_offset_z * force_stride_z;

    isotropic_stiffness_kernels::isotropic_stiffness_3d_hip(
        displacement.view().data() + disp_offset, lambda.view().data(), mu.view().data(),
        force.view().data() + force_offset,
        nnx, nny, nnz,  // Number of interior nodes
        nelx, nely, nelz,  // Number of elements
        disp_stride_x, disp_stride_y, disp_stride_z, disp_stride_d,
        mat_stride_x, mat_stride_y, mat_stride_z,
        force_stride_x, force_stride_y, force_stride_z, force_stride_d,
        G_matrix.data(), V_matrix.data(),
        alpha, increment, periodic);
}

#endif  // HIP

}  // namespace muGrid

#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP
