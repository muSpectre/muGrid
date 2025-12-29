/**
 * @file   fem_gradient_operator_gpu.cpp
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Dec 2024
 *
 * @brief  Unified CUDA/HIP implementation of hard-coded FEM gradient operator
 *
 * This file provides GPU implementations that work with both CUDA and HIP
 * backends. The kernel code is identical; only the launch mechanism and
 * runtime API differ, which are abstracted via macros.
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

// Unified GPU abstraction macros
#if defined(MUGRID_ENABLE_CUDA)
    #include <cuda_runtime.h>
    #define GPU_LAUNCH_KERNEL(kernel, grid, block, ...) \
        kernel<<<grid, block>>>(__VA_ARGS__)
    #define GPU_DEVICE_SYNCHRONIZE() cudaDeviceSynchronize()
#elif defined(MUGRID_ENABLE_HIP)
    #include <hip/hip_runtime.h>
    #define GPU_LAUNCH_KERNEL(kernel, grid, block, ...) \
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, __VA_ARGS__)
    #define GPU_DEVICE_SYNCHRONIZE() hipDeviceSynchronize()
#endif

namespace muGrid {
namespace fem_gradient_kernels {

// Block sizes for GPU kernels
constexpr int BLOCK_SIZE_2D = 16;
constexpr int BLOCK_SIZE_3D = 8;

// =========================================================================
// 2D GPU Kernels
// =========================================================================

/**
 * GPU kernel for 2D FEM gradient operator.
 *
 * Computes gradient at 2 quadrature points per pixel from 4 nodal values.
 * Each thread processes one pixel.
 */
__global__ void fem_gradient_2d_kernel(
    const Real* MUGRID_RESTRICT nodal_input,
    Real* MUGRID_RESTRICT gradient_output,
    Index_t nx, Index_t ny,
    Index_t nodal_stride_x, Index_t nodal_stride_y, Index_t nodal_stride_n,
    Index_t grad_stride_x, Index_t grad_stride_y,
    Index_t grad_stride_q, Index_t grad_stride_d,
    Real inv_hx, Real inv_hy,
    bool increment) {

    // Thread indices - each thread processes one pixel
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds (exclude last row/column which would access out-of-bounds)
    if (ix < nx - 1 && iy < ny - 1) {
        // Base indices for this pixel
        Index_t nodal_base = ix * nodal_stride_x + iy * nodal_stride_y;
        Index_t grad_base = ix * grad_stride_x + iy * grad_stride_y;

        // Get nodal values at pixel corners
        // Node 0: (ix, iy), Node 1: (ix+1, iy)
        // Node 2: (ix, iy+1), Node 3: (ix+1, iy+1)
        Real n0 = nodal_input[nodal_base];
        Real n1 = nodal_input[nodal_base + nodal_stride_x];
        Real n2 = nodal_input[nodal_base + nodal_stride_y];
        Real n3 = nodal_input[nodal_base + nodal_stride_x + nodal_stride_y];

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

/**
 * GPU kernel for 2D FEM divergence (transpose) operator.
 *
 * Uses gather pattern: each thread handles one NODE and gathers contributions
 * from adjacent pixels. This eliminates the need for atomic operations.
 *
 * For node at (ix, iy), gather from:
 * - Pixel (ix, iy): this node is corner 0 (lower-left)
 * - Pixel (ix-1, iy): this node is corner 1 (lower-right)
 * - Pixel (ix, iy-1): this node is corner 2 (upper-left)
 * - Pixel (ix-1, iy-1): this node is corner 3 (upper-right)
 */
__global__ void fem_divergence_2d_kernel(
    const Real* MUGRID_RESTRICT gradient_input,
    Real* MUGRID_RESTRICT nodal_output,
    Index_t nx, Index_t ny,
    Index_t grad_stride_x, Index_t grad_stride_y,
    Index_t grad_stride_q, Index_t grad_stride_d,
    Index_t nodal_stride_x, Index_t nodal_stride_y, Index_t nodal_stride_n,
    Real w0_inv_hx, Real w0_inv_hy,
    Real w1_inv_hx, Real w1_inv_hy,
    bool increment) {

    // Thread indices - each thread processes one NODE
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds (all nodes)
    if (ix < nx && iy < ny) {
        Real contrib = 0.0;

        // Pixel (ix, iy): this node is corner 0 (lower-left)
        // Triangle 0 contribution to node 0: w0*(-inv_hx*gx - inv_hy*gy)
        if (ix < nx - 1 && iy < ny - 1) {
            Index_t grad_base = ix * grad_stride_x + iy * grad_stride_y;
            Real gx_t0 = gradient_input[grad_base + 0 * grad_stride_d + 0 * grad_stride_q];
            Real gy_t0 = gradient_input[grad_base + 1 * grad_stride_d + 0 * grad_stride_q];
            contrib += w0_inv_hx * (-gx_t0) + w0_inv_hy * (-gy_t0);
        }

        // Pixel (ix-1, iy): this node is corner 1 (lower-right)
        // Triangle 0 contribution: w0*(inv_hx*gx)
        // Triangle 1 contribution: w1*(-inv_hy*gy)
        if (ix > 0 && iy < ny - 1) {
            Index_t grad_base = (ix - 1) * grad_stride_x + iy * grad_stride_y;
            Real gx_t0 = gradient_input[grad_base + 0 * grad_stride_d + 0 * grad_stride_q];
            Real gy_t1 = gradient_input[grad_base + 1 * grad_stride_d + 1 * grad_stride_q];
            contrib += w0_inv_hx * gx_t0 + w1_inv_hy * (-gy_t1);
        }

        // Pixel (ix, iy-1): this node is corner 2 (upper-left)
        // Triangle 0 contribution: w0*(inv_hy*gy)
        // Triangle 1 contribution: w1*(-inv_hx*gx)
        if (ix < nx - 1 && iy > 0) {
            Index_t grad_base = ix * grad_stride_x + (iy - 1) * grad_stride_y;
            Real gy_t0 = gradient_input[grad_base + 1 * grad_stride_d + 0 * grad_stride_q];
            Real gx_t1 = gradient_input[grad_base + 0 * grad_stride_d + 1 * grad_stride_q];
            contrib += w0_inv_hy * gy_t0 + w1_inv_hx * (-gx_t1);
        }

        // Pixel (ix-1, iy-1): this node is corner 3 (upper-right)
        // Triangle 1 contribution: w1*(inv_hx*gx + inv_hy*gy)
        if (ix > 0 && iy > 0) {
            Index_t grad_base = (ix - 1) * grad_stride_x + (iy - 1) * grad_stride_y;
            Real gx_t1 = gradient_input[grad_base + 0 * grad_stride_d + 1 * grad_stride_q];
            Real gy_t1 = gradient_input[grad_base + 1 * grad_stride_d + 1 * grad_stride_q];
            contrib += w1_inv_hx * gx_t1 + w1_inv_hy * gy_t1;
        }

        // Single write to this node - no atomics needed!
        Index_t nodal_idx = ix * nodal_stride_x + iy * nodal_stride_y;
        if (increment) {
            nodal_output[nodal_idx] += contrib;
        } else {
            nodal_output[nodal_idx] = contrib;
        }
    }
}

// =========================================================================
// 3D GPU Kernels
// =========================================================================

// Device-side copies of shape function gradients for 3D
// These are the same as in the header, made accessible to GPU kernels
__constant__ Real d_B_3D_REF[DIM_3D][NB_QUAD_3D][NB_NODES_3D] = {
    // d/dx gradients
    {
        {-1.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0},
        { 0.0, -1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0},
        { 0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  0.0},
        { 0.0, -1.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0},
        { 0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  1.0}
    },
    // d/dy gradients
    {
        {-1.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0},
        { 0.0,  0.0,  1.0,  0.0, -1.0,  0.0,  0.0,  0.0},
        { 0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  1.0,  0.0},
        { 0.0,  0.0, -1.0,  1.0,  0.0,  0.0,  0.0,  0.0},
        { 0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0,  1.0}
    },
    // d/dz gradients
    {
        {-1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0},
        { 0.0, -1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0},
        { 0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  1.0,  0.0},
        { 0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  0.0},
        { 0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  1.0}
    }
};

/**
 * GPU kernel for 3D FEM gradient operator with shared memory optimization.
 *
 * Computes gradient at 5 quadrature points per voxel from 8 nodal values.
 * Uses shared memory to reduce redundant global memory loads - adjacent
 * voxels share nodes, so we cooperatively load a tile of nodes.
 *
 * Hand-unrolled gradient computation exploits B matrix sparsity:
 * - Each gradient component has only 2 non-zero terms per quadrature point
 * - Avoids loop overhead and constant memory access latency
 */
__global__ void fem_gradient_3d_kernel(
    const Real* MUGRID_RESTRICT nodal_input,
    Real* MUGRID_RESTRICT gradient_output,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t nodal_stride_x, Index_t nodal_stride_y, Index_t nodal_stride_z,
    Index_t nodal_stride_n,
    Index_t grad_stride_x, Index_t grad_stride_y, Index_t grad_stride_z,
    Index_t grad_stride_q, Index_t grad_stride_d,
    Real inv_hx, Real inv_hy, Real inv_hz,
    bool increment) {

    // Shared memory for node values - needs (BLOCK+1)^3 to cover halo
    // BLOCK_SIZE_3D = 8, so we need 9x9x9 = 729 entries
    __shared__ Real s_nodes[BLOCK_SIZE_3D + 1][BLOCK_SIZE_3D + 1][BLOCK_SIZE_3D + 1];

    // Thread indices
    const Index_t tx = threadIdx.x;
    const Index_t ty = threadIdx.y;
    const Index_t tz = threadIdx.z;

    // Global voxel indices for this thread
    const Index_t ix = blockIdx.x * blockDim.x + tx;
    const Index_t iy = blockIdx.y * blockDim.y + ty;
    const Index_t iz = blockIdx.z * blockDim.z + tz;

    // Base index for this block's corner in global memory
    const Index_t block_base_x = blockIdx.x * blockDim.x;
    const Index_t block_base_y = blockIdx.y * blockDim.y;
    const Index_t block_base_z = blockIdx.z * blockDim.z;

    // Cooperatively load nodes into shared memory
    // Each thread loads one node at its position
    if (block_base_x + tx < nx && block_base_y + ty < ny && block_base_z + tz < nz) {
        s_nodes[tx][ty][tz] = nodal_input[
            (block_base_x + tx) * nodal_stride_x +
            (block_base_y + ty) * nodal_stride_y +
            (block_base_z + tz) * nodal_stride_z];
    }

    // Boundary threads load the extra halo nodes
    // X-edge: threads with tx == BLOCK_SIZE_3D-1 load the +1 in x
    if (tx == BLOCK_SIZE_3D - 1 && block_base_x + tx + 1 < nx &&
        block_base_y + ty < ny && block_base_z + tz < nz) {
        s_nodes[tx + 1][ty][tz] = nodal_input[
            (block_base_x + tx + 1) * nodal_stride_x +
            (block_base_y + ty) * nodal_stride_y +
            (block_base_z + tz) * nodal_stride_z];
    }
    // Y-edge
    if (ty == BLOCK_SIZE_3D - 1 && block_base_x + tx < nx &&
        block_base_y + ty + 1 < ny && block_base_z + tz < nz) {
        s_nodes[tx][ty + 1][tz] = nodal_input[
            (block_base_x + tx) * nodal_stride_x +
            (block_base_y + ty + 1) * nodal_stride_y +
            (block_base_z + tz) * nodal_stride_z];
    }
    // Z-edge
    if (tz == BLOCK_SIZE_3D - 1 && block_base_x + tx < nx &&
        block_base_y + ty < ny && block_base_z + tz + 1 < nz) {
        s_nodes[tx][ty][tz + 1] = nodal_input[
            (block_base_x + tx) * nodal_stride_x +
            (block_base_y + ty) * nodal_stride_y +
            (block_base_z + tz + 1) * nodal_stride_z];
    }
    // XY-edge
    if (tx == BLOCK_SIZE_3D - 1 && ty == BLOCK_SIZE_3D - 1 &&
        block_base_x + tx + 1 < nx && block_base_y + ty + 1 < ny &&
        block_base_z + tz < nz) {
        s_nodes[tx + 1][ty + 1][tz] = nodal_input[
            (block_base_x + tx + 1) * nodal_stride_x +
            (block_base_y + ty + 1) * nodal_stride_y +
            (block_base_z + tz) * nodal_stride_z];
    }
    // XZ-edge
    if (tx == BLOCK_SIZE_3D - 1 && tz == BLOCK_SIZE_3D - 1 &&
        block_base_x + tx + 1 < nx && block_base_y + ty < ny &&
        block_base_z + tz + 1 < nz) {
        s_nodes[tx + 1][ty][tz + 1] = nodal_input[
            (block_base_x + tx + 1) * nodal_stride_x +
            (block_base_y + ty) * nodal_stride_y +
            (block_base_z + tz + 1) * nodal_stride_z];
    }
    // YZ-edge
    if (ty == BLOCK_SIZE_3D - 1 && tz == BLOCK_SIZE_3D - 1 &&
        block_base_x + tx < nx && block_base_y + ty + 1 < ny &&
        block_base_z + tz + 1 < nz) {
        s_nodes[tx][ty + 1][tz + 1] = nodal_input[
            (block_base_x + tx) * nodal_stride_x +
            (block_base_y + ty + 1) * nodal_stride_y +
            (block_base_z + tz + 1) * nodal_stride_z];
    }
    // XYZ-corner
    if (tx == BLOCK_SIZE_3D - 1 && ty == BLOCK_SIZE_3D - 1 &&
        tz == BLOCK_SIZE_3D - 1 && block_base_x + tx + 1 < nx &&
        block_base_y + ty + 1 < ny && block_base_z + tz + 1 < nz) {
        s_nodes[tx + 1][ty + 1][tz + 1] = nodal_input[
            (block_base_x + tx + 1) * nodal_stride_x +
            (block_base_y + ty + 1) * nodal_stride_y +
            (block_base_z + tz + 1) * nodal_stride_z];
    }

    __syncthreads();

    // Check bounds for gradient computation (interior voxels only)
    if (ix < nx - 1 && iy < ny - 1 && iz < nz - 1) {
        // Get 8 nodal values from shared memory
        // Node layout: n0=(0,0,0), n1=(1,0,0), n2=(0,1,0), n3=(1,1,0),
        //              n4=(0,0,1), n5=(1,0,1), n6=(0,1,1), n7=(1,1,1)
        const Real n0 = s_nodes[tx][ty][tz];
        const Real n1 = s_nodes[tx + 1][ty][tz];
        const Real n2 = s_nodes[tx][ty + 1][tz];
        const Real n3 = s_nodes[tx + 1][ty + 1][tz];
        const Real n4 = s_nodes[tx][ty][tz + 1];
        const Real n5 = s_nodes[tx + 1][ty][tz + 1];
        const Real n6 = s_nodes[tx][ty + 1][tz + 1];
        const Real n7 = s_nodes[tx + 1][ty + 1][tz + 1];

        Index_t grad_base = ix * grad_stride_x +
                            iy * grad_stride_y +
                            iz * grad_stride_z;

        // Hand-unrolled gradient computation exploiting B matrix sparsity
        // Each quadrature point corresponds to a tetrahedron in the voxel
        // B matrix has only 2 non-zero entries per gradient component per quad

        // Quad 0: tetrahedron with nodes 0,1,2,4
        // grad_x = (-n0 + n1) / hx, grad_y = (-n0 + n2) / hy, grad_z = (-n0 + n4) / hz
        Real gx0 = inv_hx * (-n0 + n1);
        Real gy0 = inv_hy * (-n0 + n2);
        Real gz0 = inv_hz * (-n0 + n4);

        // Quad 1: tetrahedron with nodes 1,2,4,5
        // grad_x = (-n1 + n5) / hx, grad_y = (n2 - n4) / hy, grad_z = (-n1 + n5) / hz
        Real gx1 = inv_hx * (-n1 + n5);
        Real gy1 = inv_hy * (n2 - n4);
        Real gz1 = inv_hz * (-n1 + n5);

        // Quad 2: tetrahedron with nodes 2,4,5,6
        // grad_x = (n5 - n6) / hx, grad_y = (-n2 + n6) / hy, grad_z = (-n4 + n6) / hz
        Real gx2 = inv_hx * (n5 - n6);
        Real gy2 = inv_hy * (-n2 + n6);
        Real gz2 = inv_hz * (-n4 + n6);

        // Quad 3: tetrahedron with nodes 1,2,3,5
        // grad_x = (-n1 + n3) / hx, grad_y = (-n2 + n3) / hy, grad_z = (n5 - n6) / hz
        Real gx3 = inv_hx * (-n1 + n3);
        Real gy3 = inv_hy * (-n2 + n3);
        Real gz3 = inv_hz * (n5 - n6);

        // Quad 4: tetrahedron with nodes 2,3,5,7
        // grad_x = (-n3 + n7) / hx, grad_y = (-n2 + n7) / hy, grad_z = (-n5 + n7) / hz
        Real gx4 = inv_hx * (-n3 + n7);
        Real gy4 = inv_hy * (-n2 + n7);
        Real gz4 = inv_hz * (-n5 + n7);

        // Store gradients
        if (increment) {
            gradient_output[grad_base + 0 * grad_stride_q + 0 * grad_stride_d] += gx0;
            gradient_output[grad_base + 0 * grad_stride_q + 1 * grad_stride_d] += gy0;
            gradient_output[grad_base + 0 * grad_stride_q + 2 * grad_stride_d] += gz0;

            gradient_output[grad_base + 1 * grad_stride_q + 0 * grad_stride_d] += gx1;
            gradient_output[grad_base + 1 * grad_stride_q + 1 * grad_stride_d] += gy1;
            gradient_output[grad_base + 1 * grad_stride_q + 2 * grad_stride_d] += gz1;

            gradient_output[grad_base + 2 * grad_stride_q + 0 * grad_stride_d] += gx2;
            gradient_output[grad_base + 2 * grad_stride_q + 1 * grad_stride_d] += gy2;
            gradient_output[grad_base + 2 * grad_stride_q + 2 * grad_stride_d] += gz2;

            gradient_output[grad_base + 3 * grad_stride_q + 0 * grad_stride_d] += gx3;
            gradient_output[grad_base + 3 * grad_stride_q + 1 * grad_stride_d] += gy3;
            gradient_output[grad_base + 3 * grad_stride_q + 2 * grad_stride_d] += gz3;

            gradient_output[grad_base + 4 * grad_stride_q + 0 * grad_stride_d] += gx4;
            gradient_output[grad_base + 4 * grad_stride_q + 1 * grad_stride_d] += gy4;
            gradient_output[grad_base + 4 * grad_stride_q + 2 * grad_stride_d] += gz4;
        } else {
            gradient_output[grad_base + 0 * grad_stride_q + 0 * grad_stride_d] = gx0;
            gradient_output[grad_base + 0 * grad_stride_q + 1 * grad_stride_d] = gy0;
            gradient_output[grad_base + 0 * grad_stride_q + 2 * grad_stride_d] = gz0;

            gradient_output[grad_base + 1 * grad_stride_q + 0 * grad_stride_d] = gx1;
            gradient_output[grad_base + 1 * grad_stride_q + 1 * grad_stride_d] = gy1;
            gradient_output[grad_base + 1 * grad_stride_q + 2 * grad_stride_d] = gz1;

            gradient_output[grad_base + 2 * grad_stride_q + 0 * grad_stride_d] = gx2;
            gradient_output[grad_base + 2 * grad_stride_q + 1 * grad_stride_d] = gy2;
            gradient_output[grad_base + 2 * grad_stride_q + 2 * grad_stride_d] = gz2;

            gradient_output[grad_base + 3 * grad_stride_q + 0 * grad_stride_d] = gx3;
            gradient_output[grad_base + 3 * grad_stride_q + 1 * grad_stride_d] = gy3;
            gradient_output[grad_base + 3 * grad_stride_q + 2 * grad_stride_d] = gz3;

            gradient_output[grad_base + 4 * grad_stride_q + 0 * grad_stride_d] = gx4;
            gradient_output[grad_base + 4 * grad_stride_q + 1 * grad_stride_d] = gy4;
            gradient_output[grad_base + 4 * grad_stride_q + 2 * grad_stride_d] = gz4;
        }
    }
}

/**
 * GPU kernel for 3D FEM divergence (transpose) operator.
 *
 * Uses gather pattern: each thread handles one NODE and gathers contributions
 * from up to 8 adjacent voxels. This eliminates the need for atomic operations.
 *
 * For node at (ix, iy, iz), gather from adjacent voxels where this node
 * is at corner position node_idx:
 * - Voxel (ix, iy, iz): corner 0
 * - Voxel (ix-1, iy, iz): corner 1
 * - Voxel (ix, iy-1, iz): corner 2
 * - Voxel (ix-1, iy-1, iz): corner 3
 * - Voxel (ix, iy, iz-1): corner 4
 * - Voxel (ix-1, iy, iz-1): corner 5
 * - Voxel (ix, iy-1, iz-1): corner 6
 * - Voxel (ix-1, iy-1, iz-1): corner 7
 */
__global__ void fem_divergence_3d_kernel(
    const Real* MUGRID_RESTRICT gradient_input,
    Real* MUGRID_RESTRICT nodal_output,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t grad_stride_x, Index_t grad_stride_y, Index_t grad_stride_z,
    Index_t grad_stride_q, Index_t grad_stride_d,
    Index_t nodal_stride_x, Index_t nodal_stride_y, Index_t nodal_stride_z,
    Index_t nodal_stride_n,
    Real inv_hx, Real inv_hy, Real inv_hz,
    const Real* MUGRID_RESTRICT quad_weights,
    bool increment) {

    // Thread indices - each thread processes one NODE
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    Index_t iz = blockIdx.z * blockDim.z + threadIdx.z;

    // Check bounds (all nodes)
    if (ix < nx && iy < ny && iz < nz) {
        Real contrib = 0.0;

        // Iterate over all 8 potential adjacent voxels
        // Each voxel contributes if this node is one of its corners
        for (Index_t corner = 0; corner < NB_NODES_3D; ++corner) {
            // Compute voxel position based on which corner this node would be
            // Corner offsets: 0=(0,0,0), 1=(1,0,0), 2=(0,1,0), 3=(1,1,0),
            //                 4=(0,0,1), 5=(1,0,1), 6=(0,1,1), 7=(1,1,1)
            Index_t vx = ix - d_NODE_OFFSET_3D[corner][0];
            Index_t vy = iy - d_NODE_OFFSET_3D[corner][1];
            Index_t vz = iz - d_NODE_OFFSET_3D[corner][2];

            // Check if this voxel exists (valid interior voxel)
            if (vx >= 0 && vx < nx - 1 &&
                vy >= 0 && vy < ny - 1 &&
                vz >= 0 && vz < nz - 1) {

                Index_t grad_base = vx * grad_stride_x +
                                    vy * grad_stride_y +
                                    vz * grad_stride_z;

                // Gather B^T contributions from all quadrature points
                for (Index_t q = 0; q < NB_QUAD_3D; ++q) {
                    Index_t grad_idx = grad_base + q * grad_stride_q;
                    Real gx = gradient_input[grad_idx + 0 * grad_stride_d];
                    Real gy = gradient_input[grad_idx + 1 * grad_stride_d];
                    Real gz = gradient_input[grad_idx + 2 * grad_stride_d];

                    contrib += quad_weights[q] * (
                        d_B_3D_REF[0][q][corner] * inv_hx * gx +
                        d_B_3D_REF[1][q][corner] * inv_hy * gy +
                        d_B_3D_REF[2][q][corner] * inv_hz * gz);
                }
            }
        }

        // Single write to this node - no atomics needed!
        Index_t nodal_idx = ix * nodal_stride_x +
                            iy * nodal_stride_y +
                            iz * nodal_stride_z;
        if (increment) {
            nodal_output[nodal_idx] += contrib;
        } else {
            nodal_output[nodal_idx] = contrib;
        }
    }
}

// =========================================================================
// Launch Wrappers
// =========================================================================

// 2D Gradient - CUDA/HIP unified launch
#if defined(MUGRID_ENABLE_CUDA)
void fem_gradient_2d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
void fem_gradient_2d_hip(
#endif
    const Real* nodal_input,
    Real* gradient_output,
    Index_t nx, Index_t ny,
    Index_t nodal_stride_x, Index_t nodal_stride_y, Index_t nodal_stride_n,
    Index_t grad_stride_x, Index_t grad_stride_y,
    Index_t grad_stride_q, Index_t grad_stride_d,
    Real hx, Real hy,
    Real alpha,
    bool increment) {

    // Pre-compute scaled inverse grid spacing
    Real inv_hx = alpha / hx;
    Real inv_hy = alpha / hy;

    // Compute grid dimensions (for interior points)
    Index_t interior_nx = nx - 1;
    Index_t interior_ny = ny - 1;

    dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid(
        (interior_nx + block.x - 1) / block.x,
        (interior_ny + block.y - 1) / block.y
    );

    GPU_LAUNCH_KERNEL(fem_gradient_2d_kernel, grid, block,
        nodal_input, gradient_output, nx, ny,
        nodal_stride_x, nodal_stride_y, nodal_stride_n,
        grad_stride_x, grad_stride_y, grad_stride_q, grad_stride_d,
        inv_hx, inv_hy, increment);

    GPU_DEVICE_SYNCHRONIZE();
}

// 2D Divergence - CUDA/HIP unified launch
#if defined(MUGRID_ENABLE_CUDA)
void fem_divergence_2d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
void fem_divergence_2d_hip(
#endif
    const Real* gradient_input,
    Real* nodal_output,
    Index_t nx, Index_t ny,
    Index_t grad_stride_x, Index_t grad_stride_y,
    Index_t grad_stride_q, Index_t grad_stride_d,
    Index_t nodal_stride_x, Index_t nodal_stride_y, Index_t nodal_stride_n,
    Real hx, Real hy,
    const Real* quad_weights,
    Real alpha,
    bool increment) {

    // Pre-compute weighted inverse grid spacing
    Real w0_inv_hx = alpha * quad_weights[0] / hx;
    Real w0_inv_hy = alpha * quad_weights[0] / hy;
    Real w1_inv_hx = alpha * quad_weights[1] / hx;
    Real w1_inv_hy = alpha * quad_weights[1] / hy;

    // Grid covers ALL nodes (gather pattern: each thread writes one node)
    dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y
    );

    GPU_LAUNCH_KERNEL(fem_divergence_2d_kernel, grid, block,
        gradient_input, nodal_output, nx, ny,
        grad_stride_x, grad_stride_y, grad_stride_q, grad_stride_d,
        nodal_stride_x, nodal_stride_y, nodal_stride_n,
        w0_inv_hx, w0_inv_hy, w1_inv_hx, w1_inv_hy, increment);

    GPU_DEVICE_SYNCHRONIZE();
}

// 3D Gradient - CUDA/HIP unified launch
#if defined(MUGRID_ENABLE_CUDA)
void fem_gradient_3d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
void fem_gradient_3d_hip(
#endif
    const Real* nodal_input,
    Real* gradient_output,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t nodal_stride_x, Index_t nodal_stride_y, Index_t nodal_stride_z,
    Index_t nodal_stride_n,
    Index_t grad_stride_x, Index_t grad_stride_y, Index_t grad_stride_z,
    Index_t grad_stride_q, Index_t grad_stride_d,
    Real hx, Real hy, Real hz,
    Real alpha,
    bool increment) {

    Real inv_hx = alpha / hx;
    Real inv_hy = alpha / hy;
    Real inv_hz = alpha / hz;

    Index_t interior_nx = nx - 1;
    Index_t interior_ny = ny - 1;
    Index_t interior_nz = nz - 1;

    dim3 block(BLOCK_SIZE_3D, BLOCK_SIZE_3D, BLOCK_SIZE_3D);
    dim3 grid(
        (interior_nx + block.x - 1) / block.x,
        (interior_ny + block.y - 1) / block.y,
        (interior_nz + block.z - 1) / block.z
    );

    GPU_LAUNCH_KERNEL(fem_gradient_3d_kernel, grid, block,
        nodal_input, gradient_output, nx, ny, nz,
        nodal_stride_x, nodal_stride_y, nodal_stride_z, nodal_stride_n,
        grad_stride_x, grad_stride_y, grad_stride_z, grad_stride_q, grad_stride_d,
        inv_hx, inv_hy, inv_hz, increment);

    GPU_DEVICE_SYNCHRONIZE();
}

// 3D Divergence - CUDA/HIP unified launch
#if defined(MUGRID_ENABLE_CUDA)
void fem_divergence_3d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
void fem_divergence_3d_hip(
#endif
    const Real* gradient_input,
    Real* nodal_output,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t grad_stride_x, Index_t grad_stride_y, Index_t grad_stride_z,
    Index_t grad_stride_q, Index_t grad_stride_d,
    Index_t nodal_stride_x, Index_t nodal_stride_y, Index_t nodal_stride_z,
    Index_t nodal_stride_n,
    Real hx, Real hy, Real hz,
    const Real* quad_weights,
    Real alpha,
    bool increment) {

    Real inv_hx = alpha / hx;
    Real inv_hy = alpha / hy;
    Real inv_hz = alpha / hz;

    // Grid covers ALL nodes (gather pattern: each thread writes one node)
    dim3 block(BLOCK_SIZE_3D, BLOCK_SIZE_3D, BLOCK_SIZE_3D);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z
    );

    GPU_LAUNCH_KERNEL(fem_divergence_3d_kernel, grid, block,
        gradient_input, nodal_output, nx, ny, nz,
        grad_stride_x, grad_stride_y, grad_stride_z, grad_stride_q, grad_stride_d,
        nodal_stride_x, nodal_stride_y, nodal_stride_z, nodal_stride_n,
        inv_hx, inv_hy, inv_hz, quad_weights, increment);

    GPU_DEVICE_SYNCHRONIZE();
}

}  // namespace fem_gradient_kernels
}  // namespace muGrid
