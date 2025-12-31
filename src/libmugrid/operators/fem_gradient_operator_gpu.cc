/**
 * @file   fem_gradient_operator_gpu.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Dec 2025
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
    #define GPU_DEVICE_SYNCHRONIZE() (void)cudaDeviceSynchronize()
#elif defined(MUGRID_ENABLE_HIP)
    #include <hip/hip_runtime.h>
    #define GPU_LAUNCH_KERNEL(kernel, grid, block, ...) \
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, __VA_ARGS__)
    #define GPU_DEVICE_SYNCHRONIZE() (void)hipDeviceSynchronize()
#endif

namespace muGrid {
namespace fem_gradient_kernels {

// Block sizes for GPU kernels
constexpr int BLOCK_SIZE_2D = 16;
constexpr int BLOCK_SIZE_3D = 8;

// Custom atomicAdd for double precision (needed for shared memory on some architectures)
// Uses compare-and-swap loop
__device__ inline double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

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
// 5-tet decomposition: 1 central tetrahedron + 4 corner tetrahedra
__constant__ Real d_B_3D_REF[DIM_3D][NB_QUAD_3D][NB_NODES_3D] = {
    // d/dx gradients
    {
        { 0.0,  0.5, -0.5,  0.0, -0.5,  0.0,  0.0,  0.5},  // Tet 0: Central
        {-1.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0},  // Tet 1: Corner (0,0,0)
        { 0.0,  0.0, -1.0,  1.0,  0.0,  0.0,  0.0,  0.0},  // Tet 2: Corner (1,1,0)
        { 0.0,  0.0,  0.0,  0.0, -1.0,  1.0,  0.0,  0.0},  // Tet 3: Corner (1,0,1)
        { 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  1.0}   // Tet 4: Corner (0,1,1)
    },
    // d/dy gradients
    {
        { 0.0, -0.5,  0.5,  0.0, -0.5,  0.0,  0.0,  0.5},  // Tet 0: Central
        {-1.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0},  // Tet 1: Corner (0,0,0)
        { 0.0, -1.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0},  // Tet 2: Corner (1,1,0)
        { 0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  1.0},  // Tet 3: Corner (1,0,1)
        { 0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  1.0,  0.0}   // Tet 4: Corner (0,1,1)
    },
    // d/dz gradients
    {
        { 0.0, -0.5, -0.5,  0.0,  0.5,  0.0,  0.0,  0.5},  // Tet 0: Central
        {-1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0},  // Tet 1: Corner (0,0,0)
        { 0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  1.0},  // Tet 2: Corner (1,1,0)
        { 0.0, -1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0},  // Tet 3: Corner (1,0,1)
        { 0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  1.0,  0.0}   // Tet 4: Corner (0,1,1)
    }
};

// Node corner offsets for 3D - used by divergence kernel to map corner index
// to voxel offset (needed to implement gather pattern)
__constant__ Index_t d_NODE_OFFSET_3D[NB_NODES_3D][DIM_3D] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
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

        // Hand-unrolled gradient computation for 5-tet decomposition
        // Tet 0: Central tetrahedron (nodes 1,2,4,7) - uses all 4 vertices
        // Tet 1-4: Corner tetrahedra - use simple forward differences

        // Quad 0: Central tetrahedron (nodes 1,2,4,7)
        // grad_x = 0.5*(n1 - n2 - n4 + n7) / hx
        // grad_y = 0.5*(-n1 + n2 - n4 + n7) / hy
        // grad_z = 0.5*(-n1 - n2 + n4 + n7) / hz
        Real gx0 = inv_hx * 0.5 * (n1 - n2 - n4 + n7);
        Real gy0 = inv_hy * 0.5 * (-n1 + n2 - n4 + n7);
        Real gz0 = inv_hz * 0.5 * (-n1 - n2 + n4 + n7);

        // Quad 1: Corner at (0,0,0) - nodes 0,1,2,4
        // grad_x = (-n0 + n1) / hx, grad_y = (-n0 + n2) / hy, grad_z = (-n0 + n4) / hz
        Real gx1 = inv_hx * (-n0 + n1);
        Real gy1 = inv_hy * (-n0 + n2);
        Real gz1 = inv_hz * (-n0 + n4);

        // Quad 2: Corner at (1,1,0) - nodes 1,2,3,7
        // grad_x = (-n2 + n3) / hx, grad_y = (-n1 + n3) / hy, grad_z = (-n3 + n7) / hz
        Real gx2 = inv_hx * (-n2 + n3);
        Real gy2 = inv_hy * (-n1 + n3);
        Real gz2 = inv_hz * (-n3 + n7);

        // Quad 3: Corner at (1,0,1) - nodes 1,4,5,7
        // grad_x = (-n4 + n5) / hx, grad_y = (-n5 + n7) / hy, grad_z = (-n1 + n5) / hz
        Real gx3 = inv_hx * (-n4 + n5);
        Real gy3 = inv_hy * (-n5 + n7);
        Real gz3 = inv_hz * (-n1 + n5);

        // Quad 4: Corner at (0,1,1) - nodes 2,4,6,7
        // grad_x = (-n6 + n7) / hx, grad_y = (-n4 + n6) / hy, grad_z = (-n2 + n6) / hz
        Real gx4 = inv_hx * (-n6 + n7);
        Real gy4 = inv_hy * (-n4 + n6);
        Real gz4 = inv_hz * (-n2 + n6);

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
 * GPU kernel for 3D FEM divergence (transpose) operator - GATHER version.
 *
 * Uses gather pattern - NO ATOMICS, NO SHARED MEMORY:
 * - Each thread handles one NODE
 * - Gathers contributions from up to 8 neighboring voxels
 * - Single write per node
 *
 * This kernel relies on L2 cache for data reuse between adjacent threads.
 * Adjacent threads in x access adjacent voxels, which have adjacent memory
 * addresses (x stride = 1).
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
    const Index_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const Index_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    const Index_t iz = blockIdx.z * blockDim.z + threadIdx.z;

    // Check bounds (all nodes)
    if (ix >= nx || iy >= ny || iz >= nz) return;

    // Load weights into registers
    const Real w0 = quad_weights[0];
    const Real w1 = quad_weights[1];
    const Real w2 = quad_weights[2];
    const Real w3 = quad_weights[3];
    const Real w4 = quad_weights[4];

    // Pre-compute weighted inverse spacings
    const Real w0_ihx = w0 * inv_hx, w0_ihy = w0 * inv_hy, w0_ihz = w0 * inv_hz;
    const Real w1_ihx = w1 * inv_hx, w1_ihy = w1 * inv_hy, w1_ihz = w1 * inv_hz;
    const Real w2_ihx = w2 * inv_hx, w2_ihy = w2 * inv_hy, w2_ihz = w2 * inv_hz;
    const Real w3_ihx = w3 * inv_hx, w3_ihy = w3 * inv_hy, w3_ihz = w3 * inv_hz;
    const Real w4_ihx = w4 * inv_hx, w4_ihy = w4 * inv_hy, w4_ihz = w4 * inv_hz;

    Real contrib = 0.0;

    // 5-tet decomposition: Tet 0 = central, Tet 1-4 = corners
    // For each voxel, we check which tetrahedra this node belongs to
    // and accumulate the B^T contributions.
    //
    // Tetrahedra membership:
    //   Tet 0 (central): nodes 1, 2, 4, 7
    //   Tet 1 (corner 0,0,0): nodes 0, 1, 2, 4
    //   Tet 2 (corner 1,1,0): nodes 1, 2, 3, 7
    //   Tet 3 (corner 1,0,1): nodes 1, 4, 5, 7
    //   Tet 4 (corner 0,1,1): nodes 2, 4, 6, 7

    // Voxel (ix, iy, iz): this node is corner 0
    // Only in Tet 1: B^T = w1*(-gx/hx - gy/hy - gz/hz)
    if (ix < nx - 1 && iy < ny - 1 && iz < nz - 1) {
        Index_t grad_base = ix * grad_stride_x + iy * grad_stride_y + iz * grad_stride_z;
        Real gx_q1 = gradient_input[grad_base + 1 * grad_stride_q + 0 * grad_stride_d];
        Real gy_q1 = gradient_input[grad_base + 1 * grad_stride_q + 1 * grad_stride_d];
        Real gz_q1 = gradient_input[grad_base + 1 * grad_stride_q + 2 * grad_stride_d];
        contrib += -w1_ihx * gx_q1 - w1_ihy * gy_q1 - w1_ihz * gz_q1;
    }

    // Voxel (ix-1, iy, iz): this node is corner 1
    // In Tet 0: +0.5*gx - 0.5*gy - 0.5*gz
    // In Tet 1: +gx
    // In Tet 2: -gy
    // In Tet 3: -gz
    if (ix > 0 && iy < ny - 1 && iz < nz - 1) {
        Index_t grad_base = (ix - 1) * grad_stride_x + iy * grad_stride_y + iz * grad_stride_z;
        Real gx_q0 = gradient_input[grad_base + 0 * grad_stride_q + 0 * grad_stride_d];
        Real gy_q0 = gradient_input[grad_base + 0 * grad_stride_q + 1 * grad_stride_d];
        Real gz_q0 = gradient_input[grad_base + 0 * grad_stride_q + 2 * grad_stride_d];
        Real gx_q1 = gradient_input[grad_base + 1 * grad_stride_q + 0 * grad_stride_d];
        Real gy_q2 = gradient_input[grad_base + 2 * grad_stride_q + 1 * grad_stride_d];
        Real gz_q3 = gradient_input[grad_base + 3 * grad_stride_q + 2 * grad_stride_d];
        contrib += w0_ihx * 0.5 * gx_q0 - w0_ihy * 0.5 * gy_q0 - w0_ihz * 0.5 * gz_q0
                 + w1_ihx * gx_q1 - w2_ihy * gy_q2 - w3_ihz * gz_q3;
    }

    // Voxel (ix, iy-1, iz): this node is corner 2
    // In Tet 0: -0.5*gx + 0.5*gy - 0.5*gz
    // In Tet 1: +gy
    // In Tet 2: -gx
    // In Tet 4: -gz
    if (ix < nx - 1 && iy > 0 && iz < nz - 1) {
        Index_t grad_base = ix * grad_stride_x + (iy - 1) * grad_stride_y + iz * grad_stride_z;
        Real gx_q0 = gradient_input[grad_base + 0 * grad_stride_q + 0 * grad_stride_d];
        Real gy_q0 = gradient_input[grad_base + 0 * grad_stride_q + 1 * grad_stride_d];
        Real gz_q0 = gradient_input[grad_base + 0 * grad_stride_q + 2 * grad_stride_d];
        Real gy_q1 = gradient_input[grad_base + 1 * grad_stride_q + 1 * grad_stride_d];
        Real gx_q2 = gradient_input[grad_base + 2 * grad_stride_q + 0 * grad_stride_d];
        Real gz_q4 = gradient_input[grad_base + 4 * grad_stride_q + 2 * grad_stride_d];
        contrib += -w0_ihx * 0.5 * gx_q0 + w0_ihy * 0.5 * gy_q0 - w0_ihz * 0.5 * gz_q0
                 + w1_ihy * gy_q1 - w2_ihx * gx_q2 - w4_ihz * gz_q4;
    }

    // Voxel (ix-1, iy-1, iz): this node is corner 3
    // Only in Tet 2: +gx + gy - gz
    if (ix > 0 && iy > 0 && iz < nz - 1) {
        Index_t grad_base = (ix - 1) * grad_stride_x + (iy - 1) * grad_stride_y + iz * grad_stride_z;
        Real gx_q2 = gradient_input[grad_base + 2 * grad_stride_q + 0 * grad_stride_d];
        Real gy_q2 = gradient_input[grad_base + 2 * grad_stride_q + 1 * grad_stride_d];
        Real gz_q2 = gradient_input[grad_base + 2 * grad_stride_q + 2 * grad_stride_d];
        contrib += w2_ihx * gx_q2 + w2_ihy * gy_q2 - w2_ihz * gz_q2;
    }

    // Voxel (ix, iy, iz-1): this node is corner 4
    // In Tet 0: -0.5*gx - 0.5*gy + 0.5*gz
    // In Tet 1: +gz
    // In Tet 3: -gx
    // In Tet 4: -gy
    if (ix < nx - 1 && iy < ny - 1 && iz > 0) {
        Index_t grad_base = ix * grad_stride_x + iy * grad_stride_y + (iz - 1) * grad_stride_z;
        Real gx_q0 = gradient_input[grad_base + 0 * grad_stride_q + 0 * grad_stride_d];
        Real gy_q0 = gradient_input[grad_base + 0 * grad_stride_q + 1 * grad_stride_d];
        Real gz_q0 = gradient_input[grad_base + 0 * grad_stride_q + 2 * grad_stride_d];
        Real gz_q1 = gradient_input[grad_base + 1 * grad_stride_q + 2 * grad_stride_d];
        Real gx_q3 = gradient_input[grad_base + 3 * grad_stride_q + 0 * grad_stride_d];
        Real gy_q4 = gradient_input[grad_base + 4 * grad_stride_q + 1 * grad_stride_d];
        contrib += -w0_ihx * 0.5 * gx_q0 - w0_ihy * 0.5 * gy_q0 + w0_ihz * 0.5 * gz_q0
                 + w1_ihz * gz_q1 - w3_ihx * gx_q3 - w4_ihy * gy_q4;
    }

    // Voxel (ix-1, iy, iz-1): this node is corner 5
    // Only in Tet 3: +gx - gy + gz
    if (ix > 0 && iy < ny - 1 && iz > 0) {
        Index_t grad_base = (ix - 1) * grad_stride_x + iy * grad_stride_y + (iz - 1) * grad_stride_z;
        Real gx_q3 = gradient_input[grad_base + 3 * grad_stride_q + 0 * grad_stride_d];
        Real gy_q3 = gradient_input[grad_base + 3 * grad_stride_q + 1 * grad_stride_d];
        Real gz_q3 = gradient_input[grad_base + 3 * grad_stride_q + 2 * grad_stride_d];
        contrib += w3_ihx * gx_q3 - w3_ihy * gy_q3 + w3_ihz * gz_q3;
    }

    // Voxel (ix, iy-1, iz-1): this node is corner 6
    // Only in Tet 4: -gx + gy + gz
    if (ix < nx - 1 && iy > 0 && iz > 0) {
        Index_t grad_base = ix * grad_stride_x + (iy - 1) * grad_stride_y + (iz - 1) * grad_stride_z;
        Real gx_q4 = gradient_input[grad_base + 4 * grad_stride_q + 0 * grad_stride_d];
        Real gy_q4 = gradient_input[grad_base + 4 * grad_stride_q + 1 * grad_stride_d];
        Real gz_q4 = gradient_input[grad_base + 4 * grad_stride_q + 2 * grad_stride_d];
        contrib += -w4_ihx * gx_q4 + w4_ihy * gy_q4 + w4_ihz * gz_q4;
    }

    // Voxel (ix-1, iy-1, iz-1): this node is corner 7
    // In Tet 0: +0.5*gx + 0.5*gy + 0.5*gz
    // In Tet 2: +gz
    // In Tet 3: +gy
    // In Tet 4: +gx
    if (ix > 0 && iy > 0 && iz > 0) {
        Index_t grad_base = (ix - 1) * grad_stride_x + (iy - 1) * grad_stride_y + (iz - 1) * grad_stride_z;
        Real gx_q0 = gradient_input[grad_base + 0 * grad_stride_q + 0 * grad_stride_d];
        Real gy_q0 = gradient_input[grad_base + 0 * grad_stride_q + 1 * grad_stride_d];
        Real gz_q0 = gradient_input[grad_base + 0 * grad_stride_q + 2 * grad_stride_d];
        Real gz_q2 = gradient_input[grad_base + 2 * grad_stride_q + 2 * grad_stride_d];
        Real gy_q3 = gradient_input[grad_base + 3 * grad_stride_q + 1 * grad_stride_d];
        Real gx_q4 = gradient_input[grad_base + 4 * grad_stride_q + 0 * grad_stride_d];
        contrib += w0_ihx * 0.5 * gx_q0 + w0_ihy * 0.5 * gy_q0 + w0_ihz * 0.5 * gz_q0
                 + w2_ihz * gz_q2 + w3_ihy * gy_q3 + w4_ihx * gx_q4;
    }

    // Single write to this node - NO ATOMICS NEEDED!
    Index_t nodal_idx = ix * nodal_stride_x + iy * nodal_stride_y + iz * nodal_stride_z;
    if (increment) {
        nodal_output[nodal_idx] += contrib;
    } else {
        nodal_output[nodal_idx] = contrib;
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
