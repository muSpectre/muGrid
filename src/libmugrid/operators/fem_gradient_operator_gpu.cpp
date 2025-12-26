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
// Portable atomicAdd for double precision
// =========================================================================
// CUDA has native atomicAdd for double starting with compute capability 6.0
// (Pascal). For older architectures, we need to use atomicCAS.
// This implementation works on all architectures using the CAS fallback.
// =========================================================================

__device__ inline double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
#if defined(MUGRID_ENABLE_CUDA)
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
#elif defined(MUGRID_ENABLE_HIP)
        old = atomicCAS(address_as_ull, assumed,
                        __builtin_bit_cast(unsigned long long int, val + __builtin_bit_cast(double, assumed)));
#endif
    } while (assumed != old);
#if defined(MUGRID_ENABLE_CUDA)
    return __longlong_as_double(old);
#elif defined(MUGRID_ENABLE_HIP)
    return __builtin_bit_cast(double, old);
#endif
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
 * Uses atomic operations to accumulate contributions from quadrature points
 * to shared nodal points.
 */
__global__ void fem_divergence_2d_kernel(
    const Real* MUGRID_RESTRICT gradient_input,
    Real* MUGRID_RESTRICT nodal_output,
    Index_t nx, Index_t ny,
    Index_t grad_stride_x, Index_t grad_stride_y,
    Index_t grad_stride_q, Index_t grad_stride_d,
    Index_t nodal_stride_x, Index_t nodal_stride_y, Index_t nodal_stride_n,
    Real w0_inv_hx, Real w0_inv_hy,
    Real w1_inv_hx, Real w1_inv_hy) {

    // Thread indices - each thread processes one pixel
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (ix < nx - 1 && iy < ny - 1) {
        Index_t grad_base = ix * grad_stride_x + iy * grad_stride_y;
        Index_t nodal_base = ix * nodal_stride_x + iy * nodal_stride_y;

        // Get gradient values at quadrature points
        Real gx_t0 = gradient_input[grad_base + 0 * grad_stride_d + 0 * grad_stride_q];
        Real gy_t0 = gradient_input[grad_base + 1 * grad_stride_d + 0 * grad_stride_q];
        Real gx_t1 = gradient_input[grad_base + 0 * grad_stride_d + 1 * grad_stride_q];
        Real gy_t1 = gradient_input[grad_base + 1 * grad_stride_d + 1 * grad_stride_q];

        // Triangle 0 contributions: B^T * sigma
        Real contrib_n0_t0 = w0_inv_hx * (-gx_t0) + w0_inv_hy * (-gy_t0);
        Real contrib_n1_t0 = w0_inv_hx * (gx_t0);
        Real contrib_n2_t0 = w0_inv_hy * (gy_t0);

        // Triangle 1 contributions:
        Real contrib_n1_t1 = w1_inv_hy * (-gy_t1);
        Real contrib_n2_t1 = w1_inv_hx * (-gx_t1);
        Real contrib_n3_t1 = w1_inv_hx * (gx_t1) + w1_inv_hy * (gy_t1);

        // Accumulate to nodal points using atomics (nodes are shared between pixels)
        atomicAddDouble(&nodal_output[nodal_base], contrib_n0_t0);
        atomicAddDouble(&nodal_output[nodal_base + nodal_stride_x], contrib_n1_t0 + contrib_n1_t1);
        atomicAddDouble(&nodal_output[nodal_base + nodal_stride_y], contrib_n2_t0 + contrib_n2_t1);
        atomicAddDouble(&nodal_output[nodal_base + nodal_stride_x + nodal_stride_y], contrib_n3_t1);
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

// Node offsets for 3D [node][dim]
__constant__ Index_t d_NODE_OFFSET_3D[NB_NODES_3D][DIM_3D] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
};

/**
 * GPU kernel for 3D FEM gradient operator.
 *
 * Computes gradient at 5 quadrature points per voxel from 8 nodal values.
 * Each thread processes one voxel.
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

    // Thread indices - each thread processes one voxel
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    Index_t iz = blockIdx.z * blockDim.z + threadIdx.z;

    // Check bounds
    if (ix < nx - 1 && iy < ny - 1 && iz < nz - 1) {
        Index_t nodal_base = ix * nodal_stride_x +
                             iy * nodal_stride_y +
                             iz * nodal_stride_z;
        Index_t grad_base = ix * grad_stride_x +
                            iy * grad_stride_y +
                            iz * grad_stride_z;

        // Get all 8 nodal values
        Real n[NB_NODES_3D];
        for (Index_t node = 0; node < NB_NODES_3D; ++node) {
            Index_t ox = d_NODE_OFFSET_3D[node][0];
            Index_t oy = d_NODE_OFFSET_3D[node][1];
            Index_t oz = d_NODE_OFFSET_3D[node][2];
            n[node] = nodal_input[nodal_base +
                                  ox * nodal_stride_x +
                                  oy * nodal_stride_y +
                                  oz * nodal_stride_z];
        }

        // Compute gradients for each tetrahedron (quadrature point)
        for (Index_t q = 0; q < NB_QUAD_3D; ++q) {
            Real grad_x = 0.0, grad_y = 0.0, grad_z = 0.0;
            for (Index_t node = 0; node < NB_NODES_3D; ++node) {
                grad_x += d_B_3D_REF[0][q][node] * n[node];
                grad_y += d_B_3D_REF[1][q][node] * n[node];
                grad_z += d_B_3D_REF[2][q][node] * n[node];
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

/**
 * GPU kernel for 3D FEM divergence (transpose) operator.
 *
 * Uses atomic operations to accumulate contributions.
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
    const Real* MUGRID_RESTRICT quad_weights) {

    // Thread indices
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    Index_t iz = blockIdx.z * blockDim.z + threadIdx.z;

    // Check bounds
    if (ix < nx - 1 && iy < ny - 1 && iz < nz - 1) {
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
            for (Index_t node = 0; node < NB_NODES_3D; ++node) {
                Real contrib = w * (d_B_3D_REF[0][q][node] * inv_hx * gx +
                                    d_B_3D_REF[1][q][node] * inv_hy * gy +
                                    d_B_3D_REF[2][q][node] * inv_hz * gz);
                if (contrib != 0.0) {
                    Index_t ox = d_NODE_OFFSET_3D[node][0];
                    Index_t oy = d_NODE_OFFSET_3D[node][1];
                    Index_t oz = d_NODE_OFFSET_3D[node][2];
                    atomicAddDouble(&nodal_output[nodal_base +
                                                  ox * nodal_stride_x +
                                                  oy * nodal_stride_y +
                                                  oz * nodal_stride_z], contrib);
                }
            }
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

    // If not incrementing, we need to zero the output first
    // The kernel uses atomic adds, so we can't just overwrite
    if (!increment) {
        Index_t total_size = nx * ny;
#if defined(MUGRID_ENABLE_CUDA)
        cudaMemset(nodal_output, 0, total_size * sizeof(Real));
#elif defined(MUGRID_ENABLE_HIP)
        hipMemset(nodal_output, 0, total_size * sizeof(Real));
#endif
    }

    // Pre-compute weighted inverse grid spacing
    Real w0_inv_hx = alpha * quad_weights[0] / hx;
    Real w0_inv_hy = alpha * quad_weights[0] / hy;
    Real w1_inv_hx = alpha * quad_weights[1] / hx;
    Real w1_inv_hy = alpha * quad_weights[1] / hy;

    Index_t interior_nx = nx - 1;
    Index_t interior_ny = ny - 1;

    dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid(
        (interior_nx + block.x - 1) / block.x,
        (interior_ny + block.y - 1) / block.y
    );

    GPU_LAUNCH_KERNEL(fem_divergence_2d_kernel, grid, block,
        gradient_input, nodal_output, nx, ny,
        grad_stride_x, grad_stride_y, grad_stride_q, grad_stride_d,
        nodal_stride_x, nodal_stride_y, nodal_stride_n,
        w0_inv_hx, w0_inv_hy, w1_inv_hx, w1_inv_hy);

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

    // Zero output if not incrementing
    if (!increment) {
        Index_t total_size = nx * ny * nz;
#if defined(MUGRID_ENABLE_CUDA)
        cudaMemset(nodal_output, 0, total_size * sizeof(Real));
#elif defined(MUGRID_ENABLE_HIP)
        hipMemset(nodal_output, 0, total_size * sizeof(Real));
#endif
    }

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

    GPU_LAUNCH_KERNEL(fem_divergence_3d_kernel, grid, block,
        gradient_input, nodal_output, nx, ny, nz,
        grad_stride_x, grad_stride_y, grad_stride_z, grad_stride_q, grad_stride_d,
        nodal_stride_x, nodal_stride_y, nodal_stride_z, nodal_stride_n,
        inv_hx, inv_hy, inv_hz, quad_weights);

    GPU_DEVICE_SYNCHRONIZE();
}

}  // namespace fem_gradient_kernels
}  // namespace muGrid
