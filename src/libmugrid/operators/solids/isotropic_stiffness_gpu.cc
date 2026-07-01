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

#include "isotropic_stiffness.hh"
#include "collection/field_collection_global.hh"
#include "core/exception.hh"

#include "memory/gpu_runtime.hh"

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

// Macro-RHS constant per-element vectors Gu = G u*, Vu = V u* (uploaded per
// call, since they depend on E_macro). 8 entries in 2D, 24 in 3D.
__constant__ Real d_Gu_2D[8];
__constant__ Real d_Vu_2D[8];
__constant__ Real d_Gu_3D[24];
__constant__ Real d_Vu_3D[24];

// Stress-average constants: element-averaged gradient operator Dbar and the
// (symmetric) macro strain E_macro (uploaded per call).
__constant__ Real d_Dbar_2D[8];   // [dim][nb_nodes] = 2*4
__constant__ Real d_Dbar_3D[24];  // [dim][nb_nodes] = 3*8
__constant__ Real d_Emacro_2D[4];
__constant__ Real d_Emacro_3D[9];

// Portable double-precision atomic add. CUDA provides a native atomicAdd for
// double only on sm_60+; the TITAN X (Maxwell, sm_52) and any HIP target use
// the standard compare-and-swap fallback.
__device__ inline double mugrid_atomic_add_double(double * address,
                                                  double val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    return atomicAdd(address, val);
#else
    unsigned long long * addr_ull =
        reinterpret_cast<unsigned long long *>(address);
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
                        __double_as_longlong(
                            val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

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
template <typename T, bool Uniform>
__global__ void isotropic_stiffness_2d_kernel(
    const T* __restrict__ displacement,
    const T* __restrict__ lambda,
    const T* __restrict__ mu,
    T* __restrict__ force,
    Index_t nnx, Index_t nny,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_d,
    Index_t mat_stride_x, Index_t mat_stride_y,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_d,
    T alpha, bool increment, T lam_u, T mu_u) {

    constexpr int NB_NODES = 4;
    constexpr int NB_DOFS = 2;
    constexpr int NB_ELEM_DOFS = NB_NODES * NB_DOFS;

    // Thread indexing for NODES - iterate over all interior nodes
    // Ghost cells handle periodicity and MPI boundaries
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (ix >= nnx || iy >= nny) return;

    // Neighboring element offsets and corresponding local node index
    const int ELEM_OFFSETS[4][3] = {
        {-1, -1, 3},  // Element (ix-1, iy-1): this node is local node 3
        { 0, -1, 2},  // Element (ix,   iy-1): this node is local node 2
        {-1,  0, 1},  // Element (ix-1, iy  ): this node is local node 1
        { 0,  0, 0}   // Element (ix,   iy  ): this node is local node 0
    };

    // Accumulate force for this node
    T f[NB_DOFS] = {T(0), T(0)};

    // Loop over neighboring elements (all guaranteed to exist for this node)
    #pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
        int ex = static_cast<int>(ix) + ELEM_OFFSETS[elem][0];
        int ey = static_cast<int>(iy) + ELEM_OFFSETS[elem][1];
        int local_node = ELEM_OFFSETS[elem][2];

        // Get material parameters for this element
        T lam, mu_val;
        if constexpr (Uniform) {
            lam = lam_u;
            mu_val = mu_u;
        } else {
            lam = lambda[ex * mat_stride_x + ey * mat_stride_y];
            mu_val = mu[ex * mat_stride_x + ey * mat_stride_y];
        }

        // Gather displacements from all 4 nodes of this element
        T u[NB_ELEM_DOFS];
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

        // Compute only the rows of K @ u that correspond to this node.
        // The geometry stays in double __constant__ memory; cast each entry to
        // T at load so the inner product runs in working precision T.
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d) {
            int row = local_node * NB_DOFS + d;
            T contrib = T(0);
            #pragma unroll
            for (int j = 0; j < NB_ELEM_DOFS; ++j) {
                contrib += (static_cast<T>(2) * mu_val *
                                static_cast<T>(d_G_2D[row * NB_ELEM_DOFS + j]) +
                            lam *
                                static_cast<T>(d_V_2D[row * NB_ELEM_DOFS + j])) *
                           u[j];
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
template <typename T, bool Uniform>
__global__ void isotropic_stiffness_3d_kernel(
    const T* __restrict__ displacement,
    const T* __restrict__ lambda,
    const T* __restrict__ mu,
    T* __restrict__ force,
    Index_t nnx, Index_t nny, Index_t nnz,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_z,
    Index_t disp_stride_d,
    Index_t mat_stride_x, Index_t mat_stride_y, Index_t mat_stride_z,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_z,
    Index_t force_stride_d,
    T alpha, bool increment, T lam_u, T mu_u) {

    constexpr int NB_NODES = 8;
    constexpr int NB_DOFS = 3;
    constexpr int NB_ELEM_DOFS = NB_NODES * NB_DOFS;

    // Thread indexing for NODES - iterate over all interior nodes
    // Ghost cells handle periodicity and MPI boundaries
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    Index_t iz = blockIdx.z * blockDim.z + threadIdx.z;

    // Check bounds
    if (ix >= nnx || iy >= nny || iz >= nnz) return;

    // Neighboring element offsets and corresponding local node index
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
    T f[NB_DOFS] = {T(0), T(0), T(0)};

    // Loop over neighboring elements (all guaranteed to exist for this node)
    #pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
        int ex = static_cast<int>(ix) + ELEM_OFFSETS[elem][0];
        int ey = static_cast<int>(iy) + ELEM_OFFSETS[elem][1];
        int ez = static_cast<int>(iz) + ELEM_OFFSETS[elem][2];
        int local_node = ELEM_OFFSETS[elem][3];

        // Get material parameters for this element
        T lam, mu_val;
        if constexpr (Uniform) {
            lam = lam_u;
            mu_val = mu_u;
        } else {
            Index_t mat_idx =
                ex * mat_stride_x + ey * mat_stride_y + ez * mat_stride_z;
            lam = lambda[mat_idx];
            mu_val = mu[mat_idx];
        }

        // Gather displacements from all 8 nodes of this element
        T u[NB_ELEM_DOFS];
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

        // Compute only the rows of K @ u that correspond to this node; cast the
        // double __constant__ geometry to T at load (see the 2D kernel).
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d) {
            int row = local_node * NB_DOFS + d;
            T contrib = T(0);
            #pragma unroll
            for (int j = 0; j < NB_ELEM_DOFS; ++j) {
                contrib += (static_cast<T>(2) * mu_val *
                                static_cast<T>(d_G_3D[row * NB_ELEM_DOFS + j]) +
                            lam *
                                static_cast<T>(d_V_3D[row * NB_ELEM_DOFS + j])) *
                           u[j];
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
// Macro-RHS GPU Kernels (Gather Pattern - No Atomics)
// ============================================================================

// 2D: assemble force = B^T C E_macro = K @ u* with the affine u* folded into
// the constant per-element vectors d_Gu_2D, d_Vu_2D. One thread per node.
template <typename T>
__global__ void isotropic_stiffness_2d_macro_rhs_kernel(
    const T * __restrict__ lambda, const T * __restrict__ mu,
    T * __restrict__ force, Index_t nnx, Index_t nny, Index_t mat_stride_x,
    Index_t mat_stride_y, Index_t force_stride_x, Index_t force_stride_y,
    Index_t force_stride_d, T alpha, bool increment) {

    constexpr int NB_DOFS = 2;
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nnx || iy >= nny) return;

    const int ELEM_OFFSETS[4][3] = {
        {-1, -1, 3}, {0, -1, 2}, {-1, 0, 1}, {0, 0, 0}};

    T f[NB_DOFS] = {T(0), T(0)};
    #pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
        int ex = static_cast<int>(ix) + ELEM_OFFSETS[elem][0];
        int ey = static_cast<int>(iy) + ELEM_OFFSETS[elem][1];
        int local_node = ELEM_OFFSETS[elem][2];
        T lam = lambda[ex * mat_stride_x + ey * mat_stride_y];
        T mu_val = mu[ex * mat_stride_x + ey * mat_stride_y];
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d) {
            int row = local_node * NB_DOFS + d;
            f[d] += static_cast<T>(2) * mu_val * static_cast<T>(d_Gu_2D[row]) +
                    lam * static_cast<T>(d_Vu_2D[row]);
        }
    }
    Index_t base = ix * force_stride_x + iy * force_stride_y;
    if (increment) {
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d)
            force[base + d * force_stride_d] += alpha * f[d];
    } else {
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d)
            force[base + d * force_stride_d] = alpha * f[d];
    }
}

// 3D macro RHS (see 2D variant).
template <typename T>
__global__ void isotropic_stiffness_3d_macro_rhs_kernel(
    const T * __restrict__ lambda, const T * __restrict__ mu,
    T * __restrict__ force, Index_t nnx, Index_t nny, Index_t nnz,
    Index_t mat_stride_x, Index_t mat_stride_y, Index_t mat_stride_z,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_z,
    Index_t force_stride_d, T alpha, bool increment) {

    constexpr int NB_DOFS = 3;
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    Index_t iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= nnx || iy >= nny || iz >= nnz) return;

    const int ELEM_OFFSETS[8][4] = {
        {-1, -1, -1, 7}, {0, -1, -1, 6}, {-1, 0, -1, 5}, {0, 0, -1, 4},
        {-1, -1, 0, 3},  {0, -1, 0, 2},  {-1, 0, 0, 1},  {0, 0, 0, 0}};

    T f[NB_DOFS] = {T(0), T(0), T(0)};
    #pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
        int ex = static_cast<int>(ix) + ELEM_OFFSETS[elem][0];
        int ey = static_cast<int>(iy) + ELEM_OFFSETS[elem][1];
        int ez = static_cast<int>(iz) + ELEM_OFFSETS[elem][2];
        int local_node = ELEM_OFFSETS[elem][3];
        Index_t mat_idx =
            ex * mat_stride_x + ey * mat_stride_y + ez * mat_stride_z;
        T lam = lambda[mat_idx];
        T mu_val = mu[mat_idx];
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d) {
            int row = local_node * NB_DOFS + d;
            f[d] += static_cast<T>(2) * mu_val * static_cast<T>(d_Gu_3D[row]) +
                    lam * static_cast<T>(d_Vu_3D[row]);
        }
    }
    Index_t base =
        ix * force_stride_x + iy * force_stride_y + iz * force_stride_z;
    if (increment) {
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d)
            force[base + d * force_stride_d] += alpha * f[d];
    } else {
        #pragma unroll
        for (int d = 0; d < NB_DOFS; ++d)
            force[base + d * force_stride_d] = alpha * f[d];
    }
}

// ============================================================================
// Stress-Average GPU Kernels (grid-stride reduction into d_accum[dim*dim])
// ============================================================================

// 2D: each thread grid-strides over owned elements, accumulating the local
// stress σ = C(λ,μ):(E_macro + sym ḡ) in registers, then atomic-adds the
// Dim*Dim components into the global accumulator. d_accum is *not* scaled by
// the element volume here -- the host does that after copy-back.
template <typename T>
__global__ void isotropic_stiffness_2d_average_kernel(
    const T * __restrict__ displacement, const T * __restrict__ lambda,
    const T * __restrict__ mu, Index_t nelx, Index_t nely,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_d,
    Index_t mat_stride_x, Index_t mat_stride_y, Real * d_accum) {

    constexpr int NB_NODES = 4;
    constexpr int NB_DOFS = 2;
    constexpr int DIM = 2;
    const int NODE_OFFSET[4][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

    // Per-element stress formed in working precision T; the volume integral is
    // accumulated in double (cross-rank reduction stays double-accurate).
    Real acc[DIM * DIM] = {0.0, 0.0, 0.0, 0.0};
    Index_t nel = nelx * nely;
    Index_t stride = static_cast<Index_t>(blockDim.x) * gridDim.x;
    for (Index_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nel;
         e += stride) {
        Index_t ex = e % nelx;
        Index_t ey = e / nelx;
        T u[NB_NODES * NB_DOFS];
        #pragma unroll
        for (int node = 0; node < NB_NODES; ++node) {
            Index_t nx_pos = ex + NODE_OFFSET[node][0];
            Index_t ny_pos = ey + NODE_OFFSET[node][1];
            Index_t disp_idx = nx_pos * disp_stride_x + ny_pos * disp_stride_y;
            #pragma unroll
            for (int d = 0; d < NB_DOFS; ++d)
                u[node * NB_DOFS + d] = displacement[disp_idx + d * disp_stride_d];
        }
        Index_t mat_idx = ex * mat_stride_x + ey * mat_stride_y;
        T lam = lambda[mat_idx];
        T mu_val = mu[mat_idx];

        T g[DIM][DIM];
        #pragma unroll
        for (int i = 0; i < DIM; ++i)
            #pragma unroll
            for (int j = 0; j < DIM; ++j) {
                T s = T(0);
                #pragma unroll
                for (int n = 0; n < NB_NODES; ++n)
                    s += static_cast<T>(d_Dbar_2D[j * NB_NODES + n]) *
                         u[n * NB_DOFS + i];
                g[i][j] = s;
            }
        T E[DIM][DIM];
        #pragma unroll
        for (int i = 0; i < DIM; ++i)
            #pragma unroll
            for (int j = 0; j < DIM; ++j)
                E[i][j] = static_cast<T>(d_Emacro_2D[i * DIM + j]) +
                          static_cast<T>(0.5) * (g[i][j] + g[j][i]);
        T trE = E[0][0] + E[1][1];
        #pragma unroll
        for (int i = 0; i < DIM; ++i)
            #pragma unroll
            for (int j = 0; j < DIM; ++j)
                acc[i * DIM + j] += static_cast<Real>(
                    static_cast<T>(2) * mu_val * E[i][j] +
                    (i == j ? lam * trE : T(0)));
    }
    #pragma unroll
    for (int k = 0; k < DIM * DIM; ++k)
        mugrid_atomic_add_double(&d_accum[k], acc[k]);
}

// 3D stress average (see 2D variant).
template <typename T>
__global__ void isotropic_stiffness_3d_average_kernel(
    const T * __restrict__ displacement, const T * __restrict__ lambda,
    const T * __restrict__ mu, Index_t nelx, Index_t nely, Index_t nelz,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_z,
    Index_t disp_stride_d, Index_t mat_stride_x, Index_t mat_stride_y,
    Index_t mat_stride_z, Real * d_accum) {

    constexpr int NB_NODES = 8;
    constexpr int NB_DOFS = 3;
    constexpr int DIM = 3;
    const int NODE_OFFSET[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

    // Per-element stress in T, volume integral accumulated in double.
    Real acc[DIM * DIM];
    #pragma unroll
    for (int k = 0; k < DIM * DIM; ++k) acc[k] = 0.0;

    Index_t nelxy = nelx * nely;
    Index_t nel = nelxy * nelz;
    Index_t stride = static_cast<Index_t>(blockDim.x) * gridDim.x;
    for (Index_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nel;
         e += stride) {
        Index_t ez = e / nelxy;
        Index_t rem = e - ez * nelxy;
        Index_t ey = rem / nelx;
        Index_t ex = rem % nelx;
        T u[NB_NODES * NB_DOFS];
        #pragma unroll
        for (int node = 0; node < NB_NODES; ++node) {
            Index_t nx_pos = ex + NODE_OFFSET[node][0];
            Index_t ny_pos = ey + NODE_OFFSET[node][1];
            Index_t nz_pos = ez + NODE_OFFSET[node][2];
            Index_t disp_idx = nx_pos * disp_stride_x + ny_pos * disp_stride_y +
                               nz_pos * disp_stride_z;
            #pragma unroll
            for (int d = 0; d < NB_DOFS; ++d)
                u[node * NB_DOFS + d] = displacement[disp_idx + d * disp_stride_d];
        }
        Index_t mat_idx =
            ex * mat_stride_x + ey * mat_stride_y + ez * mat_stride_z;
        T lam = lambda[mat_idx];
        T mu_val = mu[mat_idx];

        T g[DIM][DIM];
        #pragma unroll
        for (int i = 0; i < DIM; ++i)
            #pragma unroll
            for (int j = 0; j < DIM; ++j) {
                T s = T(0);
                #pragma unroll
                for (int n = 0; n < NB_NODES; ++n)
                    s += static_cast<T>(d_Dbar_3D[j * NB_NODES + n]) *
                         u[n * NB_DOFS + i];
                g[i][j] = s;
            }
        T E[DIM][DIM];
        #pragma unroll
        for (int i = 0; i < DIM; ++i)
            #pragma unroll
            for (int j = 0; j < DIM; ++j)
                E[i][j] = static_cast<T>(d_Emacro_3D[i * DIM + j]) +
                          static_cast<T>(0.5) * (g[i][j] + g[j][i]);
        T trE = E[0][0] + E[1][1] + E[2][2];
        #pragma unroll
        for (int i = 0; i < DIM; ++i)
            #pragma unroll
            for (int j = 0; j < DIM; ++j)
                acc[i * DIM + j] += static_cast<Real>(
                    static_cast<T>(2) * mu_val * E[i][j] +
                    (i == j ? lam * trE : T(0)));
    }
    #pragma unroll
    for (int k = 0; k < DIM * DIM; ++k)
        mugrid_atomic_add_double(&d_accum[k], acc[k]);
}

// ============================================================================
// Kernel Wrapper Functions
// ============================================================================

namespace isotropic_stiffness_kernels {

// The geometry matrices (G/V/Gu/Vu/Dbar/E_macro) stay in double __constant__
// memory — uploaded here as `const Real*` regardless of T — and are cast to T
// inside the kernels. Only the field data and the per-element arithmetic are
// templated on T.
template <typename T>
void isotropic_stiffness_2d_gpu(
    const T* displacement, const T* lambda, const T* mu,
    T* force,
    Index_t nnx, Index_t nny,
    Index_t nelx, Index_t nely,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_d,
    Index_t mat_stride_x, Index_t mat_stride_y,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_d,
    const Real* G, const Real* V,
    T alpha, bool increment) {

    // Copy this instance's G and V to constant memory before every launch:
    // the matrices depend on the operator's grid_spacing, so a one-shot upload
    // would make a second operator with a different spacing silently use the
    // first one's matrices. The default-stream ordering makes the copy safe
    // (the kernel below runs after it).
    GPU_MEMCPY_TO_SYMBOL(d_G_2D, G, 64 * sizeof(Real));
    GPU_MEMCPY_TO_SYMBOL(d_V_2D, V, 64 * sizeof(Real));

    // Launch stiffness kernel - one thread per interior NODE
    dim3 block(16, 16);
    dim3 grid((nnx + block.x - 1) / block.x, (nny + block.y - 1) / block.y);

    // Alias the kernel so the launch macro does not split the `<T, false>`
    // template-argument comma into separate macro arguments.
    auto kern = isotropic_stiffness_2d_kernel<T, false>;
    GPU_LAUNCH_KERNEL(kern, grid, block,
        displacement, lambda, mu, force,
        nnx, nny,
        disp_stride_x, disp_stride_y, disp_stride_d,
        mat_stride_x, mat_stride_y,
        force_stride_x, force_stride_y, force_stride_d,
        alpha, increment, T(0), T(0));

    // Check for errors
    const char * err{gpu_last_error()};
    if (err != nullptr) {
        throw RuntimeError("GPU kernel launch failed: " + std::string(err));
    }
}

template <typename T>
void isotropic_stiffness_2d_gpu_uniform(
    const T* displacement, T lambda, T mu, T* force,
    Index_t nnx, Index_t nny,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_d,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_d,
    const Real* G, const Real* V,
    T alpha, bool increment) {

    GPU_MEMCPY_TO_SYMBOL(d_G_2D, G, 64 * sizeof(Real));
    GPU_MEMCPY_TO_SYMBOL(d_V_2D, V, 64 * sizeof(Real));

    dim3 block(16, 16);
    dim3 grid((nnx + block.x - 1) / block.x, (nny + block.y - 1) / block.y);

    auto kern = isotropic_stiffness_2d_kernel<T, true>;
    GPU_LAUNCH_KERNEL(kern, grid, block,
        displacement, nullptr, nullptr, force,
        nnx, nny,
        disp_stride_x, disp_stride_y, disp_stride_d,
        Index_t{0}, Index_t{0},
        force_stride_x, force_stride_y, force_stride_d,
        alpha, increment, lambda, mu);

    const char * err{gpu_last_error()};
    if (err != nullptr) {
        throw RuntimeError("GPU kernel launch failed: " + std::string(err));
    }
}

template <typename T>
void isotropic_stiffness_3d_gpu(
    const T* displacement, const T* lambda, const T* mu,
    T* force,
    Index_t nnx, Index_t nny, Index_t nnz,
    Index_t nelx, Index_t nely, Index_t nelz,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_z,
    Index_t disp_stride_d,
    Index_t mat_stride_x, Index_t mat_stride_y, Index_t mat_stride_z,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_z,
    Index_t force_stride_d,
    const Real* G, const Real* V,
    T alpha, bool increment) {

    // Copy this instance's G and V to constant memory before every launch (see
    // the 2D variant: the matrices depend on grid_spacing, so a one-shot upload
    // would let a second operator silently reuse the first one's matrices).
    GPU_MEMCPY_TO_SYMBOL(d_G_3D, G, 576 * sizeof(Real));
    GPU_MEMCPY_TO_SYMBOL(d_V_3D, V, 576 * sizeof(Real));

    // Launch stiffness kernel - one thread per interior NODE
    dim3 block(8, 8, 4);
    dim3 grid((nnx + block.x - 1) / block.x,
              (nny + block.y - 1) / block.y,
              (nnz + block.z - 1) / block.z);

    auto kern = isotropic_stiffness_3d_kernel<T, false>;
    GPU_LAUNCH_KERNEL(kern, grid, block,
        displacement, lambda, mu, force,
        nnx, nny, nnz,
        disp_stride_x, disp_stride_y, disp_stride_z, disp_stride_d,
        mat_stride_x, mat_stride_y, mat_stride_z,
        force_stride_x, force_stride_y, force_stride_z, force_stride_d,
        alpha, increment, T(0), T(0));

    // Check for errors
    const char * err{gpu_last_error()};
    if (err != nullptr) {
        throw RuntimeError("GPU kernel launch failed: " + std::string(err));
    }
}

template <typename T>
void isotropic_stiffness_3d_gpu_uniform(
    const T* displacement, T lambda, T mu, T* force,
    Index_t nnx, Index_t nny, Index_t nnz,
    Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_z,
    Index_t disp_stride_d,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_z,
    Index_t force_stride_d,
    const Real* G, const Real* V,
    T alpha, bool increment) {

    GPU_MEMCPY_TO_SYMBOL(d_G_3D, G, 576 * sizeof(Real));
    GPU_MEMCPY_TO_SYMBOL(d_V_3D, V, 576 * sizeof(Real));

    dim3 block(8, 8, 4);
    dim3 grid((nnx + block.x - 1) / block.x,
              (nny + block.y - 1) / block.y,
              (nnz + block.z - 1) / block.z);

    auto kern = isotropic_stiffness_3d_kernel<T, true>;
    GPU_LAUNCH_KERNEL(kern, grid, block,
        displacement, nullptr, nullptr, force,
        nnx, nny, nnz,
        disp_stride_x, disp_stride_y, disp_stride_z, disp_stride_d,
        Index_t{0}, Index_t{0}, Index_t{0},
        force_stride_x, force_stride_y, force_stride_z, force_stride_d,
        alpha, increment, lambda, mu);

    const char * err{gpu_last_error()};
    if (err != nullptr) {
        throw RuntimeError("GPU kernel launch failed: " + std::string(err));
    }
}

template <typename T>
void isotropic_stiffness_2d_gpu_macro_rhs(
    const T * lambda, const T * mu, T * force, Index_t nnx,
    Index_t nny, Index_t mat_stride_x, Index_t mat_stride_y,
    Index_t force_stride_x, Index_t force_stride_y, Index_t force_stride_d,
    const Real * Gu, const Real * Vu, T alpha, bool increment) {

    GPU_MEMCPY_TO_SYMBOL(d_Gu_2D, Gu, 8 * sizeof(Real));
    GPU_MEMCPY_TO_SYMBOL(d_Vu_2D, Vu, 8 * sizeof(Real));

    dim3 block(16, 16);
    dim3 grid((nnx + block.x - 1) / block.x, (nny + block.y - 1) / block.y);
    auto kern = isotropic_stiffness_2d_macro_rhs_kernel<T>;
    GPU_LAUNCH_KERNEL(kern, grid, block,
                      lambda, mu, force, nnx, nny, mat_stride_x, mat_stride_y,
                      force_stride_x, force_stride_y, force_stride_d, alpha,
                      increment);
    const char * err{gpu_last_error()};
    if (err != nullptr) {
        throw RuntimeError("GPU kernel launch failed: " + std::string(err));
    }
}

template <typename T>
void isotropic_stiffness_3d_gpu_macro_rhs(
    const T * lambda, const T * mu, T * force, Index_t nnx,
    Index_t nny, Index_t nnz, Index_t mat_stride_x, Index_t mat_stride_y,
    Index_t mat_stride_z, Index_t force_stride_x, Index_t force_stride_y,
    Index_t force_stride_z, Index_t force_stride_d, const Real * Gu,
    const Real * Vu, T alpha, bool increment) {

    GPU_MEMCPY_TO_SYMBOL(d_Gu_3D, Gu, 24 * sizeof(Real));
    GPU_MEMCPY_TO_SYMBOL(d_Vu_3D, Vu, 24 * sizeof(Real));

    dim3 block(8, 8, 4);
    dim3 grid((nnx + block.x - 1) / block.x, (nny + block.y - 1) / block.y,
              (nnz + block.z - 1) / block.z);
    auto kern = isotropic_stiffness_3d_macro_rhs_kernel<T>;
    GPU_LAUNCH_KERNEL(kern, grid, block,
                      lambda, mu, force, nnx, nny, nnz, mat_stride_x,
                      mat_stride_y, mat_stride_z, force_stride_x,
                      force_stride_y, force_stride_z, force_stride_d, alpha,
                      increment);
    const char * err{gpu_last_error()};
    if (err != nullptr) {
        throw RuntimeError("GPU kernel launch failed: " + std::string(err));
    }
}

template <typename T>
void isotropic_stiffness_2d_gpu_average(
    const T * displacement, const T * lambda, const T * mu,
    Index_t nelx, Index_t nely, Index_t disp_stride_x, Index_t disp_stride_y,
    Index_t disp_stride_d, Index_t mat_stride_x, Index_t mat_stride_y,
    const Real * Dbar, const Real * E_macro, Real vol_elem, Real * accum_out) {

    constexpr int NCOMP = 4;
    GPU_MEMCPY_TO_SYMBOL(d_Dbar_2D, Dbar, 8 * sizeof(Real));
    GPU_MEMCPY_TO_SYMBOL(d_Emacro_2D, E_macro, NCOMP * sizeof(Real));

    // The volume integral is reduced in double (matches the host); allocate a
    // double accumulator regardless of T.
    Real * d_accum{nullptr};
    GPU_MALLOC(reinterpret_cast<void **>(&d_accum), NCOMP * sizeof(Real));
    GPU_MEMSET(d_accum, 0, NCOMP * sizeof(Real));

    // Fixed grid-stride launch: keeps the number of atomics bounded (one set
    // of NCOMP per thread) regardless of grid size. Sized to fill the device.
    int block = 256;
    int grid = 1024;
    auto kern = isotropic_stiffness_2d_average_kernel<T>;
    GPU_LAUNCH_KERNEL(kern, grid, block,
                      displacement, lambda, mu, nelx, nely, disp_stride_x,
                      disp_stride_y, disp_stride_d, mat_stride_x, mat_stride_y,
                      d_accum);
    const char * err{gpu_last_error()};
    if (err != nullptr) {
        GPU_FREE(d_accum);
        throw RuntimeError("GPU kernel launch failed: " + std::string(err));
    }

    GPU_DEVICE_SYNCHRONIZE();
    GPU_MEMCPY_D2H(accum_out, d_accum, NCOMP * sizeof(Real));
    GPU_FREE(d_accum);
    for (int k = 0; k < NCOMP; ++k) accum_out[k] *= vol_elem;
}

template <typename T>
void isotropic_stiffness_3d_gpu_average(
    const T * displacement, const T * lambda, const T * mu,
    Index_t nelx, Index_t nely, Index_t nelz, Index_t disp_stride_x,
    Index_t disp_stride_y, Index_t disp_stride_z, Index_t disp_stride_d,
    Index_t mat_stride_x, Index_t mat_stride_y, Index_t mat_stride_z,
    const Real * Dbar, const Real * E_macro, Real vol_elem, Real * accum_out) {

    constexpr int NCOMP = 9;
    GPU_MEMCPY_TO_SYMBOL(d_Dbar_3D, Dbar, 24 * sizeof(Real));
    GPU_MEMCPY_TO_SYMBOL(d_Emacro_3D, E_macro, NCOMP * sizeof(Real));

    Real * d_accum{nullptr};
    GPU_MALLOC(reinterpret_cast<void **>(&d_accum), NCOMP * sizeof(Real));
    GPU_MEMSET(d_accum, 0, NCOMP * sizeof(Real));

    int block = 256;
    int grid = 1024;
    auto kern = isotropic_stiffness_3d_average_kernel<T>;
    GPU_LAUNCH_KERNEL(kern, grid, block,
                      displacement, lambda, mu, nelx, nely, nelz, disp_stride_x,
                      disp_stride_y, disp_stride_z, disp_stride_d, mat_stride_x,
                      mat_stride_y, mat_stride_z, d_accum);
    const char * err{gpu_last_error()};
    if (err != nullptr) {
        GPU_FREE(d_accum);
        throw RuntimeError("GPU kernel launch failed: " + std::string(err));
    }

    GPU_DEVICE_SYNCHRONIZE();
    GPU_MEMCPY_D2H(accum_out, d_accum, NCOMP * sizeof(Real));
    GPU_FREE(d_accum);
    for (int k = 0; k < NCOMP; ++k) accum_out[k] *= vol_elem;
}

}  // namespace isotropic_stiffness_kernels

// ============================================================================
// Class Method Implementations for GPU
// ============================================================================

template <>
template <typename T>
void IsotropicStiffnessOperator<2>::apply_impl(
    const TypedFieldBase<T, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<T, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<T, DefaultDeviceSpace>& mu,
    T alpha,
    TypedFieldBase<T, DefaultDeviceSpace>& force,
    bool increment) const {

    // Validate field collections and ghosts; the dimension-generic checks
    // live in internal::validate_stiffness_fields (isotropic_stiffness.hh).
    const auto info = internal::validate_stiffness_fields<2>(
        displacement.get_collection(), lambda.get_collection());
    const auto* disp_global_fc = info.disp_fc;
    const auto* mat_global_fc = info.mat_fc;

    // Computable region (node field == material field, guaranteed above).
    const Index_t nnx = info.nb_computable[0];
    const Index_t nny = info.nb_computable[1];
    const Index_t nelx = nnx;
    const Index_t nely = nny;

    // Stencil offset: one ghost cell on each side (see validation helper).
    constexpr Index_t STENCIL_LEFT = 1;

    // Node dimensions (for displacement/force fields with ghosts)
    auto nb_with_ghosts = disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
    auto mat_nb_with_ghosts = mat_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t nx = nb_with_ghosts[0];
    Index_t ny = nb_with_ghosts[1];

    // Material field dimensions with ghosts
    Index_t mat_nx = mat_nb_with_ghosts[0];
    Index_t mat_ny = mat_nb_with_ghosts[1];

    // Offset to first computable node (based on stencil requirements)
    Index_t ghost_offset_x = STENCIL_LEFT;
    Index_t ghost_offset_y = STENCIL_LEFT;
    Index_t mat_ghost_offset_x = STENCIL_LEFT;
    Index_t mat_ghost_offset_y = STENCIL_LEFT;

    // GPU uses SoA layout: [d, x, y]
    Index_t disp_stride_d = nx * ny;
    Index_t disp_stride_x = 1;
    Index_t disp_stride_y = nx;

    Index_t mat_stride_x = 1;
    Index_t mat_stride_y = mat_nx;

    Index_t force_stride_d = nx * ny;
    Index_t force_stride_x = 1;
    Index_t force_stride_y = nx;

    // Offset data pointers to account for left ghosts
    Index_t disp_offset = ghost_offset_x * disp_stride_x +
                          ghost_offset_y * disp_stride_y;
    Index_t force_offset = ghost_offset_x * force_stride_x +
                           ghost_offset_y * force_stride_y;
    Index_t mat_offset = mat_ghost_offset_x * mat_stride_x +
                         mat_ghost_offset_y * mat_stride_y;

    isotropic_stiffness_kernels::isotropic_stiffness_2d_gpu<T>(
        displacement.view().data() + disp_offset,
        lambda.view().data() + mat_offset,
        mu.view().data() + mat_offset,
        force.view().data() + force_offset,
        nnx, nny,  // Number of interior nodes
        nelx, nely,  // Number of elements (same as nodes with node-based indexing)
        disp_stride_x, disp_stride_y, disp_stride_d,
        mat_stride_x, mat_stride_y,
        force_stride_x, force_stride_y, force_stride_d,
        G_matrix.data(), V_matrix.data(),
        alpha, increment);
}

template <>
template <typename T>
void IsotropicStiffnessOperator<3>::apply_impl(
    const TypedFieldBase<T, DefaultDeviceSpace>& displacement,
    const TypedFieldBase<T, DefaultDeviceSpace>& lambda,
    const TypedFieldBase<T, DefaultDeviceSpace>& mu,
    T alpha,
    TypedFieldBase<T, DefaultDeviceSpace>& force,
    bool increment) const {

    // Validate field collections and ghosts; the dimension-generic checks
    // live in internal::validate_stiffness_fields (isotropic_stiffness.hh).
    const auto info = internal::validate_stiffness_fields<3>(
        displacement.get_collection(), lambda.get_collection());
    const auto* disp_global_fc = info.disp_fc;
    const auto* mat_global_fc = info.mat_fc;

    // Computable region (node field == material field, guaranteed above).
    const Index_t nnx = info.nb_computable[0];
    const Index_t nny = info.nb_computable[1];
    const Index_t nnz = info.nb_computable[2];
    const Index_t nelx = nnx;
    const Index_t nely = nny;
    const Index_t nelz = nnz;

    // Stencil offset: one ghost cell on each side (see validation helper).
    constexpr Index_t STENCIL_LEFT = 1;

    // Node dimensions (for displacement/force fields with ghosts)
    auto nb_with_ghosts = disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
    auto mat_nb_with_ghosts = mat_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t nx = nb_with_ghosts[0];
    Index_t ny = nb_with_ghosts[1];
    Index_t nz = nb_with_ghosts[2];

    // Material field dimensions with ghosts
    Index_t mat_nx = mat_nb_with_ghosts[0];
    Index_t mat_ny = mat_nb_with_ghosts[1];
    Index_t mat_nz = mat_nb_with_ghosts[2];

    // Offset to first computable node (based on stencil requirements)
    Index_t ghost_offset_x = STENCIL_LEFT;
    Index_t ghost_offset_y = STENCIL_LEFT;
    Index_t ghost_offset_z = STENCIL_LEFT;
    Index_t mat_ghost_offset_x = STENCIL_LEFT;
    Index_t mat_ghost_offset_y = STENCIL_LEFT;
    Index_t mat_ghost_offset_z = STENCIL_LEFT;

    // GPU uses SoA layout: [d, x, y, z]
    Index_t disp_stride_d = nx * ny * nz;
    Index_t disp_stride_x = 1;
    Index_t disp_stride_y = nx;
    Index_t disp_stride_z = nx * ny;

    Index_t mat_stride_x = 1;
    Index_t mat_stride_y = mat_nx;
    Index_t mat_stride_z = mat_nx * mat_ny;

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
    Index_t mat_offset = mat_ghost_offset_x * mat_stride_x +
                         mat_ghost_offset_y * mat_stride_y +
                         mat_ghost_offset_z * mat_stride_z;

    isotropic_stiffness_kernels::isotropic_stiffness_3d_gpu<T>(
        displacement.view().data() + disp_offset,
        lambda.view().data() + mat_offset,
        mu.view().data() + mat_offset,
        force.view().data() + force_offset,
        nnx, nny, nnz,  // Number of interior nodes
        nelx, nely, nelz,  // Number of elements (same as nodes)
        disp_stride_x, disp_stride_y, disp_stride_z, disp_stride_d,
        mat_stride_x, mat_stride_y, mat_stride_z,
        force_stride_x, force_stride_y, force_stride_z, force_stride_d,
        G_matrix.data(), V_matrix.data(),
        alpha, increment);
}

template <>
template <typename T>
void IsotropicStiffnessOperator<2>::apply_uniform_impl(
    const TypedFieldBase<T, DefaultDeviceSpace>& displacement,
    T lambda, T mu, T alpha,
    TypedFieldBase<T, DefaultDeviceSpace>& force,
    bool increment) const {

    auto& disp_coll = displacement.get_collection();
    auto* disp_global_fc = dynamic_cast<const GlobalFieldCollection*>(&disp_coll);
    if (!disp_global_fc) {
        throw RuntimeError("IsotropicStiffnessOperator2D requires GlobalFieldCollection");
    }

    auto nb_ghosts_left = disp_global_fc->get_nb_ghosts_left();
    auto nb_ghosts_right = disp_global_fc->get_nb_ghosts_right();
    if (nb_ghosts_left[0] < 1 || nb_ghosts_left[1] < 1 ||
        nb_ghosts_right[0] < 1 || nb_ghosts_right[1] < 1) {
        throw RuntimeError("IsotropicStiffnessOperator2D requires at least 1 "
                           "ghost cell on both sides of displacement/force fields");
    }

    constexpr Index_t STENCIL_LEFT = 1;
    constexpr Index_t STENCIL_RIGHT = 1;

    auto nb_with_ghosts = disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t nnx = nb_with_ghosts[0] - STENCIL_LEFT - STENCIL_RIGHT;
    Index_t nny = nb_with_ghosts[1] - STENCIL_LEFT - STENCIL_RIGHT;

    Index_t nx = nb_with_ghosts[0];
    Index_t ny = nb_with_ghosts[1];

    // GPU uses SoA layout: [d, x, y]
    Index_t disp_stride_d = nx * ny;
    Index_t disp_stride_x = 1;
    Index_t disp_stride_y = nx;
    Index_t force_stride_d = nx * ny;
    Index_t force_stride_x = 1;
    Index_t force_stride_y = nx;

    Index_t disp_offset = STENCIL_LEFT * disp_stride_x +
                          STENCIL_LEFT * disp_stride_y;
    Index_t force_offset = STENCIL_LEFT * force_stride_x +
                           STENCIL_LEFT * force_stride_y;

    isotropic_stiffness_kernels::isotropic_stiffness_2d_gpu_uniform<T>(
        displacement.view().data() + disp_offset, lambda, mu,
        force.view().data() + force_offset,
        nnx, nny,
        disp_stride_x, disp_stride_y, disp_stride_d,
        force_stride_x, force_stride_y, force_stride_d,
        G_matrix.data(), V_matrix.data(),
        alpha, increment);
}

template <>
template <typename T>
void IsotropicStiffnessOperator<3>::apply_uniform_impl(
    const TypedFieldBase<T, DefaultDeviceSpace>& displacement,
    T lambda, T mu, T alpha,
    TypedFieldBase<T, DefaultDeviceSpace>& force,
    bool increment) const {

    auto& disp_coll = displacement.get_collection();
    auto* disp_global_fc = dynamic_cast<const GlobalFieldCollection*>(&disp_coll);
    if (!disp_global_fc) {
        throw RuntimeError("IsotropicStiffnessOperator3D requires GlobalFieldCollection");
    }

    auto nb_ghosts_left = disp_global_fc->get_nb_ghosts_left();
    auto nb_ghosts_right = disp_global_fc->get_nb_ghosts_right();
    if (nb_ghosts_left[0] < 1 || nb_ghosts_left[1] < 1 || nb_ghosts_left[2] < 1 ||
        nb_ghosts_right[0] < 1 || nb_ghosts_right[1] < 1 || nb_ghosts_right[2] < 1) {
        throw RuntimeError("IsotropicStiffnessOperator3D requires at least 1 "
                           "ghost cell on both sides of displacement/force fields");
    }

    constexpr Index_t STENCIL_LEFT = 1;
    constexpr Index_t STENCIL_RIGHT = 1;

    auto nb_with_ghosts = disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t nnx = nb_with_ghosts[0] - STENCIL_LEFT - STENCIL_RIGHT;
    Index_t nny = nb_with_ghosts[1] - STENCIL_LEFT - STENCIL_RIGHT;
    Index_t nnz = nb_with_ghosts[2] - STENCIL_LEFT - STENCIL_RIGHT;

    Index_t nx = nb_with_ghosts[0];
    Index_t ny = nb_with_ghosts[1];
    Index_t nz = nb_with_ghosts[2];

    // GPU uses SoA layout: [d, x, y, z]
    Index_t disp_stride_d = nx * ny * nz;
    Index_t disp_stride_x = 1;
    Index_t disp_stride_y = nx;
    Index_t disp_stride_z = nx * ny;
    Index_t force_stride_d = nx * ny * nz;
    Index_t force_stride_x = 1;
    Index_t force_stride_y = nx;
    Index_t force_stride_z = nx * ny;

    Index_t disp_offset = STENCIL_LEFT * disp_stride_x +
                          STENCIL_LEFT * disp_stride_y +
                          STENCIL_LEFT * disp_stride_z;
    Index_t force_offset = STENCIL_LEFT * force_stride_x +
                           STENCIL_LEFT * force_stride_y +
                           STENCIL_LEFT * force_stride_z;

    isotropic_stiffness_kernels::isotropic_stiffness_3d_gpu_uniform<T>(
        displacement.view().data() + disp_offset, lambda, mu,
        force.view().data() + force_offset,
        nnx, nny, nnz,
        disp_stride_x, disp_stride_y, disp_stride_z, disp_stride_d,
        force_stride_x, force_stride_y, force_stride_z, force_stride_d,
        G_matrix.data(), V_matrix.data(),
        alpha, increment);
}

// ---- Streaming homogenization helpers (device) ----

template <>
template <typename T>
void IsotropicStiffnessOperator<2>::apply_macro_rhs_impl(
    const TypedFieldBase<T, DefaultDeviceSpace> & lambda,
    const TypedFieldBase<T, DefaultDeviceSpace> & mu,
    const std::array<Real, 4> & E_macro,
    TypedFieldBase<T, DefaultDeviceSpace> & force) const {

    const auto info = internal::validate_stiffness_fields<2>(
        force.get_collection(), lambda.get_collection());
    const auto * node_fc = info.disp_fc;
    const auto * mat_fc = info.mat_fc;
    const Index_t nnx = info.nb_computable[0];
    const Index_t nny = info.nb_computable[1];
    constexpr Index_t STENCIL_LEFT = 1;

    auto wg = node_fc->get_nb_subdomain_grid_pts_with_ghosts();
    auto mwg = mat_fc->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t nx = wg[0], ny = wg[1];
    Index_t mat_nx = mwg[0];

    // SoA layout: [d, x, y]
    Index_t force_stride_d = nx * ny, force_stride_x = 1, force_stride_y = nx;
    Index_t mat_stride_x = 1, mat_stride_y = mat_nx;
    Index_t force_offset =
        STENCIL_LEFT * force_stride_x + STENCIL_LEFT * force_stride_y;
    Index_t mat_offset =
        STENCIL_LEFT * mat_stride_x + STENCIL_LEFT * mat_stride_y;

    ElementMatrix Gu{}, Vu{};
    this->macro_rhs_vectors(E_macro, Gu, Vu);

    isotropic_stiffness_kernels::isotropic_stiffness_2d_gpu_macro_rhs<T>(
        lambda.view().data() + mat_offset, mu.view().data() + mat_offset,
        force.view().data() + force_offset, nnx, nny, mat_stride_x,
        mat_stride_y, force_stride_x, force_stride_y, force_stride_d, Gu.data(),
        Vu.data(), static_cast<T>(1), false);
}

template <>
template <typename T>
void IsotropicStiffnessOperator<3>::apply_macro_rhs_impl(
    const TypedFieldBase<T, DefaultDeviceSpace> & lambda,
    const TypedFieldBase<T, DefaultDeviceSpace> & mu,
    const std::array<Real, 9> & E_macro,
    TypedFieldBase<T, DefaultDeviceSpace> & force) const {

    const auto info = internal::validate_stiffness_fields<3>(
        force.get_collection(), lambda.get_collection());
    const auto * node_fc = info.disp_fc;
    const auto * mat_fc = info.mat_fc;
    const Index_t nnx = info.nb_computable[0];
    const Index_t nny = info.nb_computable[1];
    const Index_t nnz = info.nb_computable[2];
    constexpr Index_t STENCIL_LEFT = 1;

    auto wg = node_fc->get_nb_subdomain_grid_pts_with_ghosts();
    auto mwg = mat_fc->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t nx = wg[0], ny = wg[1], nz = wg[2];
    Index_t mat_nx = mwg[0], mat_ny = mwg[1];

    Index_t force_stride_d = nx * ny * nz, force_stride_x = 1,
            force_stride_y = nx, force_stride_z = nx * ny;
    Index_t mat_stride_x = 1, mat_stride_y = mat_nx,
            mat_stride_z = mat_nx * mat_ny;
    Index_t force_offset = STENCIL_LEFT * force_stride_x +
                           STENCIL_LEFT * force_stride_y +
                           STENCIL_LEFT * force_stride_z;
    Index_t mat_offset = STENCIL_LEFT * mat_stride_x +
                         STENCIL_LEFT * mat_stride_y +
                         STENCIL_LEFT * mat_stride_z;

    ElementMatrix Gu{}, Vu{};
    this->macro_rhs_vectors(E_macro, Gu, Vu);

    isotropic_stiffness_kernels::isotropic_stiffness_3d_gpu_macro_rhs<T>(
        lambda.view().data() + mat_offset, mu.view().data() + mat_offset,
        force.view().data() + force_offset, nnx, nny, nnz, mat_stride_x,
        mat_stride_y, mat_stride_z, force_stride_x, force_stride_y,
        force_stride_z, force_stride_d, Gu.data(), Vu.data(),
        static_cast<T>(1), false);
}

template <>
template <typename T>
void IsotropicStiffnessOperator<2>::assemble_diagonal_impl(
    const TypedFieldBase<T, DefaultDeviceSpace> & lambda,
    const TypedFieldBase<T, DefaultDeviceSpace> & mu,
    TypedFieldBase<T, DefaultDeviceSpace> & diagonal) const {

    // diag(K) = Σ_e (2μ_e diag(G) + λ_e diag(V)): the macro-RHS gather with the
    // per-element vectors replaced by the diagonals of G, V (see the host impl).
    const auto info = internal::validate_stiffness_fields<2>(
        diagonal.get_collection(), lambda.get_collection());
    const auto * node_fc = info.disp_fc;
    const auto * mat_fc = info.mat_fc;
    const Index_t nnx = info.nb_computable[0];
    const Index_t nny = info.nb_computable[1];
    constexpr Index_t STENCIL_LEFT = 1;

    auto wg = node_fc->get_nb_subdomain_grid_pts_with_ghosts();
    auto mwg = mat_fc->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t nx = wg[0], ny = wg[1];
    Index_t mat_nx = mwg[0];

    // SoA layout: [d, x, y]
    Index_t force_stride_d = nx * ny, force_stride_x = 1, force_stride_y = nx;
    Index_t mat_stride_x = 1, mat_stride_y = mat_nx;
    Index_t force_offset =
        STENCIL_LEFT * force_stride_x + STENCIL_LEFT * force_stride_y;
    Index_t mat_offset =
        STENCIL_LEFT * mat_stride_x + STENCIL_LEFT * mat_stride_y;

    ElementMatrix Gd{}, Vd{};
    for (Index_t r = 0; r < NB_ELEMENT_DOFS; ++r) {
        Gd[r] = G_matrix[r * NB_ELEMENT_DOFS + r];
        Vd[r] = V_matrix[r * NB_ELEMENT_DOFS + r];
    }

    isotropic_stiffness_kernels::isotropic_stiffness_2d_gpu_macro_rhs<T>(
        lambda.view().data() + mat_offset, mu.view().data() + mat_offset,
        diagonal.view().data() + force_offset, nnx, nny, mat_stride_x,
        mat_stride_y, force_stride_x, force_stride_y, force_stride_d, Gd.data(),
        Vd.data(), static_cast<T>(1), false);
}

template <>
template <typename T>
void IsotropicStiffnessOperator<3>::assemble_diagonal_impl(
    const TypedFieldBase<T, DefaultDeviceSpace> & lambda,
    const TypedFieldBase<T, DefaultDeviceSpace> & mu,
    TypedFieldBase<T, DefaultDeviceSpace> & diagonal) const {

    const auto info = internal::validate_stiffness_fields<3>(
        diagonal.get_collection(), lambda.get_collection());
    const auto * node_fc = info.disp_fc;
    const auto * mat_fc = info.mat_fc;
    const Index_t nnx = info.nb_computable[0];
    const Index_t nny = info.nb_computable[1];
    const Index_t nnz = info.nb_computable[2];
    constexpr Index_t STENCIL_LEFT = 1;

    auto wg = node_fc->get_nb_subdomain_grid_pts_with_ghosts();
    auto mwg = mat_fc->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t nx = wg[0], ny = wg[1], nz = wg[2];
    Index_t mat_nx = mwg[0], mat_ny = mwg[1];

    Index_t force_stride_d = nx * ny * nz, force_stride_x = 1,
            force_stride_y = nx, force_stride_z = nx * ny;
    Index_t mat_stride_x = 1, mat_stride_y = mat_nx,
            mat_stride_z = mat_nx * mat_ny;
    Index_t force_offset = STENCIL_LEFT * force_stride_x +
                           STENCIL_LEFT * force_stride_y +
                           STENCIL_LEFT * force_stride_z;
    Index_t mat_offset = STENCIL_LEFT * mat_stride_x +
                         STENCIL_LEFT * mat_stride_y +
                         STENCIL_LEFT * mat_stride_z;

    ElementMatrix Gd{}, Vd{};
    for (Index_t r = 0; r < NB_ELEMENT_DOFS; ++r) {
        Gd[r] = G_matrix[r * NB_ELEMENT_DOFS + r];
        Vd[r] = V_matrix[r * NB_ELEMENT_DOFS + r];
    }

    isotropic_stiffness_kernels::isotropic_stiffness_3d_gpu_macro_rhs<T>(
        lambda.view().data() + mat_offset, mu.view().data() + mat_offset,
        diagonal.view().data() + force_offset, nnx, nny, nnz, mat_stride_x,
        mat_stride_y, mat_stride_z, force_stride_x, force_stride_y,
        force_stride_z, force_stride_d, Gd.data(), Vd.data(),
        static_cast<T>(1), false);
}

template <>
template <typename T>
std::array<Real, 4> IsotropicStiffnessOperator<2>::average_stress_impl(
    const TypedFieldBase<T, DefaultDeviceSpace> & displacement,
    const TypedFieldBase<T, DefaultDeviceSpace> & lambda,
    const TypedFieldBase<T, DefaultDeviceSpace> & mu,
    const std::array<Real, 4> & E_macro) const {

    const auto info = internal::validate_stiffness_fields<2>(
        displacement.get_collection(), lambda.get_collection());
    const auto * disp_fc = info.disp_fc;
    const auto * mat_fc = info.mat_fc;

    auto gl = disp_fc->get_nb_ghosts_left();
    auto gr = disp_fc->get_nb_ghosts_right();
    auto wg = disp_fc->get_nb_subdomain_grid_pts_with_ghosts();
    auto mgl = mat_fc->get_nb_ghosts_left();
    auto mwg = mat_fc->get_nb_subdomain_grid_pts_with_ghosts();

    const Index_t nelx = wg[0] - gl[0] - gr[0];
    const Index_t nely = wg[1] - gl[1] - gr[1];
    Index_t nx = wg[0], ny = wg[1];
    Index_t mat_nx = mwg[0];

    Index_t disp_stride_d = nx * ny, disp_stride_x = 1, disp_stride_y = nx;
    Index_t mat_stride_x = 1, mat_stride_y = mat_nx;
    Index_t disp_offset = gl[0] * disp_stride_x + gl[1] * disp_stride_y;
    Index_t mat_offset = mgl[0] * mat_stride_x + mgl[1] * mat_stride_y;

    const Real vol_elem = grid_spacing[0] * grid_spacing[1];
    std::array<Real, 4> accum{};
    isotropic_stiffness_kernels::isotropic_stiffness_2d_gpu_average<T>(
        displacement.view().data() + disp_offset,
        lambda.view().data() + mat_offset, mu.view().data() + mat_offset, nelx,
        nely, disp_stride_x, disp_stride_y, disp_stride_d, mat_stride_x,
        mat_stride_y, Dbar_matrix.data(), E_macro.data(), vol_elem,
        accum.data());
    return accum;
}

template <>
template <typename T>
std::array<Real, 9> IsotropicStiffnessOperator<3>::average_stress_impl(
    const TypedFieldBase<T, DefaultDeviceSpace> & displacement,
    const TypedFieldBase<T, DefaultDeviceSpace> & lambda,
    const TypedFieldBase<T, DefaultDeviceSpace> & mu,
    const std::array<Real, 9> & E_macro) const {

    const auto info = internal::validate_stiffness_fields<3>(
        displacement.get_collection(), lambda.get_collection());
    const auto * disp_fc = info.disp_fc;
    const auto * mat_fc = info.mat_fc;

    auto gl = disp_fc->get_nb_ghosts_left();
    auto gr = disp_fc->get_nb_ghosts_right();
    auto wg = disp_fc->get_nb_subdomain_grid_pts_with_ghosts();
    auto mgl = mat_fc->get_nb_ghosts_left();
    auto mwg = mat_fc->get_nb_subdomain_grid_pts_with_ghosts();

    const Index_t nelx = wg[0] - gl[0] - gr[0];
    const Index_t nely = wg[1] - gl[1] - gr[1];
    const Index_t nelz = wg[2] - gl[2] - gr[2];
    Index_t nx = wg[0], ny = wg[1], nz = wg[2];
    Index_t mat_nx = mwg[0], mat_ny = mwg[1];

    Index_t disp_stride_d = nx * ny * nz, disp_stride_x = 1, disp_stride_y = nx,
            disp_stride_z = nx * ny;
    Index_t mat_stride_x = 1, mat_stride_y = mat_nx,
            mat_stride_z = mat_nx * mat_ny;
    Index_t disp_offset = gl[0] * disp_stride_x + gl[1] * disp_stride_y +
                          gl[2] * disp_stride_z;
    Index_t mat_offset = mgl[0] * mat_stride_x + mgl[1] * mat_stride_y +
                         mgl[2] * mat_stride_z;

    const Real vol_elem = grid_spacing[0] * grid_spacing[1] * grid_spacing[2];
    std::array<Real, 9> accum{};
    isotropic_stiffness_kernels::isotropic_stiffness_3d_gpu_average<T>(
        displacement.view().data() + disp_offset,
        lambda.view().data() + mat_offset, mu.view().data() + mat_offset, nelx,
        nely, nelz, disp_stride_x, disp_stride_y, disp_stride_z, disp_stride_d,
        mat_stride_x, mat_stride_y, mat_stride_z, Dbar_matrix.data(),
        E_macro.data(), vol_elem, accum.data());
    return accum;
}

// Explicit instantiations of the per-precision device impls. The (templated)
// kernels and launch wrappers above are instantiated implicitly through these.
#define MUGRID_INSTANTIATE_STIFFNESS_GPU(D, T)                                 \
    template void IsotropicStiffnessOperator<D>::apply_impl<T>(                \
        const TypedFieldBase<T, DefaultDeviceSpace> &,                         \
        const TypedFieldBase<T, DefaultDeviceSpace> &,                         \
        const TypedFieldBase<T, DefaultDeviceSpace> &, T,                      \
        TypedFieldBase<T, DefaultDeviceSpace> &, bool) const;                  \
    template void IsotropicStiffnessOperator<D>::apply_uniform_impl<T>(        \
        const TypedFieldBase<T, DefaultDeviceSpace> &, T, T, T,                \
        TypedFieldBase<T, DefaultDeviceSpace> &, bool) const;                  \
    template void IsotropicStiffnessOperator<D>::apply_macro_rhs_impl<T>(      \
        const TypedFieldBase<T, DefaultDeviceSpace> &,                         \
        const TypedFieldBase<T, DefaultDeviceSpace> &,                         \
        const std::array<Real, D * D> &,                                       \
        TypedFieldBase<T, DefaultDeviceSpace> &) const;                        \
    template std::array<Real, D * D>                                           \
    IsotropicStiffnessOperator<D>::average_stress_impl<T>(                     \
        const TypedFieldBase<T, DefaultDeviceSpace> &,                         \
        const TypedFieldBase<T, DefaultDeviceSpace> &,                         \
        const TypedFieldBase<T, DefaultDeviceSpace> &,                         \
        const std::array<Real, D * D> &) const;                                \
    template void IsotropicStiffnessOperator<D>::assemble_diagonal_impl<T>(    \
        const TypedFieldBase<T, DefaultDeviceSpace> &,                         \
        const TypedFieldBase<T, DefaultDeviceSpace> &,                         \
        TypedFieldBase<T, DefaultDeviceSpace> &) const;
    MUGRID_INSTANTIATE_STIFFNESS_GPU(2, Real)
    MUGRID_INSTANTIATE_STIFFNESS_GPU(3, Real)
    MUGRID_INSTANTIATE_STIFFNESS_GPU(2, Real32)
    MUGRID_INSTANTIATE_STIFFNESS_GPU(3, Real32)
#undef MUGRID_INSTANTIATE_STIFFNESS_GPU

}  // namespace muGrid

#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP
