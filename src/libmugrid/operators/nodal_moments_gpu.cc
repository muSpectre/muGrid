/**
 * @file   nodal_moments_gpu.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   16 Sep 2026
 *
 * @brief  CUDA/HIP kernels for NodalMomentOperator. See nodal_moments.hh for
 *         the interface and the rationale; the arithmetic is the same as the
 *         host kernel in nodal_moments.cc, verified against it.
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

#include "operators/nodal_moments.hh"
#include "memory/gpu_runtime.hh"

namespace muGrid {
    namespace nodal_moment_kernels {

        // Tile of the 3D kernel. Compile-time so the kernel can size its
        // shared-memory staging buffer; the launcher uses the same values.
        constexpr int MOMENT_TILE_X = 8;
        constexpr int MOMENT_TILE_Y = 8;
        constexpr int MOMENT_TILE_Z = 4;
        constexpr int MOMENT_TILE_2D_X = 16;
        constexpr int MOMENT_TILE_2D_Y = 16;

        /**
         * @brief Quadrature tables in the kernel's working precision.
         *
         * MomentQuadrature computes its entries constexpr, but from `Real`
         * (double). Evaluating them as double inside a float32 kernel would
         * issue every inner-loop term on the fp64 pipe -- the cost the fused
         * stiffness kernel's geometry tables were moved off in v1.1.1. So the
         * table is materialised once, at T's precision, in registers/constant
         * cache via a constexpr array.
         */
        template <typename T, class Q>
        struct QuadTable {
            T N[Q::NbQuad][Q::NbNodes];
            T W[Q::NbQuad];
        };

        template <typename T, class Q>
        constexpr QuadTable<T, Q> make_quad_table() {
            QuadTable<T, Q> t{};
            for (Index_t q = 0; q < Q::NbQuad; ++q) {
                t.W[q] = static_cast<T>(Q::weight(q));
                for (Index_t n = 0; n < Q::NbNodes; ++n) {
                    t.N[q][n] = static_cast<T>(Q::shape(q, n));
                }
            }
            return t;
        }

        /**
         * @brief 3D kernel: one thread per node, which also owns the cell of
         *        the same index.
         *
         * The node values this thread needs span [-1, +1] around it, and
         * neighbouring threads' spans overlap heavily -- each node is read by
         * the 27 threads whose cells touch it. Stage the block's tile plus its
         * one-node halo in shared memory and serve the reuse from there, as
         * the fused stiffness kernel does.
         *
         * Registers are bounded so two blocks fit per SM: the kernel is
         * arithmetic-heavy (NbNodes cells x NbQuad points x NbNodes corners)
         * and latency-bound without a second block to switch to.
         */
        template <typename T, class Element>
        __global__ __launch_bounds__(MOMENT_TILE_X * MOMENT_TILE_Y *
                                         MOMENT_TILE_Z,
                                     2) void
        nodal_moments_3d_kernel(const T * __restrict__ rho,
                                T * __restrict__ moments,
                                T * __restrict__ moment_gradients, Index_t nx,
                                Index_t ny, Index_t nz, Index_t rho_stride_x,
                                Index_t rho_stride_y, Index_t rho_stride_z,
                                Index_t mom_stride_x, Index_t mom_stride_y,
                                Index_t mom_stride_z, Index_t mom_stride_c,
                                T cell_volume) {
            using Q = MomentQuadrature<Element>;
            constexpr int NB_NODES = static_cast<int>(Q::NbNodes);
            constexpr auto TAB = make_quad_table<T, Q>();

            constexpr int SX = MOMENT_TILE_X + 2;
            constexpr int SY = MOMENT_TILE_Y + 2;
            constexpr int SZ = MOMENT_TILE_Z + 2;
            __shared__ T s_rho[SZ * SY * SX];

            const int bx = blockIdx.x * MOMENT_TILE_X;
            const int by = blockIdx.y * MOMENT_TILE_Y;
            const int bz = blockIdx.z * MOMENT_TILE_Z;
            const int tid =
                threadIdx.x +
                MOMENT_TILE_X * (threadIdx.y + MOMENT_TILE_Y * threadIdx.z);
            constexpr int NB_THREADS =
                MOMENT_TILE_X * MOMENT_TILE_Y * MOMENT_TILE_Z;

            // Every thread takes part in the cooperative load and must reach
            // __syncthreads(), so the bounds check moves down to the store.
            for (int idx = tid; idx < SZ * SY * SX; idx += NB_THREADS) {
                const int lx = idx % SX;
                const int ly = (idx / SX) % SY;
                const int lz = idx / (SX * SY);
                // The field is addressable over [-1, n] (one ghost layer);
                // clamp so a partially out-of-range block still reads real
                // memory -- its out-of-range threads discard their results.
                const int gx = min(max(bx - 1 + lx, -1), static_cast<int>(nx));
                const int gy = min(max(by - 1 + ly, -1), static_cast<int>(ny));
                const int gz = min(max(bz - 1 + lz, -1), static_cast<int>(nz));
                s_rho[idx] = rho[gx * rho_stride_x + gy * rho_stride_y +
                                 gz * rho_stride_z];
            }
            __syncthreads();

            const Index_t ix = bx + threadIdx.x;
            const Index_t iy = by + threadIdx.y;
            const Index_t iz = bz + threadIdx.z;
            const bool in_grid = (ix < nx && iy < ny && iz < nz);

            // This thread's own node in tile coordinates.
            const int lx0 = threadIdx.x + 1;
            const int ly0 = threadIdx.y + 1;
            const int lz0 = threadIdx.z + 1;

            T value[NB_NODAL_MOMENTS]{};
            T grad[NB_NODAL_MOMENTS]{};

            // Cell offsets are 0 or -1 and node offsets 0 or +1, so every
            // index lands inside the staged tile: [lx0-1, lx0+1] ⊂ [0, SX-1].
            for (int c = 0; c < NB_NODES; ++c) {
                const int ex = lx0 - static_cast<int>(fem_node_offset(c, 0));
                const int ey = ly0 - static_cast<int>(fem_node_offset(c, 1));
                const int ez = lz0 - static_cast<int>(fem_node_offset(c, 2));
                T u[NB_NODES];
                #pragma unroll
                for (int n = 0; n < NB_NODES; ++n) {
                    const int sidx =
                        (ex + static_cast<int>(fem_node_offset(n, 0))) +
                        SX * ((ey + static_cast<int>(fem_node_offset(n, 1))) +
                              SY * (ez + static_cast<int>(
                                             fem_node_offset(n, 2))));
                    u[n] = s_rho[sidx];
                }
                #pragma unroll 1
                for (int q = 0; q < static_cast<int>(Q::NbQuad); ++q) {
                    T rho_q{};
                    #pragma unroll
                    for (int n = 0; n < NB_NODES; ++n) {
                        rho_q += TAB.N[q][n] * u[n];
                    }
                    const T w{TAB.W[q]};
                    T p{rho_q};  // rho_q^(k-1), climbing with k
                    if (c == 0) {
                        T pv{rho_q * rho_q};
                        #pragma unroll
                        for (int j = 0; j < NB_NODAL_MOMENTS; ++j) {
                            value[j] += w * pv;
                            pv *= rho_q;
                        }
                    }
                    const T wN{w * TAB.N[q][c]};
                    #pragma unroll
                    for (int j = 0; j < NB_NODAL_MOMENTS; ++j) {
                        grad[j] += wN * static_cast<T>(j + FIRST_NODAL_MOMENT) *
                                   p;
                        p *= rho_q;
                    }
                }
            }

            if (!in_grid) { return; }
            const Index_t m0{ix * mom_stride_x + iy * mom_stride_y +
                             iz * mom_stride_z};
            #pragma unroll
            for (int j = 0; j < NB_NODAL_MOMENTS; ++j) {
                moments[m0 + j * mom_stride_c] = value[j] * cell_volume;
                moment_gradients[m0 + j * mom_stride_c] =
                    grad[j] * cell_volume;
            }
        }

        //! 2D kernel; same structure, 3x3 rule and 4 corners.
        template <typename T, class Element>
        __global__ __launch_bounds__(MOMENT_TILE_2D_X * MOMENT_TILE_2D_Y,
                                     2) void
        nodal_moments_2d_kernel(const T * __restrict__ rho,
                                T * __restrict__ moments,
                                T * __restrict__ moment_gradients, Index_t nx,
                                Index_t ny, Index_t rho_stride_x,
                                Index_t rho_stride_y, Index_t mom_stride_x,
                                Index_t mom_stride_y, Index_t mom_stride_c,
                                T cell_volume) {
            using Q = MomentQuadrature<Element>;
            constexpr int NB_NODES = static_cast<int>(Q::NbNodes);
            constexpr auto TAB = make_quad_table<T, Q>();

            constexpr int SX = MOMENT_TILE_2D_X + 2;
            constexpr int SY = MOMENT_TILE_2D_Y + 2;
            __shared__ T s_rho[SY * SX];

            const int bx = blockIdx.x * MOMENT_TILE_2D_X;
            const int by = blockIdx.y * MOMENT_TILE_2D_Y;
            const int tid = threadIdx.x + MOMENT_TILE_2D_X * threadIdx.y;
            constexpr int NB_THREADS = MOMENT_TILE_2D_X * MOMENT_TILE_2D_Y;

            for (int idx = tid; idx < SY * SX; idx += NB_THREADS) {
                const int lx = idx % SX;
                const int ly = idx / SX;
                const int gx = min(max(bx - 1 + lx, -1), static_cast<int>(nx));
                const int gy = min(max(by - 1 + ly, -1), static_cast<int>(ny));
                s_rho[idx] = rho[gx * rho_stride_x + gy * rho_stride_y];
            }
            __syncthreads();

            const Index_t ix = bx + threadIdx.x;
            const Index_t iy = by + threadIdx.y;
            const bool in_grid = (ix < nx && iy < ny);
            const int lx0 = threadIdx.x + 1;
            const int ly0 = threadIdx.y + 1;

            T value[NB_NODAL_MOMENTS]{};
            T grad[NB_NODAL_MOMENTS]{};

            for (int c = 0; c < NB_NODES; ++c) {
                const int ex = lx0 - static_cast<int>(fem_node_offset(c, 0));
                const int ey = ly0 - static_cast<int>(fem_node_offset(c, 1));
                T u[NB_NODES];
                #pragma unroll
                for (int n = 0; n < NB_NODES; ++n) {
                    u[n] = s_rho[(ex + static_cast<int>(
                                           fem_node_offset(n, 0))) +
                                 SX * (ey + static_cast<int>(
                                                fem_node_offset(n, 1)))];
                }
                #pragma unroll 1
                for (int q = 0; q < static_cast<int>(Q::NbQuad); ++q) {
                    T rho_q{};
                    #pragma unroll
                    for (int n = 0; n < NB_NODES; ++n) {
                        rho_q += TAB.N[q][n] * u[n];
                    }
                    const T w{TAB.W[q]};
                    T p{rho_q};
                    if (c == 0) {
                        T pv{rho_q * rho_q};
                        #pragma unroll
                        for (int j = 0; j < NB_NODAL_MOMENTS; ++j) {
                            value[j] += w * pv;
                            pv *= rho_q;
                        }
                    }
                    const T wN{w * TAB.N[q][c]};
                    #pragma unroll
                    for (int j = 0; j < NB_NODAL_MOMENTS; ++j) {
                        grad[j] += wN * static_cast<T>(j + FIRST_NODAL_MOMENT) *
                                   p;
                        p *= rho_q;
                    }
                }
            }

            if (!in_grid) { return; }
            const Index_t m0{ix * mom_stride_x + iy * mom_stride_y};
            #pragma unroll
            for (int j = 0; j < NB_NODAL_MOMENTS; ++j) {
                moments[m0 + j * mom_stride_c] = value[j] * cell_volume;
                moment_gradients[m0 + j * mom_stride_c] =
                    grad[j] * cell_volume;
            }
        }

        template <typename T, class Element>
        void nodal_moments_2d_gpu(const T * rho, T * moments,
                                  T * moment_gradients, Index_t nx, Index_t ny,
                                  Index_t rho_stride_x, Index_t rho_stride_y,
                                  Index_t mom_stride_x, Index_t mom_stride_y,
                                  Index_t mom_stride_c, T cell_volume) {
            if (nx <= 0 || ny <= 0) { return; }
            dim3 block(MOMENT_TILE_2D_X, MOMENT_TILE_2D_Y);
            dim3 grid((nx + block.x - 1) / block.x,
                      (ny + block.y - 1) / block.y);
            GPU_LAUNCH_KERNEL((nodal_moments_2d_kernel<T, Element>), grid, block, rho,
                              moments, moment_gradients, nx, ny, rho_stride_x,
                              rho_stride_y, mom_stride_x, mom_stride_y,
                              mom_stride_c, cell_volume);
        }

        template <typename T, class Element>
        void nodal_moments_3d_gpu(const T * rho, T * moments,
                                  T * moment_gradients, Index_t nx, Index_t ny,
                                  Index_t nz, Index_t rho_stride_x,
                                  Index_t rho_stride_y, Index_t rho_stride_z,
                                  Index_t mom_stride_x, Index_t mom_stride_y,
                                  Index_t mom_stride_z, Index_t mom_stride_c,
                                  T cell_volume) {
            if (nx <= 0 || ny <= 0 || nz <= 0) { return; }
            dim3 block(MOMENT_TILE_X, MOMENT_TILE_Y, MOMENT_TILE_Z);
            dim3 grid((nx + block.x - 1) / block.x,
                      (ny + block.y - 1) / block.y,
                      (nz + block.z - 1) / block.z);
            GPU_LAUNCH_KERNEL((nodal_moments_3d_kernel<T, Element>), grid, block, rho,
                              moments, moment_gradients, nx, ny, nz,
                              rho_stride_x, rho_stride_y, rho_stride_z,
                              mom_stride_x, mom_stride_y, mom_stride_z,
                              mom_stride_c, cell_volume);
        }

#define MUGRID_INSTANTIATE_MOMENTS_GPU_2D(T, E)                               \
    template void nodal_moments_2d_gpu<T, E>(                                 \
        const T *, T *, T *, Index_t, Index_t, Index_t, Index_t, Index_t,     \
        Index_t, Index_t, T);
#define MUGRID_INSTANTIATE_MOMENTS_GPU_3D(T, E)                               \
    template void nodal_moments_3d_gpu<T, E>(                                 \
        const T *, T *, T *, Index_t, Index_t, Index_t, Index_t, Index_t,     \
        Index_t, Index_t, Index_t, Index_t, Index_t, T);
        MUGRID_INSTANTIATE_MOMENTS_GPU_2D(Real, Q1Quad2D)
        MUGRID_INSTANTIATE_MOMENTS_GPU_2D(Real, P1Tri2D)
        MUGRID_INSTANTIATE_MOMENTS_GPU_2D(Real32, Q1Quad2D)
        MUGRID_INSTANTIATE_MOMENTS_GPU_2D(Real32, P1Tri2D)
        MUGRID_INSTANTIATE_MOMENTS_GPU_3D(Real, Q1Hex3D)
        MUGRID_INSTANTIATE_MOMENTS_GPU_3D(Real, P1Tet3D)
        MUGRID_INSTANTIATE_MOMENTS_GPU_3D(Real32, Q1Hex3D)
        MUGRID_INSTANTIATE_MOMENTS_GPU_3D(Real32, P1Tet3D)
#undef MUGRID_INSTANTIATE_MOMENTS_GPU_2D
#undef MUGRID_INSTANTIATE_MOMENTS_GPU_3D

    }  // namespace nodal_moment_kernels
}  // namespace muGrid
