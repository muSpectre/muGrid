/**
 * @file   variable_stencil_gpu.cc
 *
 * @author Martin Ladecký <martin.ladecky@imtek.uni-freiburg.de>
 *
 * @date   16 Nov 2026
 *
 * @brief  Device kernel of the variable-stencil operator (CUDA/HIP)
 *
 * Copyright © 2026 Martin Ladecký
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

#include "core/types.hh"
#include "operators/variable_stencil.hh"

#include "memory/gpu_runtime.hh"

namespace muGrid {
    namespace variable_stencil {

        namespace {

            //! threads per block (a multiple of the warp and wavefront width)
            constexpr int BLOCK_SIZE{256};

            //! per stored offset: scalar offsets of the neighbour node and of
            //! its stencil, passed to the kernel by value
            struct DeviceOffsets {
                Index_t node[13];
                Index_t stencil[13];
            };

            /**
             * One thread per interior node, x fastest, so that neighbouring
             * threads read neighbouring addresses of every component plane
             * (device fields are structures of arrays). The arithmetic is the
             * host kernel's: the centre block from its upper triangle, then
             * A_k(n) x(n + s_k) + A_k(n - s_k)^T x(n - s_k) per stored offset.
             */
            template <typename T, int NC>
            __global__ void apply_kernel(
                const T * MUGRID_RESTRICT stencil, const T * MUGRID_RESTRICT x,
                T * MUGRID_RESTRICT y, Index_t n0, Index_t n1, Index_t n2,
                Index_t ns0, Index_t ns1, Index_t ns2, Index_t nd, Index_t ss0,
                Index_t ss1, Index_t ss2, Index_t sd, int nb_half,
                DeviceOffsets offsets, T alpha, bool increment) {
                const Index_t index{static_cast<Index_t>(blockIdx.x) *
                                        blockDim.x +
                                    threadIdx.x};
                if (index >= n0 * n1 * n2) {
                    return;
                }
                const Index_t k0{index % n0};
                const Index_t rest{index / n0};
                const Index_t k1{rest % n1};
                const Index_t k2{rest / n1};
                const Index_t n{k0 * ns0 + k1 * ns1 + k2 * ns2};
                const Index_t e{k0 * ss0 + k1 * ss1 + k2 * ss2};
                constexpr int NN{NC * NC};

                T acc[NC];
                for (int i{0}; i < NC; ++i) {
                    acc[i] = T{0};
                }
                Index_t t{static_cast<Index_t>(nb_half) * NN};
                for (int i{0}; i < NC; ++i) {
                    for (int j{i}; j < NC; ++j, ++t) {
                        const T c{stencil[e + t * sd]};
                        acc[i] += c * x[n + j * nd];
                        if (j != i) {
                            acc[j] += c * x[n + i * nd];
                        }
                    }
                }
                for (int k{0}; k < nb_half; ++k) {
                    const T * a_plus{stencil + e + k * NN * sd};
                    const T * a_minus{stencil + e - offsets.stencil[k] +
                                      k * NN * sd};
                    const T * x_plus{x + n + offsets.node[k]};
                    const T * x_minus{x + n - offsets.node[k]};
                    for (int i{0}; i < NC; ++i) {
                        for (int j{0}; j < NC; ++j) {
                            acc[i] += a_plus[(i * NC + j) * sd] *
                                          x_plus[j * nd] +
                                      a_minus[(j * NC + i) * sd] *
                                          x_minus[j * nd];
                        }
                    }
                }
                for (int i{0}; i < NC; ++i) {
                    T & out{y[n + i * nd]};
                    out = increment ? out + alpha * acc[i] : alpha * acc[i];
                }
            }

            template <typename T, int NC>
            void launch(const T * stencil, const T * x, T * y, Index_t nb_half,
                        const Geometry & g, T alpha, bool increment) {
                DeviceOffsets offsets{};
                for (Index_t k{0}; k < nb_half; ++k) {
                    offsets.node[k] = g.node_offset[k];
                    offsets.stencil[k] = g.stencil_offset[k];
                }
                const Index_t nb_nodes{g.nb_interior[0] * g.nb_interior[1] *
                                       g.nb_interior[2]};
                if (nb_nodes == 0) {
                    return;
                }
                const dim3 block(BLOCK_SIZE);
                const dim3 grid(
                    static_cast<unsigned int>((nb_nodes + BLOCK_SIZE - 1) /
                                              BLOCK_SIZE));
                auto kernel = apply_kernel<T, NC>;
                GPU_LAUNCH_KERNEL(
                    kernel, grid, block, stencil, x, y, g.nb_interior[0],
                    g.nb_interior[1], g.nb_interior[2], g.node_stride[0],
                    g.node_stride[1], g.node_stride[2], g.node_stride_dof,
                    g.stencil_stride[0], g.stencil_stride[1],
                    g.stencil_stride[2], g.stencil_stride_dof,
                    static_cast<int>(nb_half), offsets, alpha, increment);
            }

        }  // namespace

        template <typename T>
        void apply_gpu(const T * stencil, const T * x, T * y,
                       Index_t nb_components, Index_t nb_half,
                       const Geometry & geometry, T alpha, bool increment) {
            switch (nb_components) {
            case 1:
                launch<T, 1>(stencil, x, y, nb_half, geometry, alpha,
                             increment);
                break;
            case 2:
                launch<T, 2>(stencil, x, y, nb_half, geometry, alpha,
                             increment);
                break;
            case 3:
                launch<T, 3>(stencil, x, y, nb_half, geometry, alpha,
                             increment);
                break;
            default:
                throw RuntimeError{"VariableStencilOperator: nb_components "
                                   "must be 1, 2 or 3"};
            }
        }

        template void apply_gpu<Real>(const Real *, const Real *, Real *,
                                      Index_t, Index_t, const Geometry &, Real,
                                      bool);
        template void apply_gpu<Real32>(const Real32 *, const Real32 *,
                                        Real32 *, Index_t, Index_t,
                                        const Geometry &, Real32, bool);

    }  // namespace variable_stencil
}  // namespace muGrid
