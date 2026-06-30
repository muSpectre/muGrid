/**
 * @file   fem_gradient_operator_gpu.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Dec 2025
 *
 * @brief  Unified CUDA/HIP implementation of the FEM gradient operator
 *
 * Element-generic GPU kernels: the element supplies the reference
 * shape-function-gradient table B[q][d][n] (see fem_element.hh) and the
 * kernels are templated on it. The contraction is unrolled at compile time via
 * `if constexpr` recursion, so B[q][d][n] is a constant expression — the
 * compiler folds the structural zeros / ±1 of a simplex (and skips the global
 * loads multiplied by a zero coefficient), reproducing the old hand-unrolled
 * kernels while also supporting any new element (e.g. Q1) with no new code.
 *
 * Both apply (gradient, nodal → quadrature) and transpose (divergence,
 * quadrature → nodal) use a gather pattern — one thread per output point —
 * so no atomics are needed. The kernel code is identical for CUDA and HIP;
 * only the launch mechanism differs, abstracted via macros.
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

#include "fem_gradient.hh"

#include "memory/gpu_runtime.hh"

namespace muGrid {
namespace fem_gradient_kernels {

    // Block sizes for GPU kernels
    constexpr int BLOCK_SIZE_2D = 16;
    constexpr int BLOCK_SIZE_3D = 8;

    namespace {

        // sum_n B[q][d][n] * u[n], unrolled over n at compile time. Terms with a
        // structurally-zero B are dropped (no multiply), and the gathered u
        // values are cheap corner reads, so this matches the hand-written code.
        template <class E, Index_t q, Index_t d, Index_t n = 0>
        __device__ __forceinline__ Real bsum(const Real * u) {
            if constexpr (n == E::NbNodes) {
                return Real(0);
            } else {
                constexpr Real b = E::B[q][d][n];
                if constexpr (b != Real(0)) {
                    return b * u[n] + bsum<E, q, d, n + 1>(u);
                } else {
                    return bsum<E, q, d, n + 1>(u);
                }
            }
        }

        // Write all NbQuad×Dim gradient components for one pixel, unrolled.
        template <class E, Index_t k = 0>
        __device__ __forceinline__ void
        grad_store(const Real * u, Real * grad, Index_t base, Index_t gsq,
                   Index_t gsd, const Real * inv_h, bool increment) {
            constexpr Index_t Dim = E::SpatialDim;
            if constexpr (k == E::NbQuad * Dim) {
                return;
            } else {
                constexpr Index_t q = k / Dim;
                constexpr Index_t d = k % Dim;
                Real v = bsum<E, q, d>(u) * inv_h[d];
                Index_t idx = base + q * gsq + d * gsd;
                if (increment) {
                    grad[idx] += v;
                } else {
                    grad[idx] = v;
                }
                grad_store<E, k + 1>(u, grad, base, gsq, gsd, inv_h, increment);
            }
        }

        // Contribution of one neighbouring element (with this node as local node
        // m) to the divergence at a node: sum_{q,d} coeff[q][d] B[q][d][m]
        // g[base + q·gsq + d·gsd], unrolled, skipping the global loads whose B
        // is structurally zero.
        template <class E, Index_t m, Index_t k = 0>
        __device__ __forceinline__ Real
        div_node(const Real * g, Index_t base, Index_t gsq, Index_t gsd,
                 const Real (&coeff)[E::NbQuad][E::SpatialDim]) {
            constexpr Index_t Dim = E::SpatialDim;
            if constexpr (k == E::NbQuad * Dim) {
                return Real(0);
            } else {
                constexpr Index_t q = k / Dim;
                constexpr Index_t d = k % Dim;
                constexpr Real b = E::B[q][d][m];
                Real rest = div_node<E, m, k + 1>(g, base, gsq, gsd, coeff);
                if constexpr (b != Real(0)) {
                    return coeff[q][d] * b * g[base + q * gsq + d * gsd] + rest;
                } else {
                    return rest;
                }
            }
        }

    }  // namespace

    // =====================================================================
    // Element-generic kernels (gather pattern, one thread per output point)
    // =====================================================================

    //! Per-element launch parameters (small POD, passed to the kernel by value;
    //! raw arrays would decay to host pointers, so they are wrapped here).
    template <class E>
    struct GradParams {
        Index_t nb[E::SpatialDim];
        Index_t nstride[E::SpatialDim];
        Index_t gstride[E::SpatialDim];
        Index_t gsq, gsd;
        Real inv_h[E::SpatialDim];
    };
    template <class E>
    struct DivParams {
        Index_t nb[E::SpatialDim];
        Index_t nstride[E::SpatialDim];
        Index_t gstride[E::SpatialDim];
        Index_t gsq, gsd;
        Real coeff[E::NbQuad][E::SpatialDim];
    };

    //! Gradient: one thread per pixel/voxel, gather its 2^Dim corner nodes.
    template <class E>
    __global__ void fem_gradient_kernel(const Real * MUGRID_RESTRICT nodal,
                                        Real * MUGRID_RESTRICT grad,
                                        GradParams<E> P, bool increment) {
        constexpr Index_t Dim = E::SpatialDim;
        constexpr Index_t NbNodes = E::NbNodes;
        Index_t c[Dim];
        c[0] = blockIdx.x * blockDim.x + threadIdx.x;
        c[1] = blockIdx.y * blockDim.y + threadIdx.y;
        if constexpr (Dim == 3) {
            c[2] = blockIdx.z * blockDim.z + threadIdx.z;
        }
        for (Index_t d = 0; d < Dim; ++d) {
            if (c[d] >= P.nb[d] - 1) return;  // need the +1 corner
        }
        Index_t nbase = 0, gbase = 0;
        for (Index_t d = 0; d < Dim; ++d) {
            nbase += c[d] * P.nstride[d];
            gbase += c[d] * P.gstride[d];
        }
        Real u[NbNodes];
        for (Index_t n = 0; n < NbNodes; ++n) {
            Index_t off = 0;
            for (Index_t d = 0; d < Dim; ++d) {
                off += ((n >> d) & 1) * P.nstride[d];
            }
            u[n] = nodal[nbase + off];
        }
        grad_store<E>(u, grad, gbase, P.gsq, P.gsd, P.inv_h, increment);
    }

    //! Divergence: one thread per node, gather from its 2^Dim adjacent
    //! elements (no atomics). Element po (offset p_d = -((po>>d)&1)) has this
    //! node as its local node po.
    template <class E, Index_t po = 0>
    __device__ __forceinline__ Real
    div_gather(const Real * g, const Index_t * c, const DivParams<E> & P) {
        constexpr Index_t Dim = E::SpatialDim;
        if constexpr (po == (Index_t(1) << Dim)) {
            return Real(0);
        } else {
            bool ok = true;
            Index_t base = 0;
            for (Index_t d = 0; d < Dim; ++d) {
                Index_t cd = c[d] - static_cast<Index_t>((po >> d) & 1);
                if (cd < 0 || cd >= P.nb[d] - 1) ok = false;
                base += cd * P.gstride[d];
            }
            Real here = ok ? div_node<E, po>(g, base, P.gsq, P.gsd, P.coeff)
                           : Real(0);
            return here + div_gather<E, po + 1>(g, c, P);
        }
    }

    template <class E>
    __global__ void fem_divergence_kernel(const Real * MUGRID_RESTRICT g,
                                          Real * MUGRID_RESTRICT nodal,
                                          DivParams<E> P, bool increment) {
        constexpr Index_t Dim = E::SpatialDim;
        Index_t c[Dim];
        c[0] = blockIdx.x * blockDim.x + threadIdx.x;
        c[1] = blockIdx.y * blockDim.y + threadIdx.y;
        if constexpr (Dim == 3) {
            c[2] = blockIdx.z * blockDim.z + threadIdx.z;
        }
        for (Index_t d = 0; d < Dim; ++d) {
            if (c[d] >= P.nb[d]) return;
        }
        Real contrib = div_gather<E>(g, c, P);
        Index_t idx = 0;
        for (Index_t d = 0; d < Dim; ++d) {
            idx += c[d] * P.nstride[d];
        }
        if (increment) {
            nodal[idx] += contrib;
        } else {
            nodal[idx] = contrib;
        }
    }

    // =====================================================================
    // Launch wrappers (declared in fem_gradient.hh, called by the operator)
    // =====================================================================

    template <class E>
    void fem_gradient_gpu(const Real * nodal, Real * grad, const Index_t * nb,
                          const Index_t * nstride, const Index_t * gstride,
                          Index_t gsq, Index_t gsd, const Real * h, Real alpha,
                          bool increment) {
        constexpr Index_t Dim = E::SpatialDim;
        GradParams<E> P;
        for (Index_t d = 0; d < Dim; ++d) {
            P.nb[d] = nb[d];
            P.nstride[d] = nstride[d];
            P.gstride[d] = gstride[d];
            P.inv_h[d] = alpha / h[d];
        }
        P.gsq = gsq;
        P.gsd = gsd;
        if constexpr (Dim == 2) {
            dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
            dim3 grid((nb[0] - 1 + block.x - 1) / block.x,
                      (nb[1] - 1 + block.y - 1) / block.y);
            GPU_LAUNCH_KERNEL(fem_gradient_kernel<E>, grid, block, nodal, grad,
                              P, increment);
        } else {
            dim3 block(BLOCK_SIZE_3D, BLOCK_SIZE_3D, BLOCK_SIZE_3D);
            dim3 grid((nb[0] - 1 + block.x - 1) / block.x,
                      (nb[1] - 1 + block.y - 1) / block.y,
                      (nb[2] - 1 + block.z - 1) / block.z);
            GPU_LAUNCH_KERNEL(fem_gradient_kernel<E>, grid, block, nodal, grad,
                              P, increment);
        }
        const char * err{gpu_last_error()};
        if (err != nullptr) {
            throw RuntimeError("GPU kernel launch failed: " + std::string(err));
        }
    }

    template <class E>
    void fem_divergence_gpu(const Real * grad, Real * nodal, const Index_t * nb,
                            const Index_t * nstride, const Index_t * gstride,
                            Index_t gsq, Index_t gsd, const Real * h,
                            const Real * quad_weights, Real alpha,
                            bool increment) {
        constexpr Index_t Dim = E::SpatialDim;
        DivParams<E> P;
        for (Index_t d = 0; d < Dim; ++d) {
            P.nb[d] = nb[d];
            P.nstride[d] = nstride[d];
            P.gstride[d] = gstride[d];
        }
        P.gsq = gsq;
        P.gsd = gsd;
        for (Index_t q = 0; q < E::NbQuad; ++q) {
            for (Index_t d = 0; d < Dim; ++d) {
                P.coeff[q][d] = alpha * quad_weights[q] / h[d];
            }
        }
        if constexpr (Dim == 2) {
            dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
            dim3 grid((nb[0] + block.x - 1) / block.x,
                      (nb[1] + block.y - 1) / block.y);
            GPU_LAUNCH_KERNEL(fem_divergence_kernel<E>, grid, block, grad, nodal,
                              P, increment);
        } else {
            dim3 block(BLOCK_SIZE_3D, BLOCK_SIZE_3D, BLOCK_SIZE_3D);
            dim3 grid((nb[0] + block.x - 1) / block.x,
                      (nb[1] + block.y - 1) / block.y,
                      (nb[2] + block.z - 1) / block.z);
            GPU_LAUNCH_KERNEL(fem_divergence_kernel<E>, grid, block, grad, nodal,
                              P, increment);
        }
        const char * err{gpu_last_error()};
        if (err != nullptr) {
            throw RuntimeError("GPU kernel launch failed: " + std::string(err));
        }
    }

    // Explicit instantiations for the supported elements.
    template void fem_gradient_gpu<P1Tri2D>(
        const Real *, Real *, const Index_t *, const Index_t *,
        const Index_t *, Index_t, Index_t, const Real *, Real, bool);
    template void fem_gradient_gpu<P1Tet3D>(
        const Real *, Real *, const Index_t *, const Index_t *,
        const Index_t *, Index_t, Index_t, const Real *, Real, bool);
    template void fem_divergence_gpu<P1Tri2D>(
        const Real *, Real *, const Index_t *, const Index_t *,
        const Index_t *, Index_t, Index_t, const Real *, const Real *, Real,
        bool);
    template void fem_divergence_gpu<P1Tet3D>(
        const Real *, Real *, const Index_t *, const Index_t *,
        const Index_t *, Index_t, Index_t, const Real *, const Real *, Real,
        bool);
    template void fem_gradient_gpu<Q1Quad2D>(
        const Real *, Real *, const Index_t *, const Index_t *,
        const Index_t *, Index_t, Index_t, const Real *, Real, bool);
    template void fem_gradient_gpu<Q1Hex3D>(
        const Real *, Real *, const Index_t *, const Index_t *,
        const Index_t *, Index_t, Index_t, const Real *, Real, bool);
    template void fem_divergence_gpu<Q1Quad2D>(
        const Real *, Real *, const Index_t *, const Index_t *,
        const Index_t *, Index_t, Index_t, const Real *, const Real *, Real,
        bool);
    template void fem_divergence_gpu<Q1Hex3D>(
        const Real *, Real *, const Index_t *, const Index_t *,
        const Index_t *, Index_t, Index_t, const Real *, const Real *, Real,
        bool);

}  // namespace fem_gradient_kernels
}  // namespace muGrid
