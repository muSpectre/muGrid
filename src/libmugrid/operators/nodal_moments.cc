/**
 * @file   nodal_moments.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   16 Sep 2026
 *
 * @brief  Host kernels for NodalMomentOperator. See nodal_moments.hh for the
 *         interface and the rationale.
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

namespace muGrid {
    namespace nodal_moment_kernels {

        namespace {

            /**
             * @brief One cell's contribution, evaluated in registers.
             *
             * Reads the cell's NbNodes corner values through @p corner (a
             * callable mapping a corner index to its value), runs the moment
             * quadrature, and accumulates
             *   - `value[j] += w_q rho_q^(j+2)` when @p WantValue, and
             *   - `grad[j]  += w_q (j+2) rho_q^(j+1) N[q][GradCorner]`
             * The quadrature point is never stored: this is the whole reason
             * the kernel exists, since the array-at-a-time form materialises
             * NbQuad copies of the grid instead.
             *
             * @p grad_corner is the index this cell's owner occupies in it, or
             * -1 to skip the gradient.
             */
            template <class Quadrature, typename T, class Corner>
            inline void accumulate_cell(const Corner & corner,
                                        Index_t grad_corner, T * value,
                                        T * grad) {
                for (Index_t q = 0; q < Quadrature::NbQuad; ++q) {
                    T rho_q{};
                    for (Index_t n = 0; n < Quadrature::NbNodes; ++n) {
                        rho_q += static_cast<T>(Quadrature::shape(q, n)) *
                                 corner(n);
                    }
                    const T w{static_cast<T>(Quadrature::weight(q))};
                    // rho_q^(k-1), climbing k = FIRST .. FIRST + NB - 1, so
                    // each moment costs one multiply rather than a pow().
                    T p{rho_q};  // rho_q^(FIRST_NODAL_MOMENT - 1)
                    if (value != nullptr) {
                        T pv{rho_q * rho_q};  // rho_q^FIRST_NODAL_MOMENT
                        for (Index_t j = 0; j < NB_NODAL_MOMENTS; ++j) {
                            value[j] += w * pv;
                            pv *= rho_q;
                        }
                    }
                    if (grad_corner >= 0) {
                        const T N{static_cast<T>(
                            Quadrature::shape(q, grad_corner))};
                        const T wN{w * N};
                        for (Index_t j = 0; j < NB_NODAL_MOMENTS; ++j) {
                            const T k{
                                static_cast<T>(j + FIRST_NODAL_MOMENT)};
                            grad[j] += wN * k * p;
                            p *= rho_q;
                        }
                    }
                }
            }

        }  // namespace

        template <typename T, class Element>
        void nodal_moments_2d_host(const T * rho, T * moments,
                                   T * moment_gradients, Index_t nx,
                                   Index_t ny, Index_t rho_stride_x,
                                   Index_t rho_stride_y, Index_t mom_stride_x,
                                   Index_t mom_stride_y, Index_t mom_stride_c,
                                   T cell_volume) {
            using Q = MomentQuadrature<Element>;
            constexpr Index_t NB_NODES = Q::NbNodes;
            // Node n of a cell sits at + (offset_x, offset_y) from its origin.
            std::array<Index_t, NB_NODES> node_shift{};
            for (Index_t n = 0; n < NB_NODES; ++n) {
                node_shift[n] = fem_node_offset(n, 0) * rho_stride_x +
                                fem_node_offset(n, 1) * rho_stride_y;
            }

            for (Index_t iy = 0; iy < ny; ++iy) {
                for (Index_t ix = 0; ix < nx; ++ix) {
                    const Index_t r0{ix * rho_stride_x + iy * rho_stride_y};
                    T value[NB_NODAL_MOMENTS]{};
                    T grad[NB_NODAL_MOMENTS]{};

                    // This thread's own cell carries the value; every cell
                    // touching this node (offsets 0 and -1 per axis, i.e. the
                    // cell in which the node is corner c) carries a gradient
                    // term. Corner c of cell (node - offset(c)) is this node.
                    for (Index_t c = 0; c < NB_NODES; ++c) {
                        const Index_t e0{
                            r0 - fem_node_offset(c, 0) * rho_stride_x -
                            fem_node_offset(c, 1) * rho_stride_y};
                        auto corner = [&](Index_t n) {
                            return rho[e0 + node_shift[n]];
                        };
                        accumulate_cell<Q, T>(
                            corner, c, (c == 0 ? value : nullptr), grad);
                    }

                    const Index_t m0{ix * mom_stride_x + iy * mom_stride_y};
                    for (Index_t j = 0; j < NB_NODAL_MOMENTS; ++j) {
                        moments[m0 + j * mom_stride_c] = value[j] * cell_volume;
                        moment_gradients[m0 + j * mom_stride_c] =
                            grad[j] * cell_volume;
                    }
                }
            }
        }

        template <typename T, class Element>
        void nodal_moments_3d_host(const T * rho, T * moments,
                                   T * moment_gradients, Index_t nx,
                                   Index_t ny, Index_t nz,
                                   Index_t rho_stride_x, Index_t rho_stride_y,
                                   Index_t rho_stride_z, Index_t mom_stride_x,
                                   Index_t mom_stride_y, Index_t mom_stride_z,
                                   Index_t mom_stride_c, T cell_volume) {
            using Q = MomentQuadrature<Element>;
            constexpr Index_t NB_NODES = Q::NbNodes;
            std::array<Index_t, NB_NODES> node_shift{};
            for (Index_t n = 0; n < NB_NODES; ++n) {
                node_shift[n] = fem_node_offset(n, 0) * rho_stride_x +
                                fem_node_offset(n, 1) * rho_stride_y +
                                fem_node_offset(n, 2) * rho_stride_z;
            }

            for (Index_t iz = 0; iz < nz; ++iz) {
                for (Index_t iy = 0; iy < ny; ++iy) {
                    for (Index_t ix = 0; ix < nx; ++ix) {
                        const Index_t r0{ix * rho_stride_x +
                                         iy * rho_stride_y +
                                         iz * rho_stride_z};
                        T value[NB_NODAL_MOMENTS]{};
                        T grad[NB_NODAL_MOMENTS]{};

                        for (Index_t c = 0; c < NB_NODES; ++c) {
                            const Index_t e0{
                                r0 - fem_node_offset(c, 0) * rho_stride_x -
                                fem_node_offset(c, 1) * rho_stride_y -
                                fem_node_offset(c, 2) * rho_stride_z};
                            auto corner = [&](Index_t n) {
                                return rho[e0 + node_shift[n]];
                            };
                            accumulate_cell<Q, T>(
                                corner, c, (c == 0 ? value : nullptr), grad);
                        }

                        const Index_t m0{ix * mom_stride_x +
                                         iy * mom_stride_y +
                                         iz * mom_stride_z};
                        for (Index_t j = 0; j < NB_NODAL_MOMENTS; ++j) {
                            moments[m0 + j * mom_stride_c] =
                                value[j] * cell_volume;
                            moment_gradients[m0 + j * mom_stride_c] =
                                grad[j] * cell_volume;
                        }
                    }
                }
            }
        }

        // Explicit instantiations: both precisions x both element types.
#define MUGRID_INSTANTIATE_MOMENTS_2D(T, E)                                   \
    template void nodal_moments_2d_host<T, E>(                                \
        const T *, T *, T *, Index_t, Index_t, Index_t, Index_t, Index_t,     \
        Index_t, Index_t, T);
#define MUGRID_INSTANTIATE_MOMENTS_3D(T, E)                                   \
    template void nodal_moments_3d_host<T, E>(                                \
        const T *, T *, T *, Index_t, Index_t, Index_t, Index_t, Index_t,     \
        Index_t, Index_t, Index_t, Index_t, Index_t, T);
        MUGRID_INSTANTIATE_MOMENTS_2D(Real, Q1Quad2D)
        MUGRID_INSTANTIATE_MOMENTS_2D(Real, P1Tri2D)
        MUGRID_INSTANTIATE_MOMENTS_2D(Real32, Q1Quad2D)
        MUGRID_INSTANTIATE_MOMENTS_2D(Real32, P1Tri2D)
        MUGRID_INSTANTIATE_MOMENTS_3D(Real, Q1Hex3D)
        MUGRID_INSTANTIATE_MOMENTS_3D(Real, P1Tet3D)
        MUGRID_INSTANTIATE_MOMENTS_3D(Real32, Q1Hex3D)
        MUGRID_INSTANTIATE_MOMENTS_3D(Real32, P1Tet3D)
#undef MUGRID_INSTANTIATE_MOMENTS_2D
#undef MUGRID_INSTANTIATE_MOMENTS_3D

    }  // namespace nodal_moment_kernels

    template class NodalMomentOperator<2>;
    template class NodalMomentOperator<3>;

}  // namespace muGrid
