/**
 * @file   variable_stencil.cc
 *
 * @author Martin Ladecký <martin.ladecky@imtek.uni-freiburg.de>
 *
 * @date   16 Nov 2026
 *
 * @brief  Host kernel of the variable-stencil operator
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

#include "variable_stencil.hh"

namespace muGrid {

    namespace variable_stencil {

        namespace {

            /**
             * One pass over the interior nodes, the number of components NC
             * known at compile time so the block products unroll. Per node:
             * the centre block (stored as its upper triangle), then for every
             * stored offset s_k the block A_k(n) applied to x(n + s_k) and the
             * transposed block of the neighbour, A_k(n - s_k)^T, applied to
             * x(n - s_k).
             */
            template <typename T, Index_t NC>
            void apply_kernel(const T * MUGRID_RESTRICT stencil,
                              const T * MUGRID_RESTRICT x,
                              T * MUGRID_RESTRICT y, Index_t nb_half,
                              const Geometry & g, T alpha, bool increment) {
                constexpr Index_t NN{NC * NC};
                const Index_t centre{nb_half * NN};
                const Index_t nd{g.node_stride_dof};
                const Index_t sd{g.stencil_stride_dof};
                for (Index_t k2{0}; k2 < g.nb_interior[2]; ++k2) {
                    for (Index_t k1{0}; k1 < g.nb_interior[1]; ++k1) {
                        for (Index_t k0{0}; k0 < g.nb_interior[0]; ++k0) {
                            const Index_t n{k0 * g.node_stride[0] +
                                            k1 * g.node_stride[1] +
                                            k2 * g.node_stride[2]};
                            const Index_t e{k0 * g.stencil_stride[0] +
                                            k1 * g.stencil_stride[1] +
                                            k2 * g.stencil_stride[2]};
                            T acc[NC];
                            for (Index_t i{0}; i < NC; ++i) {
                                acc[i] = T{0};
                            }
                            // centre block, symmetric: upper triangle
                            Index_t t{centre};
                            for (Index_t i{0}; i < NC; ++i) {
                                for (Index_t j{i}; j < NC; ++j, ++t) {
                                    const T c{stencil[e + t * sd]};
                                    acc[i] += c * x[n + j * nd];
                                    if (j != i) {
                                        acc[j] += c * x[n + i * nd];
                                    }
                                }
                            }
                            for (Index_t k{0}; k < nb_half; ++k) {
                                const T * a_plus{stencil + e + k * NN * sd};
                                const T * a_minus{stencil + e -
                                                  g.stencil_offset[k] +
                                                  k * NN * sd};
                                const T * x_plus{x + n + g.node_offset[k]};
                                const T * x_minus{x + n - g.node_offset[k]};
                                for (Index_t i{0}; i < NC; ++i) {
                                    for (Index_t j{0}; j < NC; ++j) {
                                        acc[i] += a_plus[(i * NC + j) * sd] *
                                                      x_plus[j * nd] +
                                                  a_minus[(j * NC + i) * sd] *
                                                      x_minus[j * nd];
                                    }
                                }
                            }
                            for (Index_t i{0}; i < NC; ++i) {
                                T & out{y[n + i * nd]};
                                out = increment ? out + alpha * acc[i]
                                                : alpha * acc[i];
                            }
                        }
                    }
                }
            }

        }  // namespace

        template <typename T>
        void apply_host(const T * MUGRID_RESTRICT stencil,
                        const T * MUGRID_RESTRICT x, T * MUGRID_RESTRICT y,
                        Index_t nb_components, Index_t nb_half,
                        const Geometry & geometry, T alpha, bool increment) {
            switch (nb_components) {
            case 1:
                apply_kernel<T, 1>(stencil, x, y, nb_half, geometry, alpha,
                                   increment);
                break;
            case 2:
                apply_kernel<T, 2>(stencil, x, y, nb_half, geometry, alpha,
                                   increment);
                break;
            case 3:
                apply_kernel<T, 3>(stencil, x, y, nb_half, geometry, alpha,
                                   increment);
                break;
            default:
                throw RuntimeError{"VariableStencilOperator: nb_components "
                                   "must be 1, 2 or 3"};
            }
        }

        template void apply_host<Real>(const Real *, const Real *, Real *,
                                       Index_t, Index_t, const Geometry &,
                                       Real, bool);
        template void apply_host<Real32>(const Real32 *, const Real32 *,
                                         Real32 *, Index_t, Index_t,
                                         const Geometry &, Real32, bool);

    }  // namespace variable_stencil

}  // namespace muGrid
