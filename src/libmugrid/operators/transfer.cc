/**
 * @file   transfer.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Sep 2026
 *
 * @brief  Host kernels for the multigrid grid-transfer operators
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

#include "transfer.hh"

namespace muGrid {

    namespace transfer_kernels {

        template <typename T>
        void prolong_2d_host(const T * MUGRID_RESTRICT coarse,
                             T * MUGRID_RESTRICT fine,
                             Index_t ncx, Index_t ncy, Index_t nb_dof,
                             Index_t csx, Index_t csy, Index_t csd,
                             Index_t fsx, Index_t fsy, Index_t fsd) {
            constexpr T HALF{static_cast<T>(0.5)};
            constexpr T QUARTER{static_cast<T>(0.25)};

            for (Index_t dof{0}; dof < nb_dof; ++dof) {
                const T * c = coarse + dof * csd;
                T * f = fine + dof * fsd;
                for (Index_t cy{0}; cy < ncy; ++cy) {
                    for (Index_t cx{0}; cx < ncx; ++cx) {
                        const Index_t ci{cx * csx + cy * csy};
                        const T c00{c[ci]};
                        const T c10{c[ci + csx]};
                        const T c01{c[ci + csy]};
                        const T c11{c[ci + csx + csy]};

                        // The four fine nodes of this coarse cell. Each fine
                        // node belongs to exactly one coarse cell, so every
                        // one is written exactly once.
                        const Index_t fi{2 * cx * fsx + 2 * cy * fsy};
                        f[fi] = c00;
                        f[fi + fsx] = HALF * (c00 + c10);
                        f[fi + fsy] = HALF * (c00 + c01);
                        f[fi + fsx + fsy] = QUARTER * (c00 + c10 + c01 + c11);
                    }
                }
            }
        }

        template <typename T>
        void prolong_3d_host(const T * MUGRID_RESTRICT coarse,
                             T * MUGRID_RESTRICT fine,
                             Index_t ncx, Index_t ncy, Index_t ncz,
                             Index_t nb_dof,
                             Index_t csx, Index_t csy, Index_t csz, Index_t csd,
                             Index_t fsx, Index_t fsy, Index_t fsz,
                             Index_t fsd) {
            constexpr T HALF{static_cast<T>(0.5)};
            constexpr T QUARTER{static_cast<T>(0.25)};
            constexpr T EIGHTH{static_cast<T>(0.125)};

            for (Index_t dof{0}; dof < nb_dof; ++dof) {
                const T * c = coarse + dof * csd;
                T * f = fine + dof * fsd;
                for (Index_t cz{0}; cz < ncz; ++cz) {
                    for (Index_t cy{0}; cy < ncy; ++cy) {
                        for (Index_t cx{0}; cx < ncx; ++cx) {
                            const Index_t ci{cx * csx + cy * csy + cz * csz};
                            const T c000{c[ci]};
                            const T c100{c[ci + csx]};
                            const T c010{c[ci + csy]};
                            const T c001{c[ci + csz]};
                            const T c110{c[ci + csx + csy]};
                            const T c101{c[ci + csx + csz]};
                            const T c011{c[ci + csy + csz]};
                            const T c111{c[ci + csx + csy + csz]};

                            // The eight fine nodes of this coarse cell: a
                            // vertex copies, an edge midpoint averages two
                            // corners, a face centre four, the cell centre all
                            // eight.
                            const Index_t fi{2 * cx * fsx + 2 * cy * fsy +
                                             2 * cz * fsz};
                            f[fi] = c000;
                            f[fi + fsx] = HALF * (c000 + c100);
                            f[fi + fsy] = HALF * (c000 + c010);
                            f[fi + fsz] = HALF * (c000 + c001);
                            f[fi + fsx + fsy] =
                                QUARTER * (c000 + c100 + c010 + c110);
                            f[fi + fsx + fsz] =
                                QUARTER * (c000 + c100 + c001 + c101);
                            f[fi + fsy + fsz] =
                                QUARTER * (c000 + c010 + c001 + c011);
                            f[fi + fsx + fsy + fsz] =
                                EIGHTH * (c000 + c100 + c010 + c001 + c110 +
                                          c101 + c011 + c111);
                        }
                    }
                }
            }
        }

        template <typename T>
        void restrict_2d_host(const T * MUGRID_RESTRICT fine,
                              T * MUGRID_RESTRICT coarse,
                              Index_t ncx, Index_t ncy, Index_t nb_dof,
                              Index_t fsx, Index_t fsy, Index_t fsd,
                              Index_t csx, Index_t csy, Index_t csd) {
            // Tensor product of [1/2, 1, 1/2]; indexed by offset + 1.
            const T w[3]{static_cast<T>(0.5), static_cast<T>(1),
                         static_cast<T>(0.5)};

            for (Index_t dof{0}; dof < nb_dof; ++dof) {
                const T * f = fine + dof * fsd;
                T * c = coarse + dof * csd;
                for (Index_t cy{0}; cy < ncy; ++cy) {
                    for (Index_t cx{0}; cx < ncx; ++cx) {
                        const Index_t fi{2 * cx * fsx + 2 * cy * fsy};
                        T acc{0};
                        for (Index_t oy{-1}; oy <= 1; ++oy) {
                            const T wy{w[oy + 1]};
                            for (Index_t ox{-1}; ox <= 1; ++ox) {
                                acc += w[ox + 1] * wy *
                                       f[fi + ox * fsx + oy * fsy];
                            }
                        }
                        c[cx * csx + cy * csy] = acc;
                    }
                }
            }
        }

        template <typename T>
        void restrict_3d_host(const T * MUGRID_RESTRICT fine,
                              T * MUGRID_RESTRICT coarse,
                              Index_t ncx, Index_t ncy, Index_t ncz,
                              Index_t nb_dof,
                              Index_t fsx, Index_t fsy, Index_t fsz,
                              Index_t fsd,
                              Index_t csx, Index_t csy, Index_t csz,
                              Index_t csd) {
            const T w[3]{static_cast<T>(0.5), static_cast<T>(1),
                         static_cast<T>(0.5)};

            for (Index_t dof{0}; dof < nb_dof; ++dof) {
                const T * f = fine + dof * fsd;
                T * c = coarse + dof * csd;
                for (Index_t cz{0}; cz < ncz; ++cz) {
                    for (Index_t cy{0}; cy < ncy; ++cy) {
                        for (Index_t cx{0}; cx < ncx; ++cx) {
                            const Index_t fi{2 * cx * fsx + 2 * cy * fsy +
                                             2 * cz * fsz};
                            // The 27-point gather is the separable rule
                            // written out; it costs 27/8 multiply-adds per
                            // fine node, which is not worth splitting into
                            // three passes plus the temporaries they need.
                            T acc{0};
                            for (Index_t oz{-1}; oz <= 1; ++oz) {
                                const T wz{w[oz + 1]};
                                for (Index_t oy{-1}; oy <= 1; ++oy) {
                                    const T wyz{w[oy + 1] * wz};
                                    for (Index_t ox{-1}; ox <= 1; ++ox) {
                                        acc += w[ox + 1] * wyz *
                                               f[fi + ox * fsx + oy * fsy +
                                                 oz * fsz];
                                    }
                                }
                            }
                            c[cx * csx + cy * csy + cz * csz] = acc;
                        }
                    }
                }
            }
        }

        // Explicit instantiations for double and single precision.
        template void prolong_2d_host<Real>(const Real *, Real *, Index_t,
                                            Index_t, Index_t, Index_t, Index_t,
                                            Index_t, Index_t, Index_t, Index_t);
        template void prolong_2d_host<Real32>(const Real32 *, Real32 *, Index_t,
                                              Index_t, Index_t, Index_t,
                                              Index_t, Index_t, Index_t,
                                              Index_t, Index_t);
        template void prolong_3d_host<Real>(const Real *, Real *, Index_t,
                                            Index_t, Index_t, Index_t, Index_t,
                                            Index_t, Index_t, Index_t, Index_t,
                                            Index_t, Index_t, Index_t);
        template void prolong_3d_host<Real32>(const Real32 *, Real32 *, Index_t,
                                              Index_t, Index_t, Index_t,
                                              Index_t, Index_t, Index_t,
                                              Index_t, Index_t, Index_t,
                                              Index_t, Index_t);
        template void restrict_2d_host<Real>(const Real *, Real *, Index_t,
                                             Index_t, Index_t, Index_t, Index_t,
                                             Index_t, Index_t, Index_t,
                                             Index_t);
        template void restrict_2d_host<Real32>(const Real32 *, Real32 *,
                                               Index_t, Index_t, Index_t,
                                               Index_t, Index_t, Index_t,
                                               Index_t, Index_t, Index_t);
        template void restrict_3d_host<Real>(const Real *, Real *, Index_t,
                                             Index_t, Index_t, Index_t, Index_t,
                                             Index_t, Index_t, Index_t, Index_t,
                                             Index_t, Index_t, Index_t);
        template void restrict_3d_host<Real32>(const Real32 *, Real32 *,
                                               Index_t, Index_t, Index_t,
                                               Index_t, Index_t, Index_t,
                                               Index_t, Index_t, Index_t,
                                               Index_t, Index_t, Index_t);

    }  // namespace transfer_kernels

}  // namespace muGrid
