/**
 * @file   fem_element.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   30 Jun 2026
 *
 * @brief  Finite-element traits: the single source of truth for the reference
 *         shape-function gradients, quadrature weights and node layout of each
 *         supported element, on the regular grid used throughout µGrid.
 *
 * The FE operators (gradient/divergence and the fused isotropic stiffness) on a
 * regular grid all reduce to the same computation: a per-pixel contraction of
 * the nodal values with a reference shape-function-gradient table B, scaled by
 * the (diagonal, constant) inverse grid spacing 1/h_d, summed over quadrature
 * points with weights w_q. The element type only changes the numbers in that
 * table — not the kernel structure — so all element-specific data lives here
 * and the kernels are templated on the element. This covers linear simplices
 * (the historical 2-triangle / 5-tetrahedron decompositions) and, by adding a
 * traits struct, Q1 (bilinear quad / trilinear hex) elements, with no new
 * kernel code.
 *
 * Conventions
 * -----------
 * - Nodes are the 2^Dim corners of a pixel/voxel, indexed by binary offsets:
 *   node n sits at offset ((n>>0)&1, (n>>1)&1, ...), i.e. x varies fastest.
 * - `B[q][d][n]` is the reference shape-function gradient of node n in
 *   direction d at quadrature point q; the *physical* gradient is
 *   `B[q][d][n] / h_d`. (Canonical layout [quad][dim][node], shared with the
 *   stiffness operator.)
 * - `Wfrac[q]` is the quadrature weight as a *fraction* of the cell volume
 *   (Σ_q Wfrac = 1); the physical weight is `Wfrac[q] * (h_x h_y [h_z])`.
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

#ifndef SRC_LIBMUGRID_OPERATORS_FEM_ELEMENT_HH_
#define SRC_LIBMUGRID_OPERATORS_FEM_ELEMENT_HH_

#include "core/types.hh"

namespace muGrid {

    //! Binary corner offset (0 or 1) of node `n` along axis `d`: x fastest.
    constexpr Index_t fem_node_offset(Index_t n, Index_t d) {
        return (n >> d) & Index_t{1};
    }

    /**
     * @struct P1Tri2D
     * @brief 2D pixel split into 2 linear triangles (the historical element).
     *
     * Quadrature point q is the (constant-gradient) interior of triangle q;
     * each weight is half the pixel area.
     */
    struct P1Tri2D {
        static constexpr Dim_t SpatialDim = 2;
        static constexpr Index_t NbNodes = 4;
        static constexpr Index_t NbQuad = 2;
        //! B[q][d][n] — entries are ±1 (constant-gradient simplices).
        static constexpr Real B[NbQuad][SpatialDim][NbNodes] = {
            // Triangle 0 (lower-left): nodes 0,1,2
            {{-1.0, 1.0, 0.0, 0.0},   // d/dx
             {-1.0, 0.0, 1.0, 0.0}},  // d/dy
            // Triangle 1 (upper-right): nodes 1,2,3
            {{0.0, 0.0, -1.0, 1.0},   // d/dx
             {0.0, -1.0, 0.0, 1.0}},  // d/dy
        };
        static constexpr Real Wfrac[NbQuad] = {0.5, 0.5};
    };

    /**
     * @struct P1Tet3D
     * @brief 3D voxel split into 5 linear tetrahedra (Kuhn triangulation).
     *
     * Quadrature point 0 is the central tetrahedron (volume fraction 1/3); the
     * four corner tetrahedra each have volume fraction 1/6.
     */
    struct P1Tet3D {
        static constexpr Dim_t SpatialDim = 3;
        static constexpr Index_t NbNodes = 8;
        static constexpr Index_t NbQuad = 5;
        //! B[q][d][n]; node order is the binary corner indexing (x fastest).
        static constexpr Real B[NbQuad][SpatialDim][NbNodes] = {
            // q0: central tetrahedron (nodes 1,2,4,7)
            {{0.0, 0.5, -0.5, 0.0, -0.5, 0.0, 0.0, 0.5},
             {0.0, -0.5, 0.5, 0.0, -0.5, 0.0, 0.0, 0.5},
             {0.0, -0.5, -0.5, 0.0, 0.5, 0.0, 0.0, 0.5}},
            // q1: corner at (0,0,0) — nodes 0,1,2,4
            {{-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
             {-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
             {-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}},
            // q2: corner at (1,1,0) — nodes 1,2,3,7
            {{0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
             {0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
             {0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0}},
            // q3: corner at (1,0,1) — nodes 1,4,5,7
            {{0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0},
             {0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0},
             {0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}},
            // q4: corner at (0,1,1) — nodes 2,4,6,7
            {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0},
             {0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0},
             {0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0}},
        };
        static constexpr Real Wfrac[NbQuad] = {1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0,
                                               1.0 / 6.0, 1.0 / 6.0};
    };

    /**
     * @struct Q1Quad2D
     * @brief 2D bilinear quadrilateral (one Q1 element per pixel) with 2×2
     *        Gauss quadrature (4 points). Unlike the simplex, the shape-function
     *        gradients vary within the element, so B has no structural zeros.
     */
    struct Q1Quad2D {
        static constexpr Dim_t SpatialDim = 2;
        static constexpr Index_t NbNodes = 4;
        static constexpr Index_t NbQuad = 4;
        // B[q][d][n] = bilinear ∂N_n/∂ξ_d at the 2×2 Gauss points of the unit
        // square (physical gradient = B/h_d). Node order is the binary corner
        // index; Gauss point q = (q&1, q>>1) over {0.5∓1/(2√3)}.
        static constexpr Real B[NbQuad][SpatialDim][NbNodes] = {
            {{-0.78867513459481287, 0.78867513459481287, -0.21132486540518708,
              0.21132486540518708},
             {-0.78867513459481287, -0.21132486540518708, 0.78867513459481287,
              0.21132486540518708}},
            {{-0.78867513459481287, 0.78867513459481287, -0.21132486540518708,
              0.21132486540518708},
             {-0.21132486540518713, -0.78867513459481287, 0.21132486540518713,
              0.78867513459481287}},
            {{-0.21132486540518713, 0.21132486540518713, -0.78867513459481287,
              0.78867513459481287},
             {-0.78867513459481287, -0.21132486540518708, 0.78867513459481287,
              0.21132486540518708}},
            {{-0.21132486540518713, 0.21132486540518713, -0.78867513459481287,
              0.78867513459481287},
             {-0.21132486540518713, -0.78867513459481287, 0.21132486540518713,
              0.78867513459481287}},
        };
        static constexpr Real Wfrac[NbQuad] = {0.25, 0.25, 0.25, 0.25};
    };

    /**
     * @struct Q1Hex3D
     * @brief 3D trilinear hexahedron (one Q1 element per voxel) with 2×2×2
     *        Gauss quadrature (8 points).
     */
    struct Q1Hex3D {
        static constexpr Dim_t SpatialDim = 3;
        static constexpr Index_t NbNodes = 8;
        static constexpr Index_t NbQuad = 8;
        static constexpr Real B[NbQuad][SpatialDim][NbNodes] = {
            {{-0.62200846792814624, 0.62200846792814624, -0.16666666666666663,
              0.16666666666666663, -0.16666666666666663, 0.16666666666666663,
              -0.044658198738520435, 0.044658198738520435},
             {-0.62200846792814624, -0.16666666666666663, 0.62200846792814624,
              0.16666666666666663, -0.16666666666666663, -0.044658198738520435,
              0.16666666666666663, 0.044658198738520435},
             {-0.62200846792814624, -0.16666666666666663, -0.16666666666666663,
              -0.044658198738520435, 0.62200846792814624, 0.16666666666666663,
              0.16666666666666663, 0.044658198738520435}},
            {{-0.62200846792814624, 0.62200846792814624, -0.16666666666666663,
              0.16666666666666663, -0.16666666666666663, 0.16666666666666663,
              -0.044658198738520435, 0.044658198738520435},
             {-0.16666666666666669, -0.62200846792814624, 0.16666666666666669,
              0.62200846792814624, -0.044658198738520449, -0.16666666666666663,
              0.044658198738520449, 0.16666666666666663},
             {-0.16666666666666669, -0.62200846792814624, -0.044658198738520449,
              -0.16666666666666663, 0.16666666666666669, 0.62200846792814624,
              0.044658198738520449, 0.16666666666666663}},
            {{-0.16666666666666669, 0.16666666666666669, -0.62200846792814624,
              0.62200846792814624, -0.044658198738520449, 0.044658198738520449,
              -0.16666666666666663, 0.16666666666666663},
             {-0.62200846792814624, -0.16666666666666663, 0.62200846792814624,
              0.16666666666666663, -0.16666666666666663, -0.044658198738520435,
              0.16666666666666663, 0.044658198738520435},
             {-0.16666666666666669, -0.044658198738520449, -0.62200846792814624,
              -0.16666666666666663, 0.16666666666666669, 0.044658198738520449,
              0.62200846792814624, 0.16666666666666663}},
            {{-0.16666666666666669, 0.16666666666666669, -0.62200846792814624,
              0.62200846792814624, -0.044658198738520449, 0.044658198738520449,
              -0.16666666666666663, 0.16666666666666663},
             {-0.16666666666666669, -0.62200846792814624, 0.16666666666666669,
              0.62200846792814624, -0.044658198738520449, -0.16666666666666663,
              0.044658198738520449, 0.16666666666666663},
             {-0.044658198738520456, -0.16666666666666669, -0.16666666666666669,
              -0.62200846792814624, 0.044658198738520456, 0.16666666666666669,
              0.16666666666666669, 0.62200846792814624}},
            {{-0.16666666666666669, 0.16666666666666669, -0.044658198738520449,
              0.044658198738520449, -0.62200846792814624, 0.62200846792814624,
              -0.16666666666666663, 0.16666666666666663},
             {-0.16666666666666669, -0.044658198738520449, 0.16666666666666669,
              0.044658198738520449, -0.62200846792814624, -0.16666666666666663,
              0.62200846792814624, 0.16666666666666663},
             {-0.62200846792814624, -0.16666666666666663, -0.16666666666666663,
              -0.044658198738520435, 0.62200846792814624, 0.16666666666666663,
              0.16666666666666663, 0.044658198738520435}},
            {{-0.16666666666666669, 0.16666666666666669, -0.044658198738520449,
              0.044658198738520449, -0.62200846792814624, 0.62200846792814624,
              -0.16666666666666663, 0.16666666666666663},
             {-0.044658198738520456, -0.16666666666666669, 0.044658198738520456,
              0.16666666666666669, -0.16666666666666669, -0.62200846792814624,
              0.16666666666666669, 0.62200846792814624},
             {-0.16666666666666669, -0.62200846792814624, -0.044658198738520449,
              -0.16666666666666663, 0.16666666666666669, 0.62200846792814624,
              0.044658198738520449, 0.16666666666666663}},
            {{-0.044658198738520456, 0.044658198738520456, -0.16666666666666669,
              0.16666666666666669, -0.16666666666666669, 0.16666666666666669,
              -0.62200846792814624, 0.62200846792814624},
             {-0.16666666666666669, -0.044658198738520449, 0.16666666666666669,
              0.044658198738520449, -0.62200846792814624, -0.16666666666666663,
              0.62200846792814624, 0.16666666666666663},
             {-0.16666666666666669, -0.044658198738520449, -0.62200846792814624,
              -0.16666666666666663, 0.16666666666666669, 0.044658198738520449,
              0.62200846792814624, 0.16666666666666663}},
            {{-0.044658198738520456, 0.044658198738520456, -0.16666666666666669,
              0.16666666666666669, -0.16666666666666669, 0.16666666666666669,
              -0.62200846792814624, 0.62200846792814624},
             {-0.044658198738520456, -0.16666666666666669, 0.044658198738520456,
              0.16666666666666669, -0.16666666666666669, -0.62200846792814624,
              0.16666666666666669, 0.62200846792814624},
             {-0.044658198738520456, -0.16666666666666669, -0.16666666666666669,
              -0.62200846792814624, 0.044658198738520456, 0.16666666666666669,
              0.16666666666666669, 0.62200846792814624}},
        };
        static constexpr Real Wfrac[NbQuad] = {0.125, 0.125, 0.125, 0.125,
                                               0.125, 0.125, 0.125, 0.125};
    };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_FEM_ELEMENT_HH_
