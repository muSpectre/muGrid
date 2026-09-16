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


    /* ----------------------------------------------------------------------
     * Moment quadrature
     *
     * The gradient/stiffness kernels integrate products of shape-function
     * *gradients*, for which the tables above (2^Dim-point Gauss on Q1,
     * one point per simplex on P1) are exact. Integrating a nonlinear
     * pointwise function of the interpolant itself -- e.g. the phase-field
     * double well W(rho) = rho^2 (1-rho)^2, quartic -- needs two things those
     * tables do not carry: the shape-function *values* N[q][n], and a rule of
     * high enough order. A trilinear interpolant raised to the fourth power is
     * degree 4 per axis, so the 2-point rule (exact to degree 3) is not enough
     * and a 3-point rule per axis (exact to degree 5) is.
     *
     * These live in their own tables rather than replacing Wfrac/B: the
     * stiffness path must keep the quadrature it was built and validated with.
     * ------------------------------------------------------------------- */

    //! 3-point Gauss-Legendre rule on the unit interval [0, 1], exact to
    //! polynomial degree 5. Positions 1/2 ∓ sqrt(3/5)/2, weights 5/18, 8/18.
    struct Gauss3Unit {
        static constexpr Index_t NbPts = 3;
        //! sqrt(3/5) / 2 = 0.3872983346207416885179265...
        static constexpr Real Xi[NbPts] = {
            0.5 - 0.38729833462074168851, 0.5,
            0.5 + 0.38729833462074168851};
        static constexpr Real W[NbPts] = {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0};
    };

    /**
     * @struct MomentQuadrature
     * @brief Shape-function *values* and weights of a rule exact for the
     *        fourth power of the element's interpolant.
     *
     * Specialised per element type. `N[q][n]` is the value of node n's shape
     * function at point q and `Wfrac[q]` its weight as a fraction of the cell
     * volume (Σ_q Wfrac = 1), matching the convention of the gradient tables.
     * The interpolant at q is then `rho_q = Σ_n N[q][n] rho_n` and the integral
     * of any f(rho) over the cell is `h_x h_y [h_z] Σ_q Wfrac[q] f(rho_q)`.
     *
     * Defined for the Q1 elements only. The simplex (P1) interpolant is
     * piecewise linear across the sub-simplices, so a cell-level smooth rule is
     * *not* exact for it; its moments have a closed form instead (the integral
     * of a linear function's k-th power over a simplex is a complete
     * homogeneous symmetric polynomial in the corner values), which is where a
     * P1 specialisation should go if one is needed.
     */
    template <class Element>
    struct MomentQuadrature;

    //! Q1 bilinear quad: 3x3 tensor Gauss (9 points).
    template <>
    struct MomentQuadrature<Q1Quad2D> {
        static constexpr Dim_t SpatialDim = 2;
        static constexpr Index_t NbNodes = 4;
        static constexpr Index_t NbQuad = 9;

        //! Tensor-product index: q = qx + 3 qy, matching the binary node order
        //! (x fastest) used everywhere else.
        static constexpr Real shape(Index_t q, Index_t n) {
            Real v{1.0};
            Index_t r{q};
            for (Dim_t d = 0; d < SpatialDim; ++d) {
                const Real xi{Gauss3Unit::Xi[r % Gauss3Unit::NbPts]};
                v *= fem_node_offset(n, d) ? xi : (1.0 - xi);
                r /= Gauss3Unit::NbPts;
            }
            return v;
        }
        static constexpr Real weight(Index_t q) {
            Real w{1.0};
            Index_t r{q};
            for (Dim_t d = 0; d < SpatialDim; ++d) {
                w *= Gauss3Unit::W[r % Gauss3Unit::NbPts];
                r /= Gauss3Unit::NbPts;
            }
            return w;
        }
    };

    //! Q1 trilinear hexahedron: 3x3x3 tensor Gauss (27 points).
    template <>
    struct MomentQuadrature<Q1Hex3D> {
        static constexpr Dim_t SpatialDim = 3;
        static constexpr Index_t NbNodes = 8;
        static constexpr Index_t NbQuad = 27;

        static constexpr Real shape(Index_t q, Index_t n) {
            Real v{1.0};
            Index_t r{q};
            for (Dim_t d = 0; d < SpatialDim; ++d) {
                const Real xi{Gauss3Unit::Xi[r % Gauss3Unit::NbPts]};
                v *= fem_node_offset(n, d) ? xi : (1.0 - xi);
                r /= Gauss3Unit::NbPts;
            }
            return v;
        }
        static constexpr Real weight(Index_t q) {
            Real w{1.0};
            Index_t r{q};
            for (Dim_t d = 0; d < SpatialDim; ++d) {
                w *= Gauss3Unit::W[r % Gauss3Unit::NbPts];
                r /= Gauss3Unit::NbPts;
            }
            return w;
        }
    };

    //! Reference-simplex moment rule shared by all 2 sub-simplices of P1Tri2D:
    //! 9 Gauss-Jacobi points, barycentric coordinates.
    struct P1Tri2DMomentRef {
        static constexpr Index_t NbPts = 9;
        static constexpr Index_t NbCorners = 3;
        static constexpr Index_t NbSimplices = 2;
        //! Barycentric coordinates of each point.
        static constexpr Real Lambda[NbPts][NbCorners] = {
            {0.80869438567766982, 0.088587959512703873, 0.10271765480962626},
            {0.45570602024364804, 0.088587959512703873, 0.45570602024364804},
            {0.10271765480962625, 0.088587959512703873, 0.80869438567766982},
            {0.52397906772010083, 0.40946686444073471, 0.066554067839164496},
            {0.29526656777963267, 0.40946686444073471, 0.29526656777963267},
            {0.06655406783916451, 0.40946686444073471, 0.52397906772010083},
            {0.18840940595207223, 0.78765946176084722, 0.023931132287080596},
            {0.10617026911957639, 0.78765946176084722, 0.10617026911957639},
            {0.023931132287080548, 0.78765946176084722, 0.18840940595207217},
        };
        //! Weight of each point, as a fraction of the simplex.
        static constexpr Real W[NbPts] = {
            0.1116288409660886, 0.17860614554574167, 0.1116288409660886,
            0.12735617019977019, 0.2037698723196322, 0.12735617019977019,
            0.038792766611919008, 0.062068426579070385, 0.038792766611919008,
        };
        //! Cell-node index of each sub-simplex's corners.
        static constexpr Index_t Nodes[NbSimplices][NbCorners] = {
            {0, 1, 2},
            {1, 2, 3},
        };
        //! Volume of each sub-simplex as a fraction of the cell.
        static constexpr Real Frac[NbSimplices] = {
            0.5, 0.5};
    };

    //! Reference-simplex moment rule shared by all 5 sub-simplices of P1Tet3D:
    //! 27 Gauss-Jacobi points, barycentric coordinates.
    struct P1Tet3DMomentRef {
        static constexpr Index_t NbPts = 27;
        static constexpr Index_t NbCorners = 4;
        static constexpr Index_t NbSimplices = 5;
        //! Barycentric coordinates of each point.
        static constexpr Real Lambda[NbPts][NbCorners] = {
            {0.74966452822169316, 0.072994024073149588, 0.082121567863442366, 0.095219879841714927},
            {0.42244220403170396, 0.072994024073149588, 0.082121567863442366, 0.42244220403170402},
            {0.095219879841714983, 0.072994024073149588, 0.082121567863442366, 0.74966452822169305},
            {0.48573172703711331, 0.072994024073149588, 0.37957823028059062, 0.061696018609146495},
            {0.27371387282312987, 0.072994024073149588, 0.37957823028059062, 0.27371387282312992},
            {0.061696018609146419, 0.072994024073149588, 0.37957823028059062, 0.48573172703711331},
            {0.17465664523839874, 0.072994024073149588, 0.73016502804763195, 0.022184302640819709},
            {0.098420473939609177, 0.072994024073149588, 0.73016502804763195, 0.09842047393960926},
            {0.02218430264081972, 0.072994024073149588, 0.73016502804763195, 0.1746566452383988},
            {0.5280743882734471, 0.34700376603835187, 0.057847603936142598, 0.067074241752058519},
            {0.29757431501275278, 0.34700376603835187, 0.057847603936142598, 0.29757431501275278},
            {0.067074241752058561, 0.34700376603835187, 0.057847603936142598, 0.52807438827344699},
            {0.34215635789596122, 0.34700376603835187, 0.26738032041188448, 0.043459555653802467},
            {0.19280795677488183, 0.34700376603835187, 0.26738032041188448, 0.19280795677488186},
            {0.04345955565380244, 0.34700376603835187, 0.26738032041188448, 0.34215635789596127},
            {0.1230306325296544, 0.34700376603835187, 0.51433866217409208, 0.015626939257901633},
            {0.069328785893777889, 0.34700376603835187, 0.51433866217409208, 0.069328785893778055},
            {0.015626939257901484, 0.34700376603835187, 0.51433866217409208, 0.12303063252965447},
            {0.23856305665049093, 0.70500220988849849, 0.026133252286734812, 0.030301481174275793},
            {0.13443226891238336, 0.70500220988849849, 0.026133252286734812, 0.13443226891238333},
            {0.030301481174275779, 0.70500220988849849, 0.026133252286734812, 0.23856305665049091},
            {0.15457266704211459, 0.70500220988849849, 0.12079182013390249, 0.019633302935484483},
            {0.087102984988799537, 0.70500220988849849, 0.12079182013390249, 0.087102984988799509},
            {0.019633302935484487, 0.70500220988849849, 0.12079182013390249, 0.15457266704211456},
            {0.055580358392082085, 0.70500220988849849, 0.23235780057986466, 0.0070596311395547794},
            {0.031319994765818482, 0.70500220988849849, 0.23235780057986466, 0.031319994765818426},
            {0.0070596311395547673, 0.70500220988849849, 0.23235780057986466, 0.055580358392082072},
        };
        //! Weight of each point, as a fraction of the simplex.
        static constexpr Real W[NbPts] = {
            0.052622849577906312, 0.084196559324650061, 0.052622849577906312,
            0.060036855433056716, 0.096058968692890698, 0.060036855433056716,
            0.01828726254310915, 0.029259620068974621, 0.01828726254310915,
            0.048975904599280064, 0.078361447358848063, 0.048975904599280064,
            0.055876094276822739, 0.089401750842916336, 0.055876094276822739,
            0.017019892173785543, 0.027231827478056855, 0.017019892173785543,
            0.010030086788902218, 0.016048138862243544, 0.010030086788902218,
            0.011443220489890717, 0.018309152783825139, 0.011443220489890717,
            0.0034856118950243142, 0.0055769790320389003, 0.0034856118950243142,
        };
        //! Cell-node index of each sub-simplex's corners.
        static constexpr Index_t Nodes[NbSimplices][NbCorners] = {
            {1, 2, 4, 7},
            {0, 1, 2, 4},
            {1, 2, 3, 7},
            {1, 4, 5, 7},
            {2, 4, 6, 7},
        };
        //! Volume of each sub-simplex as a fraction of the cell.
        static constexpr Real Frac[NbSimplices] = {
            0.33333333333333331, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666};
    };

    /**
     * @brief Moment quadrature on a P1 cell: the reference rule above, placed
     *        on each sub-simplex.
     *
     * The simplex interpolant is only *piecewise* linear over the cell, so a
     * cell-level smooth rule is not exact for it -- the rule has to live on
     * the sub-simplices. Placing it there leaves the interface identical to
     * the Q1 case: shape(q, n) is still the value of cell-node n's shape
     * function at point q, zero for the nodes of the other sub-simplices, and
     * the kernels need no knowledge of which element they are running on.
     *
     * The points are Gauss-Jacobi rather than Gauss-Legendre. The collapsed
     * (Duffy) map from the cube carries a Jacobian -- (1-u) in 2D,
     * (1-u)^2 (1-v) in 3D -- and folding it into the *weight function* keeps
     * 3 points per axis exact to degree 5 for the quartic integrand. Leaving
     * it in the integrand instead inflates the degree by up to 2, and a
     * 3-point Gauss-Legendre rule is then exact for M2 and M3 but silently
     * wrong for M4. Every weight is positive, as a Gauss rule for a positive
     * weight function must be, so a cell's double-well energy can never come
     * out negative.
     */
    template <class Element, class Ref>
    struct P1MomentQuadrature {
        static constexpr Dim_t SpatialDim = Element::SpatialDim;
        static constexpr Index_t NbNodes = Element::NbNodes;
        static constexpr Index_t NbQuad = Ref::NbPts * Ref::NbSimplices;

        //! Point q is point (q % NbPts) of sub-simplex (q / NbPts).
        static constexpr Real shape(Index_t q, Index_t n) {
            const Index_t s{q / Ref::NbPts};
            const Index_t r{q % Ref::NbPts};
            for (Index_t i = 0; i < Ref::NbCorners; ++i) {
                if (Ref::Nodes[s][i] == n) { return Ref::Lambda[r][i]; }
            }
            return 0.0;  // n is not a corner of this sub-simplex
        }
        static constexpr Real weight(Index_t q) {
            return Ref::W[q % Ref::NbPts] * Ref::Frac[q / Ref::NbPts];
        }
    };

    //! P1 triangles: the reference rule on each of the 2 sub-triangles.
    template <>
    struct MomentQuadrature<P1Tri2D>
        : P1MomentQuadrature<P1Tri2D, P1Tri2DMomentRef> {};

    //! P1 tetrahedra: the reference rule on each of the 5 sub-tetrahedra.
    template <>
    struct MomentQuadrature<P1Tet3D>
        : P1MomentQuadrature<P1Tet3D, P1Tet3DMomentRef> {};

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_FEM_ELEMENT_HH_
