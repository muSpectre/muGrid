/**
 * @file   nodal_moments.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   16 Sep 2026
 *
 * @brief  Fused element moments of a nodal scalar field: the integrals
 *         M_k = ∫_e rho(x)^k dx of the FE interpolant over each cell, and
 *         their derivatives with respect to the nodal values.
 *
 * Why moments and not a named energy
 * ----------------------------------
 * The consumer is the phase-field regularisation of a topology optimisation,
 * whose double well W(rho) = rho^2 (1-rho)^2 expands as rho^2 - 2 rho^3 + rho^4.
 * Its cell integral is therefore M_2 - 2 M_3 + M_4 and its nodal gradient the
 * same combination of the moment gradients. Keeping the moments as the
 * interface leaves the *energy* to the caller, exactly as
 * IsotropicStiffnessOperator::compute_sensitivity leaves the material model to
 * the caller: any other polynomial well is a change of coefficients, with no
 * kernel change and nothing material-specific compiled into µGrid.
 *
 * Why it is one fused pass
 * ------------------------
 * Evaluated in array-at-a-time form, this computation materialises the
 * interpolant at every quadrature point of every cell at once -- for the
 * 3x3x3 rule a 27-fold copy of the grid, plus a temporary per term of the
 * polynomial. That is what makes the host implementation cost two orders of
 * magnitude more memory than the solver it regularises. Here every quadrature
 * point is an intermediate consumed in registers and the pass is O(1) in
 * scratch memory.
 *
 * Layout
 * ------
 * On the periodic grid µGrid uses there is exactly one node per cell (the
 * cell's lower-left corner), so nodal and cell fields have the same shape and
 * thread `i` owns both node `i` and cell `i`. `moments` are cell quantities,
 * `moment_gradients` nodal ones; both are written only on the interior
 * (owned) region, and the caller must have communicated the ghosts of `rho`
 * beforehand. Because each thread writes only its own entries, there are no
 * atomics and no scatter/ghost-reduction pass -- the same property that makes
 * the fused stiffness kernel cheap.
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

#ifndef SRC_LIBMUGRID_OPERATORS_NODAL_MOMENTS_HH_
#define SRC_LIBMUGRID_OPERATORS_NODAL_MOMENTS_HH_

#include <array>
#include <string>

#include "collection/field_collection_global.hh"
#include "core/exception.hh"
#include "core/types.hh"
#include "field/field_typed.hh"
#include "operators/fem_element.hh"
#include "operators/solids/isotropic_stiffness.hh"

namespace muGrid {

    //! Moments computed by NodalMomentOperator: k = 2, 3, 4, the powers the
    //! quartic double well needs. k = 0 is the cell volume and k = 1 the
    //! element average, both already available elsewhere.
    constexpr Index_t NB_NODAL_MOMENTS = 3;
    //! Lowest power computed (M_2); moment j corresponds to k = j + FIRST.
    constexpr Index_t FIRST_NODAL_MOMENT = 2;

    namespace nodal_moment_kernels {

        //! Host kernel, 2D. `rho` and the outputs point at the *ghosted*
        //! buffers; (nx, ny) is the interior (owned) extent and the pointers
        //! are pre-offset to its first entry.
        template <typename T, class Element>
        void nodal_moments_2d_host(const T * rho, T * moments,
                                   T * moment_gradients, Index_t nx,
                                   Index_t ny, Index_t rho_stride_x,
                                   Index_t rho_stride_y, Index_t mom_stride_x,
                                   Index_t mom_stride_y, Index_t mom_stride_c,
                                   T cell_volume);

        //! Host kernel, 3D.
        template <typename T, class Element>
        void nodal_moments_3d_host(const T * rho, T * moments,
                                   T * moment_gradients, Index_t nx,
                                   Index_t ny, Index_t nz,
                                   Index_t rho_stride_x, Index_t rho_stride_y,
                                   Index_t rho_stride_z, Index_t mom_stride_x,
                                   Index_t mom_stride_y, Index_t mom_stride_z,
                                   Index_t mom_stride_c, T cell_volume);

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        template <typename T, class Element>
        void nodal_moments_2d_gpu(const T * rho, T * moments,
                                  T * moment_gradients, Index_t nx, Index_t ny,
                                  Index_t rho_stride_x, Index_t rho_stride_y,
                                  Index_t mom_stride_x, Index_t mom_stride_y,
                                  Index_t mom_stride_c, T cell_volume);

        template <typename T, class Element>
        void nodal_moments_3d_gpu(const T * rho, T * moments,
                                  T * moment_gradients, Index_t nx, Index_t ny,
                                  Index_t nz, Index_t rho_stride_x,
                                  Index_t rho_stride_y, Index_t rho_stride_z,
                                  Index_t mom_stride_x, Index_t mom_stride_y,
                                  Index_t mom_stride_z, Index_t mom_stride_c,
                                  T cell_volume);
#endif

    }  // namespace nodal_moment_kernels

    /**
     * @class NodalMomentOperator
     * @brief Cell moments ∫_e rho^k dx (k = 2, 3, 4) of a nodal scalar field
     *        and their nodal gradients, in one fused pass.
     *
     * Q1 elements only: the rule is the 3-point-per-axis tensor Gauss rule of
     * MomentQuadrature, exact for the quartic integrand. The simplex
     * interpolant is only piecewise linear over the cell, so it needs the
     * closed form rather than a cell-level rule (see MomentQuadrature).
     */
    template <Dim_t Dim>
    class NodalMomentOperator {
        static_assert(Dim == 2 || Dim == 3,
                      "NodalMomentOperator is only implemented for 2D and 3D");

    public:
        //! The two element families, selected at construction. Both put one
        //! node on each of the cell's 2^Dim corners; they differ only in the
        //! interpolant, hence in the moment quadrature.
        using Q1 = std::conditional_t<Dim == 2, Q1Quad2D, Q1Hex3D>;
        using P1 = std::conditional_t<Dim == 2, P1Tri2D, P1Tet3D>;
        static constexpr Index_t NB_NODES = MomentQuadrature<Q1>::NbNodes;

        /**
         * @brief Construct from the grid spacing.
         *
         * Only the cell volume enters: the moments integrate a function of the
         * interpolant, which carries no derivative and hence no 1/h factors.
         */
        explicit NodalMomentOperator(
            const std::array<Real, Dim> & grid_spacing,
            FEMElementKind element = FEMElementKind::Q1)
            : element{element} {
            Real vol{1.0};
            for (Dim_t d = 0; d < Dim; ++d) {
                if (not(grid_spacing[d] > 0.0)) {
                    throw RuntimeError{operator_name() +
                                       ": grid spacing must be positive"};
                }
                vol *= grid_spacing[d];
            }
            this->cell_volume = vol;
        }

        NodalMomentOperator() = delete;
        NodalMomentOperator(const NodalMomentOperator &) = delete;
        NodalMomentOperator(NodalMomentOperator &&) = default;
        ~NodalMomentOperator() = default;
        NodalMomentOperator & operator=(const NodalMomentOperator &) = delete;
        NodalMomentOperator & operator=(NodalMomentOperator &&) = default;

        static std::string operator_name() {
            return "NodalMomentOperator" + std::to_string(Dim) + "D";
        }

        //! Cell volume h_x h_y [h_z].
        Real get_cell_volume() const { return this->cell_volume; }

        //! Element family this operator integrates.
        FEMElementKind get_element() const { return this->element; }

        //! Number of quadrature points per cell, for the chosen element.
        Index_t get_nb_quad() const {
            return this->element == FEMElementKind::Q1
                       ? MomentQuadrature<Q1>::NbQuad
                       : MomentQuadrature<P1>::NbQuad;
        }

        /**
         * @brief Compute the moments and their nodal gradients.
         *
         * @param rho               nodal scalar field, ghosts already
         *                          communicated
         * @param moments           cell field with NB_NODAL_MOMENTS
         *                          components; component j receives
         *                          ∫_e rho^(j+2) dx
         * @param moment_gradients  nodal field with NB_NODAL_MOMENTS
         *                          components; component j receives
         *                          Σ_e ∂(∫_e rho^(j+2) dx)/∂rho_n
         */
        void compute(const TypedFieldBase<Real> & rho,
                     TypedFieldBase<Real> & moments,
                     TypedFieldBase<Real> & moment_gradients) const {
            this->compute_impl<Real>(rho, moments, moment_gradients);
        }

        //! Single-precision (Real32) overload.
        void compute(const TypedFieldBase<Real32> & rho,
                     TypedFieldBase<Real32> & moments,
                     TypedFieldBase<Real32> & moment_gradients) const {
            this->compute_impl<Real32>(rho, moments, moment_gradients);
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        //! Device overloads.
        void compute(const TypedFieldBase<Real, DefaultDeviceSpace> & rho,
                     TypedFieldBase<Real, DefaultDeviceSpace> & moments,
                     TypedFieldBase<Real, DefaultDeviceSpace> & grads) const {
            this->compute_impl<Real>(rho, moments, grads);
        }
        void compute(const TypedFieldBase<Real32, DefaultDeviceSpace> & rho,
                     TypedFieldBase<Real32, DefaultDeviceSpace> & moments,
                     TypedFieldBase<Real32, DefaultDeviceSpace> & grads) const {
            this->compute_impl<Real32>(rho, moments, grads);
        }
#endif

    protected:
        /**
         * @brief Validate the three fields and return the interior extent.
         *
         * The kernel reads, for its own node, the cells at offset -1 and their
         * nodes at offset +1, so one ghost layer on each side of every axis is
         * required -- the same stencil as the fused stiffness operator.
         */
        template <typename T, typename Space>
        std::array<Index_t, Dim>
        validate(const TypedFieldBase<T, Space> & rho,
                 const TypedFieldBase<T, Space> & moments,
                 const TypedFieldBase<T, Space> & grads,
                 std::array<Index_t, Dim> & nb_with_ghosts) const {
            const std::string op{operator_name()};
            auto * fc = dynamic_cast<const GlobalFieldCollection *>(
                &rho.get_collection());
            if (!fc) {
                throw RuntimeError{op + " requires GlobalFieldCollection"};
            }
            if (&moments.get_collection() != &rho.get_collection() ||
                &grads.get_collection() != &rho.get_collection()) {
                throw RuntimeError{
                    op + ": rho, moments and moment_gradients must live on the "
                         "same field collection"};
            }
            const auto left = fc->get_nb_ghosts_left();
            const auto right = fc->get_nb_ghosts_right();
            const auto with_ghosts = fc->get_nb_subdomain_grid_pts_with_ghosts();
            std::array<Index_t, Dim> interior{};
            for (Dim_t d = 0; d < Dim; ++d) {
                if (left[d] < 1 || right[d] < 1) {
                    throw RuntimeError{
                        op + " requires at least 1 ghost cell on each side of "
                             "every axis"};
                }
                nb_with_ghosts[d] = with_ghosts[d];
                interior[d] = with_ghosts[d] - 2;
            }
            auto check_components = [&op](const TypedFieldBase<T, Space> & f,
                                          const std::string & what) {
                if (f.get_nb_components() != NB_NODAL_MOMENTS) {
                    throw RuntimeError{
                        op + ": " + what + " must have " +
                        std::to_string(NB_NODAL_MOMENTS) + " components, got " +
                        std::to_string(f.get_nb_components())};
                }
            };
            if (rho.get_nb_components() != 1) {
                throw RuntimeError{op +
                                   ": rho must be a scalar (1-component) field"};
            }
            check_components(moments, "moments");
            check_components(grads, "moment_gradients");
            return interior;
        }

        /**
         * @brief Spatial strides, component stride and interior offset of a
         *        field with @p nb_components components.
         *
         * The two memory spaces do *not* share a layout, and nothing in the
         * field API makes that visible at the call site:
         *
         *  - host is array-of-structures — component stride 1, spatial strides
         *    scaled by the component count;
         *  - device is structure-of-arrays — spatial stride 1, one component a
         *    whole (ghosted) grid apart, which is what makes a warp's access
         *    to one component contiguous.
         *
         * A scalar field is identical under both, which is why a kernel can
         * look correct on a 1-component input and scatter its multi-component
         * output. @p aos selects the layout.
         */
        struct Strides {
            std::array<Index_t, Dim> spatial;
            Index_t component;
            Index_t offset;  //!< buffer origin to the first interior entry
        };

        static Strides strides_of(
            const std::array<Index_t, Dim> & nb_with_ghosts,
            Index_t nb_components, bool aos) {
            Strides s{};
            Index_t nb_pts{1};
            for (Dim_t d = 0; d < Dim; ++d) { nb_pts *= nb_with_ghosts[d]; }
            Index_t stride{aos ? nb_components : Index_t{1}};
            s.component = aos ? Index_t{1} : nb_pts;
            s.offset = 0;
            for (Dim_t d = 0; d < Dim; ++d) {
                s.spatial[d] = stride;
                // one ghost on the low side of every axis
                s.offset += stride;
                stride *= nb_with_ghosts[d];
            }
            return s;
        }

        template <typename T>
        void compute_impl(const TypedFieldBase<T> & rho,
                          TypedFieldBase<T> & moments,
                          TypedFieldBase<T> & grads) const {
            std::array<Index_t, Dim> ng{};
            const auto n = this->validate(rho, moments, grads, ng);
            const auto rs = strides_of(ng, 1, true);
            const auto ms = strides_of(ng, NB_NODAL_MOMENTS, true);
            const T * r = rho.data() + rs.offset;
            T * m = moments.data() + ms.offset;
            T * g = grads.data() + ms.offset;
            const T vol{static_cast<T>(this->cell_volume)};
            const bool q1{this->element == FEMElementKind::Q1};
            if constexpr (Dim == 2) {
                auto call = [&](auto tag) {
                    nodal_moment_kernels::nodal_moments_2d_host<
                        T, typename decltype(tag)::type>(
                        r, m, g, n[0], n[1], rs.spatial[0], rs.spatial[1],
                        ms.spatial[0], ms.spatial[1], ms.component, vol);
                };
                if (q1) { call(Tag<Q1>{}); } else { call(Tag<P1>{}); }
            } else {
                auto call = [&](auto tag) {
                    nodal_moment_kernels::nodal_moments_3d_host<
                        T, typename decltype(tag)::type>(
                        r, m, g, n[0], n[1], n[2], rs.spatial[0],
                        rs.spatial[1], rs.spatial[2], ms.spatial[0],
                        ms.spatial[1], ms.spatial[2], ms.component, vol);
                };
                if (q1) { call(Tag<Q1>{}); } else { call(Tag<P1>{}); }
            }
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        template <typename T>
        void compute_impl(const TypedFieldBase<T, DefaultDeviceSpace> & rho,
                          TypedFieldBase<T, DefaultDeviceSpace> & moments,
                          TypedFieldBase<T, DefaultDeviceSpace> & grads) const {
            std::array<Index_t, Dim> ng{};
            const auto n = this->validate(rho, moments, grads, ng);
            // Device buffers are structure-of-arrays; see strides_of().
            const auto rs = strides_of(ng, 1, false);
            const auto ms = strides_of(ng, NB_NODAL_MOMENTS, false);
            const T * r = rho.view().data() + rs.offset;
            T * m = moments.view().data() + ms.offset;
            T * g = grads.view().data() + ms.offset;
            const T vol{static_cast<T>(this->cell_volume)};
            const bool q1{this->element == FEMElementKind::Q1};
            if constexpr (Dim == 2) {
                auto call = [&](auto tag) {
                    nodal_moment_kernels::nodal_moments_2d_gpu<
                        T, typename decltype(tag)::type>(
                        r, m, g, n[0], n[1], rs.spatial[0], rs.spatial[1],
                        ms.spatial[0], ms.spatial[1], ms.component, vol);
                };
                if (q1) { call(Tag<Q1>{}); } else { call(Tag<P1>{}); }
            } else {
                auto call = [&](auto tag) {
                    nodal_moment_kernels::nodal_moments_3d_gpu<
                        T, typename decltype(tag)::type>(
                        r, m, g, n[0], n[1], n[2], rs.spatial[0],
                        rs.spatial[1], rs.spatial[2], ms.spatial[0],
                        ms.spatial[1], ms.spatial[2], ms.component, vol);
                };
                if (q1) { call(Tag<Q1>{}); } else { call(Tag<P1>{}); }
            }
        }
#endif

        //! Carries an element type into a generic lambda.
        template <class E>
        struct Tag { using type = E; };

        Real cell_volume;
        FEMElementKind element;
    };

    //! 2D alias.
    using NodalMomentOperator2D = NodalMomentOperator<2>;
    //! 3D alias.
    using NodalMomentOperator3D = NodalMomentOperator<3>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_NODAL_MOMENTS_HH_
