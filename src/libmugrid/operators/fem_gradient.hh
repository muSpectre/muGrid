/**
 * @file   fem_gradient.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   26 Jun 2026
 *
 * @brief  Dimension-templated hard-coded linear FEM gradient operator (2D/3D)
 *
 * The 2D (linear triangles, 4 nodes / 2 quad points) and 3D (linear
 * tetrahedra, 8 nodes / 5 quad points) FEM gradient operators shared the
 * entire apply/transpose interface, the device dispatch and the field
 * validation, differing only in the dimension, the element counts and the
 * shape-function tables. They are unified here into a single
 * `template <Dim_t Dim> FEMGradientOperator`, with the dimension-varying facts
 * in `FEMGradientTraits<Dim>` and the divergent stride/kernel selection behind
 * `if constexpr`. The dimension-specialised stencil kernels keep their distinct
 * signatures (defined in fem_gradient_2d.cc / fem_gradient_3d.cc for the host
 * and fem_gradient_gpu.cc for the device). The historical class names remain as
 * type aliases (FEMGradientOperator2D / FEMGradientOperator3D) so no binding or
 * downstream code changes.
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

#ifndef SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_HH_
#define SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_HH_

#include "core/types.hh"
#include "field/field_typed.hh"
#include "collection/field_collection_global.hh"
#include "memory/memory_space.hh"
#include "operators/linear.hh"
#include "operators/fem_element.hh"

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
#include "memory/gpu_runtime.hh"
#include "memory/device_alloc.hh"
#endif

#include <array>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace muGrid {

    // Kernel implementations and reference tables. The host kernels are defined
    // in fem_gradient_2d.cc / fem_gradient_3d.cc, the device kernels in
    // fem_gradient_gpu.cc; both translation units include this header for the
    // declarations and the shape-function tables below.
    namespace fem_gradient_kernels {

        // Element data (B, weights, node layout) now lives in fem_element.hh,
        // and the host gradient/divergence kernels are the element-generic
        // templates defined further below; the old per-dimension host kernel
        // declarations and shape-function tables have been removed.

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        // Element-generic device launch wrappers (defined and explicitly
        // instantiated in fem_gradient_gpu.cc). Strides are passed as length-Dim
        // arrays, matching the host kernels, so the same call site serves 2D and
        // 3D. `nb` are the with-ghost grid points, `gsq`/`gsd` the gradient
        // quad/dim strides, `h` the grid spacing.
        template <class Element>
        void fem_gradient_gpu(const Real * nodal, Real * grad,
                              const Index_t * nb, const Index_t * nstride,
                              const Index_t * gstride, Index_t gsq, Index_t gsd,
                              const Real * h, Real alpha, bool increment);
        template <class Element>
        void fem_divergence_gpu(const Real * grad, Real * nodal,
                                const Index_t * nb, const Index_t * nstride,
                                const Index_t * gstride, Index_t gsq,
                                Index_t gsd, const Real * h,
                                const Real * quad_weights, Real alpha,
                                bool increment);
#endif

        // Compile-time unrolled 0..N-1 loop: the body is called with an
        // std::integral_constant, so loop-carried indices are template
        // parameters. This is what makes the constexpr table access
        // `Element::B[q][d][n]` a core constant expression that the compiler
        // folds (dropping the structural zeros / ±1 of a simplex), so the
        // element-generic kernel matches the old hand-unrolled code on every
        // compiler instead of relying on unroll heuristics.
        template <Index_t N, class F>
        inline void static_for(F && f) {
            [&]<Index_t... I>(std::integer_sequence<Index_t, I...>) {
                (f(std::integral_constant<Index_t, I>{}), ...);
            }(std::make_integer_sequence<Index_t, N>{});
        }

        // ===================================================================
        // Generic, element-templated host kernels
        // ===================================================================
        // These replace the per-dimension hand-written kernels above: the
        // element supplies B[q][d][n] and the node count, and the loops are
        // fully unrolled because NbQuad/NbNodes/Dim are compile-time, so for a
        // P1 simplex the compiler folds B's ±1 / zeros and reproduces the
        // hand-written code. Works for any element (e.g. Q1) with no new code.

        //! Gradient (nodal → quadrature): grad[q][d] = (Σ_n B[q][d][n] u[n])/h_d.
        template <class Element>
        void fem_gradient_host_generic(
            const Real * MUGRID_RESTRICT nodal_input,
            Real * MUGRID_RESTRICT gradient_output, const Index_t * nb_grid_pts,
            const Index_t * nodal_stride, const Index_t * grad_stride,
            Index_t grad_stride_q, Index_t grad_stride_d,
            const Real * grid_spacing, Real alpha, bool increment) {
            constexpr Dim_t Dim = Element::SpatialDim;
            constexpr Index_t NbNodes = Element::NbNodes;
            constexpr Index_t NbQuad = Element::NbQuad;

            Index_t node_lin[NbNodes];
            for (Index_t n = 0; n < NbNodes; ++n) {
                Index_t off = 0;
                for (Dim_t d = 0; d < Dim; ++d) {
                    off += fem_node_offset(n, d) * nodal_stride[d];
                }
                node_lin[n] = off;
            }
            Real inv_h[Dim];
            for (Dim_t d = 0; d < Dim; ++d) {
                inv_h[d] = alpha / grid_spacing[d];
            }

            auto pixel = [&](Index_t nodal_base, Index_t grad_base) {
                Real u[NbNodes];
                static_for<NbNodes>([&](auto Nn) {
                    constexpr Index_t n = decltype(Nn)::value;
                    u[n] = nodal_input[nodal_base + node_lin[n]];
                });
                // Contract first (B folds to ±1/0 at compile time), then write
                // with a single branch — keeps the strided stores out of the
                // increment test, matching the hand-written kernels.
                Real out[NbQuad][Dim];
                static_for<NbQuad>([&](auto Q) {
                    constexpr Index_t q = decltype(Q)::value;
                    static_for<Dim>([&](auto D) {
                        constexpr Index_t d = decltype(D)::value;
                        Real acc = 0.0;
                        static_for<NbNodes>([&](auto Nn) {
                            constexpr Index_t n = decltype(Nn)::value;
                            acc += Element::B[q][d][n] * u[n];
                        });
                        out[q][d] = acc * inv_h[d];
                    });
                });
                if (increment) {
                    static_for<NbQuad>([&](auto Q) {
                        constexpr Index_t q = decltype(Q)::value;
                        static_for<Dim>([&](auto D) {
                            constexpr Index_t d = decltype(D)::value;
                            gradient_output[grad_base + q * grad_stride_q +
                                            d * grad_stride_d] += out[q][d];
                        });
                    });
                } else {
                    static_for<NbQuad>([&](auto Q) {
                        constexpr Index_t q = decltype(Q)::value;
                        static_for<Dim>([&](auto D) {
                            constexpr Index_t d = decltype(D)::value;
                            gradient_output[grad_base + q * grad_stride_q +
                                            d * grad_stride_d] = out[q][d];
                        });
                    });
                }
            };

            if constexpr (Dim == 2) {
                for (Index_t iy = 0; iy < nb_grid_pts[1] - 1; ++iy) {
                    for (Index_t ix = 0; ix < nb_grid_pts[0] - 1; ++ix) {
                        pixel(ix * nodal_stride[0] + iy * nodal_stride[1],
                              ix * grad_stride[0] + iy * grad_stride[1]);
                    }
                }
            } else {
                for (Index_t iz = 0; iz < nb_grid_pts[2] - 1; ++iz) {
                    for (Index_t iy = 0; iy < nb_grid_pts[1] - 1; ++iy) {
                        for (Index_t ix = 0; ix < nb_grid_pts[0] - 1; ++ix) {
                            pixel(ix * nodal_stride[0] + iy * nodal_stride[1] +
                                      iz * nodal_stride[2],
                                  ix * grad_stride[0] + iy * grad_stride[1] +
                                      iz * grad_stride[2]);
                        }
                    }
                }
            }
        }

        //! Divergence / transpose (quadrature → nodal), scatter with weights:
        //! f[n] += alpha Σ_q w_q Σ_d B[q][d][n] g[q][d] / h_d.
        template <class Element>
        void fem_divergence_host_generic(
            const Real * MUGRID_RESTRICT gradient_input,
            Real * MUGRID_RESTRICT nodal_output, const Index_t * nb_grid_pts,
            const Index_t * grad_stride, Index_t grad_stride_q,
            Index_t grad_stride_d, const Index_t * nodal_stride,
            const Real * grid_spacing, const Real * quad_weights, Real alpha,
            bool increment) {
            constexpr Dim_t Dim = Element::SpatialDim;
            constexpr Index_t NbNodes = Element::NbNodes;
            constexpr Index_t NbQuad = Element::NbQuad;

            Index_t node_lin[NbNodes];
            for (Index_t n = 0; n < NbNodes; ++n) {
                Index_t off = 0;
                for (Dim_t d = 0; d < Dim; ++d) {
                    off += fem_node_offset(n, d) * nodal_stride[d];
                }
                node_lin[n] = off;
            }
            // Precompute the per-(quad, dim) coefficient w_q · α / h_d once,
            // hoisting the quadrature-weight and 1/h loads out of the pixel
            // loop (the hand-written kernels do the same). B folds against this
            // at compile time, dropping the zero entries.
            Real coeff[NbQuad][Dim];
            for (Index_t q = 0; q < NbQuad; ++q) {
                for (Dim_t d = 0; d < Dim; ++d) {
                    coeff[q][d] = alpha * quad_weights[q] / grid_spacing[d];
                }
            }

            // The scatter accumulates into shared nodes, so zero the whole
            // (owned + ghost) nodal region first unless incrementing.
            if (!increment) {
                if constexpr (Dim == 2) {
                    for (Index_t iy = 0; iy < nb_grid_pts[1]; ++iy) {
                        for (Index_t ix = 0; ix < nb_grid_pts[0]; ++ix) {
                            nodal_output[ix * nodal_stride[0] +
                                         iy * nodal_stride[1]] = 0.0;
                        }
                    }
                } else {
                    for (Index_t iz = 0; iz < nb_grid_pts[2]; ++iz) {
                        for (Index_t iy = 0; iy < nb_grid_pts[1]; ++iy) {
                            for (Index_t ix = 0; ix < nb_grid_pts[0]; ++ix) {
                                nodal_output[ix * nodal_stride[0] +
                                             iy * nodal_stride[1] +
                                             iz * nodal_stride[2]] = 0.0;
                            }
                        }
                    }
                }
            }

            auto pixel = [&](Index_t grad_base, Index_t nodal_base) {
                // Accumulate each node's contribution quad-point by quad-point
                // (B folds, so only the nodes touched by quad q contribute),
                // then scatter once — the structure of the hand-written kernel.
                Real fa[NbNodes] = {};
                static_for<NbQuad>([&](auto Q) {
                    constexpr Index_t q = decltype(Q)::value;
                    Real cg[Dim];
                    static_for<Dim>([&](auto D) {
                        constexpr Index_t d = decltype(D)::value;
                        cg[d] = coeff[q][d] *
                                gradient_input[grad_base + q * grad_stride_q +
                                               d * grad_stride_d];
                    });
                    static_for<NbNodes>([&](auto Nn) {
                        constexpr Index_t n = decltype(Nn)::value;
                        static_for<Dim>([&](auto D) {
                            constexpr Index_t d = decltype(D)::value;
                            fa[n] += Element::B[q][d][n] * cg[d];
                        });
                    });
                });
                static_for<NbNodes>([&](auto Nn) {
                    constexpr Index_t n = decltype(Nn)::value;
                    nodal_output[nodal_base + node_lin[n]] += fa[n];
                });
            };

            if constexpr (Dim == 2) {
                for (Index_t iy = 0; iy < nb_grid_pts[1] - 1; ++iy) {
                    for (Index_t ix = 0; ix < nb_grid_pts[0] - 1; ++ix) {
                        pixel(ix * grad_stride[0] + iy * grad_stride[1],
                              ix * nodal_stride[0] + iy * nodal_stride[1]);
                    }
                }
            } else {
                for (Index_t iz = 0; iz < nb_grid_pts[2] - 1; ++iz) {
                    for (Index_t iy = 0; iy < nb_grid_pts[1] - 1; ++iy) {
                        for (Index_t ix = 0; ix < nb_grid_pts[0] - 1; ++ix) {
                            pixel(ix * grad_stride[0] + iy * grad_stride[1] +
                                      iz * grad_stride[2],
                                  ix * nodal_stride[0] + iy * nodal_stride[1] +
                                      iz * nodal_stride[2]);
                        }
                    }
                }
            }
        }

    }  // namespace fem_gradient_kernels

    /**
     * @class FEMGradientOperator
     * @brief FEM gradient operator templated on the element type.
     *
     * apply() computes the gradient (nodal → quadrature points); transpose()
     * computes the (negative) discretised divergence (quadrature → nodal
     * points). The element-specific data (shape-function-gradient table B,
     * quadrature weights, node count) comes from the @p Element traits (see
     * fem_element.hh); the kernels are generic over it. Supported elements:
     * `P1Tri2D/3D` (triangles/tets) and, by adding a traits struct, Q1.
     */
    template <class Element>
    class FEMGradientOperator : public LinearOperator {
        static_assert(Element::SpatialDim == 2 || Element::SpatialDim == 3,
                      "FEMGradientOperator is only implemented for 2D and 3D");

       public:
        using Parent = LinearOperator;

        //! Spatial dimension of the element.
        static constexpr Dim_t Dim = Element::SpatialDim;
        //! Number of nodes per pixel/voxel (4 in 2D, 8 in 3D)
        static constexpr Index_t NB_NODES = Element::NbNodes;
        //! Number of quadrature points per pixel/voxel
        static constexpr Index_t NB_QUAD = Element::NbQuad;
        //! Spatial dimension
        static constexpr Dim_t DIM = Dim;

        /**
         * @brief Construct a FEM gradient operator.
         * @param grid_spacing Grid spacing in each direction (default: all 1.0)
         */
        explicit FEMGradientOperator(std::vector<Real> grid_spacing = {})
            : Parent{}, grid_spacing{std::move(grid_spacing)} {
            if (this->grid_spacing.empty()) {
                this->grid_spacing.resize(Dim, 1.0);
            }
            if (static_cast<Index_t>(this->grid_spacing.size()) != Dim) {
                std::stringstream err;
                err << "Grid spacing must have " << Dim
                    << " components for " << Dim << "D operator";
                throw RuntimeError(err.str());
            }
        }

        //! Default constructor is deleted
        FEMGradientOperator() = delete;
        //! Copy constructor is deleted
        FEMGradientOperator(const FEMGradientOperator & other) = delete;
        //! Move constructor
        FEMGradientOperator(FEMGradientOperator && other) = default;
        //! Destructor
        ~FEMGradientOperator() override = default;
        //! Copy assignment operator is deleted
        FEMGradientOperator & operator=(const FEMGradientOperator & other) = delete;
        //! Move assignment operator
        FEMGradientOperator & operator=(FEMGradientOperator && other) = default;

        // ---- Host interface ----
        void apply(const TypedFieldBase<Real> & nodal_field,
                   TypedFieldBase<Real> & gradient_field) const override {
            this->apply_impl(nodal_field, gradient_field, 1.0, false);
        }
        void apply_increment(const TypedFieldBase<Real> & nodal_field,
                             const Real & alpha,
                             TypedFieldBase<Real> & gradient_field)
            const override {
            this->apply_impl(nodal_field, gradient_field, alpha, true);
        }
        void transpose(const TypedFieldBase<Real> & gradient_field,
                       TypedFieldBase<Real> & nodal_field,
                       const std::vector<Real> & weights = {}) const override {
            this->transpose_impl(gradient_field, nodal_field, 1.0, false,
                                 weights);
        }
        void transpose_increment(const TypedFieldBase<Real> & gradient_field,
                                 const Real & alpha,
                                 TypedFieldBase<Real> & nodal_field,
                                 const std::vector<Real> & weights = {})
            const override {
            this->transpose_impl(gradient_field, nodal_field, alpha, true,
                                 weights);
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        // ---- Device interface ----
        void apply(const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
                   TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field)
            const {
            this->apply_impl(nodal_field, gradient_field, 1.0, false);
        }
        void apply_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field) const {
            this->apply_impl(nodal_field, gradient_field, alpha, true);
        }
        void transpose(
            const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights = {}) const {
            this->transpose_impl(gradient_field, nodal_field, 1.0, false,
                                 weights);
        }
        void transpose_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights = {}) const {
            this->transpose_impl(gradient_field, nodal_field, alpha, true,
                                 weights);
        }
#endif

        //! Number of output components (gradient components = spatial dim).
        Index_t get_nb_output_components() const override { return DIM; }
        //! Number of quadrature points per pixel/voxel.
        Index_t get_nb_quad_pts() const override { return NB_QUAD; }
        //! Number of input components (one scalar per grid point).
        Index_t get_nb_input_components() const override { return 1; }
        //! Spatial dimension.
        Dim_t get_spatial_dim() const override { return DIM; }

        //! Grid spacing.
        const std::vector<Real> & get_grid_spacing() const {
            return grid_spacing;
        }

        //! Quadrature weights (one per quadrature point): the element's
        //! fractional weights scaled by the cell volume.
        std::vector<Real> get_quadrature_weights() const {
            Real cell_volume = 1.0;
            for (Dim_t d = 0; d < Dim; ++d) {
                cell_volume *= grid_spacing[d];
            }
            std::vector<Real> weights(NB_QUAD);
            for (Index_t q = 0; q < NB_QUAD; ++q) {
                weights[q] = Element::Wfrac[q] * cell_volume;
            }
            return weights;
        }

        //! Stencil offset in pixels (the element spans [0, +1] in each axis).
        Shape_t get_offset() const override {
            if constexpr (Dim == 2) {
                return Shape_t{0, 0};
            } else {
                return Shape_t{0, 0, 0};
            }
        }

        //! Stencil shape in pixels (2 in every direction).
        Shape_t get_stencil_shape() const override {
            if constexpr (Dim == 2) {
                return Shape_t{2, 2};
            } else {
                return Shape_t{2, 2, 2};
            }
        }

        /**
         * @brief Ghost layers required by transpose().
         *
         * The transpose scatters into the same ghost buffers that apply()
         * reads (followed by ghost reduction), so it has the same ghost
         * requirement as apply().
         */
        GhostRequirement get_transpose_ghost_requirement() const override {
            return this->get_apply_ghost_requirement();
        }

        //! Shape function gradients as a flat (Fortran-order) array, scaled by
        //! the inverse grid spacing. Shape (DIM, NB_QUAD, 1, [2]*Dim). Node
        //! order is the binary corner index (x fastest), matching the stencil
        //! layout, and the data comes from the element traits (single source).
        std::vector<Real> get_coefficients() const {
            std::vector<Real> result;
            result.reserve(NB_QUAD * DIM * NB_NODES);
            for (Index_t node = 0; node < NB_NODES; ++node) {
                for (Index_t q = 0; q < NB_QUAD; ++q) {
                    for (Dim_t d = 0; d < DIM; ++d) {
                        result.push_back(Element::B[q][d][node] /
                                         grid_spacing[d]);
                    }
                }
            }
            return result;
        }

       private:
        std::vector<Real> grid_spacing;

        //! Operator name used in validation error messages.
        static const char * operator_name() {
            return Dim == 2 ? "FEMGradientOperator2D" : "FEMGradientOperator3D";
        }

        //! Common validation plus the operator-specific component-count check.
        const GlobalFieldCollection &
        validate_fields(const Field & nodal_field, const Field & gradient_field,
                        Index_t & nb_components) const {
            // Same collection, global, matching dimension, ghost layers (the
            // scatter-style transpose has the same requirement as apply).
            const auto & global_fc =
                this->check_fields(nodal_field, gradient_field,
                                   operator_name());

            const Index_t nb_nodal_components = nodal_field.get_nb_components();
            const Index_t nb_grad_components =
                gradient_field.get_nb_components();
            const Index_t expected_grad_components =
                this->get_nb_output_components() * nb_nodal_components;
            if (nb_grad_components != expected_grad_components) {
                std::stringstream err_msg;
                err_msg << "Component mismatch: Expected gradient field with "
                        << expected_grad_components << " components ("
                        << this->get_nb_output_components()
                        << " output components × " << nb_nodal_components
                        << " nodal components) but got " << nb_grad_components
                        << " components.";
                throw RuntimeError(err_msg.str());
            }
            nb_components = nb_nodal_components;
            return global_fc;
        }

        //! Host apply/gradient with optional increment.
        void apply_impl(const TypedFieldBase<Real> & nodal_field,
                        TypedFieldBase<Real> & gradient_field, Real alpha,
                        bool increment) const {
            Index_t nb_components;
            const auto & collection =
                this->validate_fields(nodal_field, gradient_field,
                                      nb_components);
            const auto nb_grid_pts =
                collection.get_nb_subdomain_grid_pts_with_ghosts();
            const Real * nodal = nodal_field.data();
            Real * gradient = gradient_field.data();
            const Index_t nb_sub = this->get_nb_input_components();

            // AoS layout: components fastest, then sub-pts/quad/operators, then
            // the spatial axes. Build the per-axis strides as arrays so the
            // element-generic kernel handles 2D and 3D uniformly.
            std::array<Index_t, Dim> nbg{}, nstride{}, gstride{};
            std::array<Real, Dim> h{};
            Index_t span = 1;
            for (Dim_t d = 0; d < Dim; ++d) {
                nbg[d] = nb_grid_pts[d];
                nstride[d] = nb_components * nb_sub * span;
                gstride[d] = nb_components * DIM * NB_QUAD * span;
                h[d] = grid_spacing[d];
                span *= nb_grid_pts[d];
            }
            const Index_t grad_stride_d = nb_components;
            const Index_t grad_stride_q = nb_components * DIM;

            for (Index_t comp = 0; comp < nb_components; ++comp) {
                fem_gradient_kernels::fem_gradient_host_generic<Element>(
                    nodal + comp, gradient + comp, nbg.data(), nstride.data(),
                    gstride.data(), grad_stride_q, grad_stride_d, h.data(),
                    alpha, increment);
            }
        }

        //! Host transpose/divergence with optional increment.
        void transpose_impl(const TypedFieldBase<Real> & gradient_field,
                            TypedFieldBase<Real> & nodal_field, Real alpha,
                            bool increment,
                            const std::vector<Real> & weights) const {
            Index_t nb_components;
            const auto & collection =
                this->validate_fields(nodal_field, gradient_field,
                                      nb_components);
            const auto nb_grid_pts =
                collection.get_nb_subdomain_grid_pts_with_ghosts();
            const Real * gradient = gradient_field.data();
            Real * nodal = nodal_field.data();
            const std::vector<Real> quad_weights =
                weights.empty() ? this->get_quadrature_weights() : weights;
            const Index_t nb_sub = this->get_nb_input_components();

            std::array<Index_t, Dim> nbg{}, nstride{}, gstride{};
            std::array<Real, Dim> h{};
            Index_t span = 1;
            for (Dim_t d = 0; d < Dim; ++d) {
                nbg[d] = nb_grid_pts[d];
                nstride[d] = nb_components * nb_sub * span;
                gstride[d] = nb_components * DIM * NB_QUAD * span;
                h[d] = grid_spacing[d];
                span *= nb_grid_pts[d];
            }
            const Index_t grad_stride_d = nb_components;
            const Index_t grad_stride_q = nb_components * DIM;

            for (Index_t comp = 0; comp < nb_components; ++comp) {
                fem_gradient_kernels::fem_divergence_host_generic<Element>(
                    gradient + comp, nodal + comp, nbg.data(), gstride.data(),
                    grad_stride_q, grad_stride_d, nstride.data(), h.data(),
                    quad_weights.data(), alpha, increment);
            }
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        //! Device apply/gradient with optional increment.
        void apply_impl(
            const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            Real alpha, bool increment) const {
            Index_t nb_components;
            const auto & collection =
                this->validate_fields(nodal_field, gradient_field,
                                      nb_components);
            const auto nb_grid_pts =
                collection.get_nb_subdomain_grid_pts_with_ghosts();
            const Real * nodal = nodal_field.view().data();
            Real * gradient = gradient_field.view().data();
            const Index_t nb_input = this->get_nb_input_components();

            // SoA layout: spatial axes fastest, then sub-pts/quad/dim, then
            // components. Build per-axis strides as arrays so one call site
            // serves 2D and 3D (the kernel is element-generic).
            std::array<Index_t, Dim> nbg{}, nstride{}, gstride{};
            std::array<Real, Dim> h{};
            Index_t span = 1;
            for (Dim_t d = 0; d < Dim; ++d) {
                nbg[d] = nb_grid_pts[d];
                nstride[d] = span;
                gstride[d] = span;
                h[d] = grid_spacing[d];
                span *= nb_grid_pts[d];
            }
            const Index_t gsq = span;
            const Index_t gsd = span * NB_QUAD;
            const Index_t nodal_stride_c = span * nb_input;
            const Index_t grad_stride_c = span * NB_QUAD * DIM;
            for (Index_t comp = 0; comp < nb_components; ++comp) {
                fem_gradient_kernels::fem_gradient_gpu<Element>(
                    nodal + comp * nodal_stride_c,
                    gradient + comp * grad_stride_c, nbg.data(),
                    nstride.data(), gstride.data(), gsq, gsd, h.data(), alpha,
                    increment);
            }
        }

        //! Device transpose/divergence with optional increment.
        void transpose_impl(
            const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field, Real alpha,
            bool increment, const std::vector<Real> & weights) const {
            Index_t nb_components;
            const auto & collection =
                this->validate_fields(nodal_field, gradient_field,
                                      nb_components);
            const auto nb_grid_pts =
                collection.get_nb_subdomain_grid_pts_with_ghosts();
            const Real * gradient = gradient_field.view().data();
            Real * nodal = nodal_field.view().data();
            const std::vector<Real> quad_weights =
                weights.empty() ? this->get_quadrature_weights() : weights;
            const Index_t nb_input = this->get_nb_input_components();

            std::array<Index_t, Dim> nbg{}, nstride{}, gstride{};
            std::array<Real, Dim> h{};
            Index_t span = 1;
            for (Dim_t d = 0; d < Dim; ++d) {
                nbg[d] = nb_grid_pts[d];
                nstride[d] = span;
                gstride[d] = span;
                h[d] = grid_spacing[d];
                span *= nb_grid_pts[d];
            }
            const Index_t gsq = span;
            const Index_t gsd = span * NB_QUAD;
            const Index_t nodal_stride_c = span * nb_input;
            const Index_t grad_stride_c = span * NB_QUAD * DIM;
            // The wrapper folds the quadrature weights into per-(quad, dim)
            // coefficients on the host, so no device staging is needed.
            for (Index_t comp = 0; comp < nb_components; ++comp) {
                fem_gradient_kernels::fem_divergence_gpu<Element>(
                    gradient + comp * grad_stride_c,
                    nodal + comp * nodal_stride_c, nbg.data(), nstride.data(),
                    gstride.data(), gsq, gsd, h.data(), quad_weights.data(),
                    alpha, increment);
            }
        }
#endif
    };

    //! 2D linear-triangle FEM gradient. Preserves the historical name.
    using FEMGradientOperator2D = FEMGradientOperator<P1Tri2D>;
    //! 3D linear-tetrahedra FEM gradient. Preserves the historical name.
    using FEMGradientOperator3D = FEMGradientOperator<P1Tet3D>;
    //! 2D bilinear-quad (Q1) FEM gradient.
    using FEMGradientOperatorQ1_2D = FEMGradientOperator<Q1Quad2D>;
    //! 3D trilinear-hex (Q1) FEM gradient.
    using FEMGradientOperatorQ1_3D = FEMGradientOperator<Q1Hex3D>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_HH_
