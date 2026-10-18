/**
 * @file   laplace.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   26 Jun 2026
 *
 * @brief  Dimension-templated hard-coded Laplace operator (2D/3D)
 *
 * A single `template <Dim_t Dim> LaplaceOperator` covers the 2D (5-point)
 * and 3D (7-point) stencils: apply/apply_increment/transpose, field
 * validation, device dispatch and metadata depend on the dimension only
 * through the stencil size (`LaplaceTraits<Dim>`) and the stencil itself,
 * which is selected with `if constexpr`. `LaplaceOperator2D` and
 * `LaplaceOperator3D` are type aliases for the two instantiations; the
 * Python bindings and the C++ tests use these names.
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

#ifndef SRC_LIBMUGRID_OPERATORS_LAPLACE_HH_
#define SRC_LIBMUGRID_OPERATORS_LAPLACE_HH_

#include "core/types.hh"
#include "field/field_typed.hh"
#include "collection/field_collection_global.hh"
#include "memory/memory_space.hh"
#include "operators/linear.hh"

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
#include "memory/gpu_runtime.hh"
#endif

#include <array>
#include <sstream>
#include <string>
#include <vector>

namespace muGrid {

    // Kernel implementations (defined in laplace.cc for the host, laplace_gpu.cc
    // for the device). The dimension-specialised kernels keep their distinct
    // signatures; the templated operator below selects the matching one.
    namespace laplace_kernels {

        /**
         * @brief Apply the 5-point 2D Laplace stencil on host.
         *
         * Computes, for every point (ix, iy) with 1 <= ix < nx - 1 and
         * 1 <= iy < ny - 1 (all points whose stencil lies inside the ghosted
         * buffer),
         *
         *     out(i) [+]= sum_d w[d] (in(i + e_d) - 2 in(i) + in(i - e_d)),
         *
         * i.e. w[0] on the two x neighbours, w[1] on the two y neighbours and
         * -2 (w[0] + w[1]) on the centre; for w = scale * (1, 1) the classic
         * scale * [0, 1, 0; 1, -4, 1; 0, 1, 0]. Points on the outermost layer
         * are not written. Cost O(nx ny), one pass over input and output.
         *
         * @param input     Scalar input buffer including ghosts
         * @param output    Scalar output buffer (same layout as input)
         * @param nx, ny    Buffer extent per axis, ghosts included
         * @param stride_x, stride_y Element strides per axis
         * @param w         Per-axis neighbour weights (`Dim` entries)
         * @param increment Add to `output` instead of overwriting it
         */
        template <typename T>
        void laplace_2d_host(
            const T* MUGRID_RESTRICT input,
            T* MUGRID_RESTRICT output,
            Index_t nx, Index_t ny,
            Index_t stride_x, Index_t stride_y,
            const T * w,
            bool increment = false);

        /**
         * @brief Apply the 7-point 3D Laplace stencil on host.
         *
         * Stencil: w[d] on the two face neighbours along axis d and
         * -2 (w[0] + w[1] + w[2]) on the centre, evaluated on every point
         * with 1 <= i_d < n_d - 1; parameters as for laplace_2d_host.
         */
        template <typename T>
        void laplace_3d_host(
            const T* MUGRID_RESTRICT input,
            T* MUGRID_RESTRICT output,
            Index_t nx, Index_t ny, Index_t nz,
            Index_t stride_x, Index_t stride_y, Index_t stride_z,
            const T * w,
            bool increment = false);

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        // Single GPU launcher per stencil (the CUDA/HIP backend split lives
        // inside gpu_runtime.hh, not in the operator), matching the generic and
        // stiffness operators' *_gpu convention. Same arguments and computed
        // region as the host kernels.
        template <typename T>
        void laplace_2d_gpu(
            const T* input, T* output, Index_t nx, Index_t ny,
            Index_t stride_x, Index_t stride_y, const T * w,
            bool increment = false);
        template <typename T>
        void laplace_3d_gpu(
            const T* input, T* output, Index_t nx, Index_t ny, Index_t nz,
            Index_t stride_x, Index_t stride_y, Index_t stride_z,
            const T * w, bool increment = false);
#endif

    }  // namespace laplace_kernels

    /**
     * @struct LaplaceTraits
     * @brief Dimension-varying facts of the Laplace stencil.
     */
    template <Dim_t Dim>
    struct LaplaceTraits;
    template <>
    struct LaplaceTraits<2> {
        static constexpr Index_t nb_stencil_pts = 5;  //!< 5-point stencil
    };
    template <>
    struct LaplaceTraits<3> {
        static constexpr Index_t nb_stencil_pts = 7;  //!< 7-point stencil
    };

    /**
     * @class LaplaceOperator
     * @brief Hard-coded Laplace operator with an optimized stencil (2D/3D).
     *
     * Uses a 5-point stencil in 2D ([0,1,0; 1,-4,1; 0,1,0]) and a 7-point
     * stencil in 3D (center -6, six neighbours +1). The output is multiplied by
     * a scale factor, which can incorporate grid spacing and sign conventions
     * (e.g. for a positive-definite operator usable with CG solvers).
     *
     * With a grid spacing the stencil is the second-order central-difference
     * Laplacian of an anisotropic grid,
     *
     *     (L u)_i = sum_{d < Dim} w_d (u_{i+e_d} - 2 u_i + u_{i-e_d}),
     *     w_d = scale / h_d^2,
     *
     * i.e. the two neighbours along axis d carry `scale / h_d^2` and the
     * centre `-2 scale * sum_d 1 / h_d^2`. Without a grid spacing every axis
     * carries `scale`, the unit-spacing stencil above. The truncation error
     * is O(h_d^2) per axis.
     *
     * Like every LinearOperator, it assumes nothing about boundary
     * conditions: it reads one ghost layer on each side (filled beforehand,
     * e.g. periodically by the domain decomposition) and writes every point
     * whose stencil lies inside the ghosted buffer.
     *
     * Conventions to keep in mind when mixing it with other operators:
     * - With scale = 1/h^2 this is the finite-difference div grad, which is
     *   negative semidefinite; CG needs scale = -1/h^2. The FE stiffness of
     *   FEMGradientOperator, transpose(apply(u)), is the positive
     *   semidefinite -h^(d-2) div grad (the cell-volume weighting h^d times
     *   1/h^2 per derivative).
     * - Without a grid spacing the single scale applies to every axis;
     *   pass the grid spacing for anisotropic grids (the generic
     *   operator with per-axis coefficients 1/h_d^2 is the slower
     *   equivalent).
     *
     * The hard-coded implementation is faster than the generic convolution
     * operator (GenericLinearOperator) applying the same coefficients: it
     * makes a single unit-stride pass over input and output with the Dim
     * weights in registers, whereas the generic kernel sweeps the whole
     * output once per stencil non-zero (5 or 7 passes) and resolves offsets
     * at run time. The fixed access pattern also lets the compiler vectorize
     * the x loop. (A speed-up of ~3-10x is quoted for this comparison;
     * TODO: rationale for the 3-10x figure -- benchmark and grid sizes?)
     *
     * Since the Laplacian is self-adjoint (symmetric), the transpose operation
     * is identical to the forward apply operation.
     */
    template <Dim_t Dim>
    class LaplaceOperator : public LinearOperator {
        static_assert(Dim == 2 || Dim == 3,
                      "LaplaceOperator is only implemented for 2D and 3D");

    public:
        using Parent = LinearOperator;

        //! Number of stencil points (compile-time constant)
        static constexpr Index_t NB_STENCIL_PTS =
            LaplaceTraits<Dim>::nb_stencil_pts;

        /**
         * @brief Construct a Laplace operator on a unit-spacing grid.
         * @param scale Scale factor applied to the output (default: 1.0)
         */
        explicit LaplaceOperator(Real scale = 1.0)
            : Parent{}, scale{scale}, grid_spacing{unit_spacing()},
              weights{compute_weights(scale, grid_spacing)} {}

        /**
         * @brief Construct a Laplace operator on an anisotropic grid.
         * @param grid_spacing Grid spacing per axis (`Dim` positive values);
         *        the two neighbours along axis d are weighted by
         *        `scale / h_d^2`
         * @param scale Scale factor applied to the output (default: 1.0)
         */
        LaplaceOperator(const std::vector<Real> & grid_spacing,
                        Real scale = 1.0)
            : Parent{}, scale{scale},
              grid_spacing{checked_spacing(grid_spacing)},
              weights{compute_weights(scale, this->grid_spacing)} {}

        //! Default constructor is deleted
        LaplaceOperator() = delete;

        //! Copy constructor is deleted
        LaplaceOperator(const LaplaceOperator & other) = delete;

        //! Move constructor
        LaplaceOperator(LaplaceOperator && other) = default;

        //! Destructor
        ~LaplaceOperator() override = default;

        //! Copy assignment operator is deleted
        LaplaceOperator & operator=(const LaplaceOperator & other) = delete;

        //! Move assignment operator
        LaplaceOperator & operator=(LaplaceOperator && other) = default;

        /**
         * @brief Apply the Laplace operator on host fields.
         *        Computes output = scale * Laplace(input).
         */
        void apply(const TypedFieldBase<Real> & input_field,
                   TypedFieldBase<Real> & output_field) const override {
            this->apply_impl<Real>(input_field, output_field, 1.0, false);
        }

        /**
         * @brief Apply with increment: output += alpha * scale * Laplace(input).
         */
        void apply_increment(const TypedFieldBase<Real> & input_field,
                             const Real & alpha,
                             TypedFieldBase<Real> & output_field)
            const override {
            this->apply_impl<Real>(input_field, output_field, alpha, true);
        }

        /**
         * @brief Throws if quadrature weights are passed: the Laplace stencil
         *        has no quadrature points, so accepting (and ignoring) them
         *        would silently diverge from the other LinearOperator
         *        implementations, which scale their transpose by the weights.
         */
        static void check_no_weights(const std::vector<Real> & weights) {
            if (not weights.empty()) {
                throw RuntimeError(
                    std::string(operator_name()) +
                    " does not support quadrature weights in "
                    "transpose()/transpose_increment(); the Laplacian is "
                    "self-adjoint and has no quadrature points, so weights "
                    "would be silently ignored");
            }
        }

        /**
         * @brief Apply the transpose (identical to apply for the self-adjoint
         *        Laplacian).
         */
        void transpose(const TypedFieldBase<Real> & input_field,
                       TypedFieldBase<Real> & output_field,
                       const std::vector<Real> & weights = {}) const override {
            check_no_weights(weights);
            this->apply(input_field, output_field);
        }

        /**
         * @brief Apply the transpose with increment (identical to
         *        apply_increment for the self-adjoint Laplacian).
         */
        void transpose_increment(const TypedFieldBase<Real> & input_field,
                                 const Real & alpha,
                                 TypedFieldBase<Real> & output_field,
                                 const std::vector<Real> & weights = {})
            const override {
            check_no_weights(weights);
            this->apply_increment(input_field, alpha, output_field);
        }

        //! Single-precision (Real32) host overloads.
        void apply(const TypedFieldBase<Real32> & input_field,
                   TypedFieldBase<Real32> & output_field) const override {
            this->apply_impl<Real32>(input_field, output_field, 1.0f, false);
        }
        void apply_increment(const TypedFieldBase<Real32> & input_field,
                             const Real32 & alpha,
                             TypedFieldBase<Real32> & output_field)
            const override {
            this->apply_impl<Real32>(input_field, output_field, alpha, true);
        }
        void transpose(const TypedFieldBase<Real32> & input_field,
                       TypedFieldBase<Real32> & output_field,
                       const std::vector<Real> & weights = {}) const override {
            check_no_weights(weights);
            this->apply(input_field, output_field);
        }
        void transpose_increment(const TypedFieldBase<Real32> & input_field,
                                 const Real32 & alpha,
                                 TypedFieldBase<Real32> & output_field,
                                 const std::vector<Real> & weights = {})
            const override {
            check_no_weights(weights);
            this->apply_increment(input_field, alpha, output_field);
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        //! Apply the Laplace operator on device (GPU) fields.
        void apply(const TypedFieldBase<Real, DefaultDeviceSpace> & input_field,
                   TypedFieldBase<Real, DefaultDeviceSpace> & output_field)
            const {
            this->apply_impl<Real>(input_field, output_field, 1.0, false);
        }

        //! Apply with increment on device (GPU) fields.
        void apply_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & input_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & output_field) const {
            this->apply_impl<Real>(input_field, output_field, alpha, true);
        }

        //! Single-precision (Real32) device overloads.
        void apply(
            const TypedFieldBase<Real32, DefaultDeviceSpace> & input_field,
            TypedFieldBase<Real32, DefaultDeviceSpace> & output_field) const {
            this->apply_impl<Real32>(input_field, output_field, 1.0f, false);
        }
        void apply_increment(
            const TypedFieldBase<Real32, DefaultDeviceSpace> & input_field,
            const Real32 & alpha,
            TypedFieldBase<Real32, DefaultDeviceSpace> & output_field) const {
            this->apply_impl<Real32>(input_field, output_field, alpha, true);
        }
#endif

        //! Number of output components (always 1 for the Laplacian).
        Index_t get_nb_output_components() const override { return 1; }

        //! Number of quadrature points (always 1 for the Laplacian).
        Index_t get_nb_quad_pts() const override { return 1; }

        //! Number of input components (always 1 for the Laplacian).
        Index_t get_nb_input_components() const override { return 1; }

        //! Spatial dimension.
        Dim_t get_spatial_dim() const override { return Dim; }

        //! Number of stencil points (5 in 2D, 7 in 3D).
        Index_t get_nb_stencil_pts() const { return NB_STENCIL_PTS; }

        //! Scale factor applied to the output.
        Real get_scale() const { return scale; }

        //! Grid spacing per axis (all 1 for the scale-only constructor)
        std::vector<Real> get_grid_spacing() const {
            return std::vector<Real>(grid_spacing.begin(), grid_spacing.end());
        }

        //! Stencil weight of the two neighbours along each axis,
        //! `scale / h_d^2`
        const std::array<Real, Dim> & get_weights() const { return weights; }

        //! Stencil offset in pixels (centered: -1 in every direction).
        Shape_t get_offset() const override {
            if constexpr (Dim == 2) {
                return Shape_t{-1, -1};
            } else {
                return Shape_t{-1, -1, -1};
            }
        }

        //! Stencil shape in pixels (3 in every direction).
        Shape_t get_stencil_shape() const override {
            if constexpr (Dim == 2) {
                return Shape_t{3, 3};
            } else {
                return Shape_t{3, 3, 3};
            }
        }

        /**
         * @brief Stencil coefficients as a flat column-major (x fastest)
         *        3^Dim array, the layout `GenericLinearOperator` and the
         *        Python `coefficients` property use.
         *
         * 2D: [0, w_y, 0, w_x, -2 (w_x + w_y), w_x, 0, w_y, 0] with
         *     w_d = scale / h_d^2 (all equal to `scale` on a unit grid).
         * 3D: 27 entries, center (index 13) = -2 (w_x + w_y + w_z), the two
         *     neighbours along x (index 13 +- 1) = w_x, along y (+- 3) = w_y,
         *     along z (+- 9) = w_z, rest zero.
         *
         * The flat index of stencil offset (i_x, i_y, i_z) in {0, 1, 2}^3 is
         * i_x + 3 i_y + 9 i_z (x fastest), so the x neighbours must sit at
         * +- 1 and the z neighbours at +- 9; only then does a
         * GenericLinearOperator built from these coefficients reproduce
         * apply() on an anisotropic grid (with equal weights an x/z swap
         * would go unnoticed).
         */
        std::vector<Real> get_coefficients() const {
            const auto & w{this->weights};
            if constexpr (Dim == 2) {
                return {0.0, w[1], 0.0,
                        w[0], -2.0 * (w[0] + w[1]), w[0],
                        0.0, w[1], 0.0};
            } else {
                std::vector<Real> stencil(27, 0.0);
                stencil[13] = -2.0 * (w[0] + w[1] + w[2]);  // center [1,1,1]
                stencil[13 - 1] = w[0];                     // [0,1,1]
                stencil[13 + 1] = w[0];                     // [2,1,1]
                stencil[13 - 3] = w[1];                     // [1,0,1]
                stencil[13 + 3] = w[1];                     // [1,2,1]
                stencil[13 - 9] = w[2];                     // [1,1,0]
                stencil[13 + 9] = w[2];                     // [1,1,2]
                return stencil;
            }
        }

    private:
        Real scale;
        std::array<Real, Dim> grid_spacing;
        //! scale / h_d^2 per axis
        std::array<Real, Dim> weights;

        static std::array<Real, Dim> unit_spacing() {
            std::array<Real, Dim> h{};
            h.fill(1.0);
            return h;
        }

        static std::array<Real, Dim>
        checked_spacing(const std::vector<Real> & grid_spacing) {
            if (grid_spacing.size() != static_cast<std::size_t>(Dim)) {
                std::stringstream err_msg{};
                err_msg << operator_name() << ": grid_spacing must have "
                        << Dim << " entries, got " << grid_spacing.size();
                throw RuntimeError{err_msg.str()};
            }
            std::array<Real, Dim> h{};
            for (Dim_t d = 0; d < Dim; ++d) {
                if (!(grid_spacing[d] > 0.0)) {
                    std::stringstream err_msg{};
                    err_msg << operator_name()
                            << ": grid_spacing must be positive, got "
                            << grid_spacing[d] << " on axis " << d;
                    throw RuntimeError{err_msg.str()};
                }
                h[d] = grid_spacing[d];
            }
            return h;
        }

        static std::array<Real, Dim>
        compute_weights(Real scale, const std::array<Real, Dim> & h) {
            std::array<Real, Dim> w{};
            for (Dim_t d = 0; d < Dim; ++d) {
                w[d] = scale / (h[d] * h[d]);
            }
            return w;
        }

        //! The per-axis weights times alpha, in the field precision
        template <typename T>
        std::array<T, Dim> effective_weights(T alpha) const {
            std::array<T, Dim> w{};
            for (Dim_t d = 0; d < Dim; ++d) {
                w[d] = alpha * static_cast<T>(this->weights[d]);
            }
            return w;
        }

        //! Operator name used in validation error messages.
        static const char * operator_name() {
            return Dim == 2 ? "LaplaceOperator2D" : "LaplaceOperator3D";
        }

        /**
         * @brief Common validation for the apply paths. The traversal
         *        addresses one scalar per pixel with dense column-major
         *        pixel strides, so both fields must be scalar (a
         *        multi-component field would silently mix components with
         *        spatial neighbours) and the pixel layout must match.
         */
        const GlobalFieldCollection &
        validate_fields(const Field & input_field,
                        const Field & output_field) const {
            const auto & collection =
                this->check_fields(input_field, output_field, operator_name());
            this->check_pixel_storage_order(collection, operator_name());
            for (const Field * field : {&input_field, &output_field}) {
                const Index_t nb_dof_per_pixel{field->get_nb_components() *
                                               field->get_nb_sub_pts()};
                if (nb_dof_per_pixel != 1) {
                    std::stringstream err_msg{};
                    err_msg << operator_name()
                            << " only supports scalar fields (one degree of "
                               "freedom per pixel), but field '"
                            << field->get_name() << "' has "
                            << field->get_nb_components()
                            << " component(s) on "
                            << field->get_nb_sub_pts() << " sub-point(s)";
                    throw RuntimeError{err_msg.str()};
                }
            }
            return collection;
        }

        /**
         * @brief Host apply with optional increment. The Laplacian is
         *        self-adjoint, so apply and transpose share this requirement.
         */
        template <typename T>
        void apply_impl(const TypedFieldBase<T> & input_field,
                        TypedFieldBase<T> & output_field, T alpha,
                        bool increment) const {
            const auto & collection =
                this->validate_fields(input_field, output_field);
            const auto nb_grid_pts =
                collection.get_nb_subdomain_grid_pts_with_ghosts();
            const T * input = input_field.data();
            T * output = output_field.data();
            const auto w = this->effective_weights<T>(alpha);

            // Scalar field (one value per pixel, so AoS and SoA coincide)
            // with column-major pixels: stride_x = 1, stride_y = nx,
            // stride_z = nx*ny, ghosts included.
            if constexpr (Dim == 2) {
                laplace_kernels::laplace_2d_host<T>(
                    input, output, nb_grid_pts[0], nb_grid_pts[1], 1,
                    nb_grid_pts[0], w.data(), increment);
            } else {
                laplace_kernels::laplace_3d_host<T>(
                    input, output, nb_grid_pts[0], nb_grid_pts[1],
                    nb_grid_pts[2], 1, nb_grid_pts[0],
                    nb_grid_pts[0] * nb_grid_pts[1], w.data(), increment);
            }
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        //! Device apply with optional increment. A scalar field has the same
        //! strides in SoA (device) as in AoS (host) storage.
        template <typename T>
        void apply_impl(
            const TypedFieldBase<T, DefaultDeviceSpace> & input_field,
            TypedFieldBase<T, DefaultDeviceSpace> & output_field, T alpha,
            bool increment) const {
            DeviceGuard device_guard{input_field.get_device_id()};
            const auto & collection =
                this->validate_fields(input_field, output_field);
            const auto nb_grid_pts =
                collection.get_nb_subdomain_grid_pts_with_ghosts();
            const T * input = input_field.view().data();
            T * output = output_field.view().data();
            const auto w = this->effective_weights<T>(alpha);

            if constexpr (Dim == 2) {
                laplace_kernels::laplace_2d_gpu<T>(
                    input, output, nb_grid_pts[0], nb_grid_pts[1], 1,
                    nb_grid_pts[0], w.data(), increment);
            } else {
                laplace_kernels::laplace_3d_gpu<T>(
                    input, output, nb_grid_pts[0], nb_grid_pts[1],
                    nb_grid_pts[2], 1, nb_grid_pts[0],
                    nb_grid_pts[0] * nb_grid_pts[1], w.data(), increment);
            }
        }
#endif
    };

    //! 2D Laplace operator (5-point stencil); name used by the bindings.
    using LaplaceOperator2D = LaplaceOperator<2>;
    //! 3D Laplace operator (7-point stencil); name used by the bindings.
    using LaplaceOperator3D = LaplaceOperator<3>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_LAPLACE_HH_
