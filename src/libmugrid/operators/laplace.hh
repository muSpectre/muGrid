/**
 * @file   laplace.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   26 Jun 2026
 *
 * @brief  Dimension-templated hard-coded Laplace operator (2D/3D)
 *
 * The 2D (5-point) and 3D (7-point) Laplace operators were ~90% identical: the
 * apply/apply_increment/transpose quartet, the field validation, the device
 * dispatch and the metadata differed only in the dimension and the stencil
 * size. They are unified here into a single `template <Dim_t Dim>
 * LaplaceOperator`, with the dimension-varying facts in `LaplaceTraits<Dim>`
 * and the only true divergence — the stencil itself — selected with
 * `if constexpr`. The historical class names remain as type aliases
 * (LaplaceOperator2D / LaplaceOperator3D) so no binding or downstream code
 * changes.
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

#include <vector>

namespace muGrid {

    // Kernel implementations (defined in laplace.cc for the host, laplace_gpu.cc
    // for the device). The dimension-specialised kernels keep their distinct
    // signatures; the templated operator below selects the matching one.
    namespace laplace_kernels {

        /**
         * @brief Apply 5-point 2D Laplace stencil on host.
         *
         * Stencil: scale * [0, 1, 0; 1,-4, 1; 0, 1, 0]
         */
        template <typename T>
        void laplace_2d_host(
            const T* MUGRID_RESTRICT input,
            T* MUGRID_RESTRICT output,
            Index_t nx, Index_t ny,
            Index_t stride_x, Index_t stride_y,
            T scale,
            bool increment = false);

        /**
         * @brief Apply 7-point 3D Laplace stencil on host.
         *
         * Stencil: center = -6*scale, the six face neighbours = +scale.
         */
        template <typename T>
        void laplace_3d_host(
            const T* MUGRID_RESTRICT input,
            T* MUGRID_RESTRICT output,
            Index_t nx, Index_t ny, Index_t nz,
            Index_t stride_x, Index_t stride_y, Index_t stride_z,
            T scale,
            bool increment = false);

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        // Single GPU launcher per stencil (the CUDA/HIP backend split lives
        // inside gpu_runtime.hh, not in the operator), matching the generic and
        // stiffness operators' *_gpu convention.
        template <typename T>
        void laplace_2d_gpu(
            const T* input, T* output, Index_t nx, Index_t ny,
            Index_t stride_x, Index_t stride_y, T scale,
            bool increment = false);
        template <typename T>
        void laplace_3d_gpu(
            const T* input, T* output, Index_t nx, Index_t ny, Index_t nz,
            Index_t stride_x, Index_t stride_y, Index_t stride_z, T scale,
            bool increment = false);
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
     * The hard-coded implementation provides significantly better performance
     * (~3-10x) than the generic convolution operator due to compile-time known
     * memory access patterns that enable SIMD vectorization.
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
         * @brief Construct a Laplace operator.
         * @param scale Scale factor applied to the output (default: 1.0)
         */
        explicit LaplaceOperator(Real scale = 1.0) : Parent{}, scale{scale} {}

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
         * @brief Apply the transpose (identical to apply for the self-adjoint
         *        Laplacian; weights are ignored).
         */
        void transpose(const TypedFieldBase<Real> & input_field,
                       TypedFieldBase<Real> & output_field,
                       const std::vector<Real> & weights = {}) const override {
            (void)weights;
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
            (void)weights;
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
            (void)weights;
            this->apply(input_field, output_field);
        }
        void transpose_increment(const TypedFieldBase<Real32> & input_field,
                                 const Real32 & alpha,
                                 TypedFieldBase<Real32> & output_field,
                                 const std::vector<Real> & weights = {})
            const override {
            (void)weights;
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
         * @brief Stencil coefficients in reshaped (row-major) form, scaled.
         *
         * 2D: [0, 1, 0, 1, -4, 1, 0, 1, 0] * scale.
         * 3D: 27 entries, center (index 13) = -6*scale, six face neighbours
         *     = scale, rest zero.
         */
        std::vector<Real> get_coefficients() const {
            if constexpr (Dim == 2) {
                return {0.0, scale, 0.0,
                        scale, -4.0 * scale, scale,
                        0.0, scale, 0.0};
            } else {
                std::vector<Real> stencil(27, 0.0);
                stencil[13] = -6.0 * scale;  // center [1,1,1]
                stencil[13 - 1] = scale;     // [1,1,0]
                stencil[13 + 1] = scale;     // [1,1,2]
                stencil[13 - 3] = scale;     // [1,0,1]
                stencil[13 + 3] = scale;     // [1,2,1]
                stencil[13 - 9] = scale;     // [0,1,1]
                stencil[13 + 9] = scale;     // [2,1,1]
                return stencil;
            }
        }

    private:
        Real scale;

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
            const T effective_scale = alpha * static_cast<T>(this->scale);

            // ArrayOfStructures layout: stride_x = 1, stride_y = nx,
            // stride_z = nx*ny.
            if constexpr (Dim == 2) {
                laplace_kernels::laplace_2d_host<T>(
                    input, output, nb_grid_pts[0], nb_grid_pts[1], 1,
                    nb_grid_pts[0], effective_scale, increment);
            } else {
                laplace_kernels::laplace_3d_host<T>(
                    input, output, nb_grid_pts[0], nb_grid_pts[1],
                    nb_grid_pts[2], 1, nb_grid_pts[0],
                    nb_grid_pts[0] * nb_grid_pts[1], effective_scale, increment);
            }
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        //! Device apply with optional increment.
        template <typename T>
        void apply_impl(
            const TypedFieldBase<T, DefaultDeviceSpace> & input_field,
            TypedFieldBase<T, DefaultDeviceSpace> & output_field, T alpha,
            bool increment) const {
            const auto & collection =
                this->validate_fields(input_field, output_field);
            const auto nb_grid_pts =
                collection.get_nb_subdomain_grid_pts_with_ghosts();
            const T * input = input_field.view().data();
            T * output = output_field.view().data();
            const T effective_scale = alpha * static_cast<T>(this->scale);

            if constexpr (Dim == 2) {
                laplace_kernels::laplace_2d_gpu<T>(
                    input, output, nb_grid_pts[0], nb_grid_pts[1], 1,
                    nb_grid_pts[0], effective_scale, increment);
            } else {
                laplace_kernels::laplace_3d_gpu<T>(
                    input, output, nb_grid_pts[0], nb_grid_pts[1],
                    nb_grid_pts[2], 1, nb_grid_pts[0],
                    nb_grid_pts[0] * nb_grid_pts[1], effective_scale, increment);
            }
        }
#endif
    };

    //! 2D Laplace operator (5-point stencil). Preserves the historical name.
    using LaplaceOperator2D = LaplaceOperator<2>;
    //! 3D Laplace operator (7-point stencil). Preserves the historical name.
    using LaplaceOperator3D = LaplaceOperator<3>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_LAPLACE_HH_
