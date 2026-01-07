/**
 * @file   convolution_operator.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   16 Jun 2020
 *
 * @brief  D operator based on the shape function gradients for each quadrature
 * point
 *
 * Copyright © 2020 Till Junge, Martin Ladecký
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

#ifndef SRC_LIBMUGRID_GENERIC_HH_
#define SRC_LIBMUGRID_GENERIC_HH_

#include "core/coordinates.hh"
#include "operators/linear.hh"
#include "operators/kernel_dispatcher.hh"

#include "Eigen/Dense"

#include <vector>
#include <optional>
#include <span>

namespace muGrid {

    // Forward declaration
    class GlobalFieldCollection;

    // SparseOperatorSoA and GridTraversalParams are defined in
    // memory/kernels_cpu.hh

    /**
     * @brief Cache key for sparse operator memoization
     */
    struct SparseOperatorCacheKey {
        DynGridIndex nb_grid_pts;
        Index_t nb_nodal_components;

        bool operator==(const SparseOperatorCacheKey & other) const {
            return nb_grid_pts == other.nb_grid_pts &&
                   nb_nodal_components == other.nb_nodal_components;
        }
    };

    /**
     * @class GenericLinearOperator
     * @brief Implements gradient operations using a stencil-based approach.
     *
     * This class extends GradientOperator to provide specific implementations
     * for gradient and divergence operations based on the shape function
     * gradients for each quadrature point. It is designed to work with fields
     * defined on nodal points and quadrature points, facilitating the
     * evaluation of gradients and the discretised divergence.
     *
     * The implementation supports portable execution across CPU and GPU
     * (CUDA/HIP) architectures through backend-specific kernel implementations.
     * The sparse operator representation uses a Structure-of-Arrays (SoA)
     * layout for optimal memory access patterns on both CPU and GPU.
     *
     * @note This class cannot be instantiated directly and does not support
     * copy construction or copy assignment.
     */
    class GenericLinearOperator : public LinearOperator {
       public:
        using Parent = LinearOperator;

        //! Default constructor is deleted to prevent instantiation.
        GenericLinearOperator() = delete;

        /**
         * @brief Constructs a GenericLinearOperator object. It initializes
         * the gradient operator with the provided stencil coefficients,
         * and necessary information to indicate its shape.
         *
         * @param offset Stencil offset in number of pixels
         * @param coefficients The stencil coefficients raveled as an array.
         * @param stencil_shape Shape of the stencil.
         * @param nb_pixel_input_components Number of input components per
         * pixel.
         * @param nb_quad_pts Number of quadrature points per pixel.
         * @param nb_output_components Number of output components.
         */
        GenericLinearOperator(const Shape_t & offset,
                              std::span<const Real> coefficients,
                              const Shape_t & stencil_shape,
                              const Index_t & nb_pixel_input_components,
                              const Index_t & nb_quad_pts,
                              const Index_t & nb_output_components);

        //! Copy constructor
        GenericLinearOperator(const GenericLinearOperator & other) = delete;

        //! Move constructor
        GenericLinearOperator(GenericLinearOperator && other) = default;

        //! Destructor
        ~GenericLinearOperator() override = default;

        //! Copy assignment operator
        GenericLinearOperator &
        operator=(const GenericLinearOperator & other) = delete;

        //! Move assignment operator
        GenericLinearOperator &
        operator=(GenericLinearOperator && other) = default;

        /**
         * Evaluates the gradient of nodal_field into quadrature_point_field
         *
         * @param nodal_field input field of which to take gradient. Defined on
         * nodal points
         * @param quadrature_point_field output field to write gradient into.
         * Defined on quadrature points
         */
        void apply(const TypedFieldBase<Real> & nodal_field,
                   TypedFieldBase<Real> & quadrature_point_field) const final;

        /**
         * Evaluates the gradient of nodal_field and adds it to
         * quadrature_point_field
         *
         * @param nodal_field input field of which to take gradient. Defined on
         * nodal points
         * @param alpha scaling factor for the increment
         * @param quadrature_point_field output field to increment by the
         * gradient field. Defined on quadrature points
         */
        void apply_increment(
            const TypedFieldBase<Real> & nodal_field, const Real & alpha,
            TypedFieldBase<Real> & quadrature_point_field) const override;

        /**
         * Evaluates the discretised divergence of quadrature_point_field into
         * nodal_field, weights corresponds to Gaussian quadrature weights. If
         * weights are omitted, this returns some scaled version of discretised
         * divergence.
         * @param quadrature_point_field input field of which to take
         * the divergence. Defined on quadrature points.
         * @param nodal_field output field into which divergence is written
         * @param weights Gaussian quadrature weights
         */
        void transpose(const TypedFieldBase<Real> & quadrature_point_field,
                       TypedFieldBase<Real> & nodal_field,
                       const std::vector<Real> & weights = {}) const final;

        /**
         * Evaluates the discretised divergence of quadrature_point_field and
         * adds the result to nodal_field, weights corresponds to Gaussian
         * quadrature weights. If weights are omitted, this returns some scaled
         * version of discretised divergence.
         * @param quadrature_point_field input field of which to take
         * the divergence. Defined on quadrature points.
         * @param alpha scaling factor for the increment
         * @param nodal_field output field to be incremented by the divergence
         * @param weights Gaussian quadrature weights
         */
        void
        transpose_increment(const TypedFieldBase<Real> & quadrature_point_field,
                            const Real & alpha,
                            TypedFieldBase<Real> & nodal_field,
                            const std::vector<Real> & weights = {}) const final;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        /**
         * Evaluates the gradient of nodal_field into quadrature_point_field
         * on device (GPU) memory.
         *
         * @param nodal_field input field in device memory
         * @param quadrature_point_field output field in device memory
         */
        void apply(const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
                   TypedFieldBase<Real, DefaultDeviceSpace> &
                       quadrature_point_field) const;

        /**
         * Evaluates the gradient of nodal_field and adds it to
         * quadrature_point_field on device (GPU) memory.
         *
         * @param nodal_field input field in device memory
         * @param alpha scaling factor for the increment
         * @param quadrature_point_field output field in device memory
         */
        void apply_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & quadrature_point_field)
            const;

        /**
         * Evaluates the discretised divergence of quadrature_point_field into
         * nodal_field on device (GPU) memory.
         *
         * @param quadrature_point_field input field in device memory
         * @param nodal_field output field in device memory
         * @param weights Gaussian quadrature weights (currently ignored for
         * device)
         */
        void transpose(const TypedFieldBase<Real, DefaultDeviceSpace> &
                           quadrature_point_field,
                       TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
                       const std::vector<Real> & weights = {}) const;

        /**
         * Evaluates the discretised divergence of quadrature_point_field and
         * adds the result to nodal_field on device (GPU) memory.
         *
         * @param quadrature_point_field input field in device memory
         * @param alpha scaling factor for the increment
         * @param nodal_field output field in device memory
         * @param weights Gaussian quadrature weights (currently ignored for
         * device)
         */
        void transpose_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> &
                quadrature_point_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights = {}) const;
#endif

        /**
         * Return the stencil coefficients.
         */
        const std::vector<Real> & get_coefficients() const;

        /**
         * Return the stencil offset in number of pixels.
         */
        const Shape_t & get_offset() const;

        /**
         * Return the shape of the stencil.
         */
        const Shape_t & get_stencil_shape() const;

        /**
         * returns the number of quadrature points are associated with any
         * pixel/voxel (i.e., the sum of the number of quadrature points
         * associated with each element belonging to any pixel/voxel.
         */
        Index_t get_nb_quad_pts() const final;

        /**
         * returns the number of output components
         */
        Index_t get_nb_output_components() const final;

        /**
         * returns the number of input components associated with any
         * pixel/voxel. (Every node belonging to at least one of the elements
         * belonging to any pixel/voxel, without recounting nodes that appear
         * multiple times)
         */
        Index_t get_nb_input_components() const final;

        /**
         * return the spatial dimension of this gradient operator
         */
        Dim_t get_spatial_dim() const final;

        /**
         * Clear the cached sparse operator (useful if memory is a concern)
         */
        void clear_cache() const;

        /**
         * Compute the Fourier representation of this stencil operator
         *
         * Converts a translationally invariant linear combination of grid
         * values into a multiplication with a complex number in Fourier space.
         *
         * @param phase The wavevector times cell dimension (lacking factor of
         * 2π)
         * @return Complex Fourier representation of the operator
         */
        Complex fourier(const Eigen::VectorXd & phase) const;

       private:
        /**
         * stencil offset in number of pixels
         */
        Shape_t offset_{};
        /**
         * Stencil coefficients linking the nodal degrees of freedom to their
         * quadrature-point values.
         */
        std::vector<Real> coefficients_{};
        /**
         * Shape of the stencil (number of pixels in each dimension).
         */
        Shape_t stencil_shape_;
        /**
         * number of input components per pixel. When the grid gets complicated,
         * it shall be divided into sub-grids, where each of them is a
         * regular grid.
         */
        Index_t nb_pixel_input_components_;
        /**
         * number of quadrature points per pixel (e.g.. 4 for linear
         * quadrilateral)
         */
        Index_t nb_quad_pts_;
        /**
         * number of output components (e.g., spatial dimension for gradients)
         */
        Index_t nb_output_components_;
        /**
         * the spatial dimension & number of stencil points
         */
        Dim_t spatial_dim_;
        Index_t nb_stencil_pts_;

        //! Tolerance for considering operator values as zero
        static constexpr Real zero_tolerance = 1e-14;

        //! Cached sparse operators for reuse (two orderings for optimal access)
        //! Row-major: groups by quad index (optimal for apply - write locality)
        mutable std::optional<SparseOperatorSoA<HostSpace>> cached_apply_op_{};
        //! Column-major: groups by nodal index (optimal for transpose - write
        //! locality)
        mutable std::optional<SparseOperatorSoA<HostSpace>>
            cached_transpose_op_{};
        mutable std::optional<SparseOperatorCacheKey> cached_key_{};

        //! Device-space cached operators (lazily copied from host)
        mutable std::optional<SparseOperatorSoA<DefaultDeviceSpace>>
            cached_device_apply_op_{};
        mutable std::optional<SparseOperatorSoA<DefaultDeviceSpace>>
            cached_device_transpose_op_{};

        /**
         * @brief Validate that fields are compatible with this operator
         * @param nodal_field The nodal field
         * @param quad_field The quadrature point field
         * @param is_transpose If true, swap ghost requirements (transpose reads
         *        at negative offsets, so left/right ghost needs are flipped)
         * @return Reference to the GlobalFieldCollection
         * @throws RuntimeError if validation fails
         */
        const GlobalFieldCollection &
        validate_fields(const TypedFieldBase<Real> & nodal_field,
                        const TypedFieldBase<Real> & quad_field,
                        bool is_transpose = false) const;

        /**
         * @brief Validate that fields are compatible with this operator
         * (generic version)
         *
         * This version accepts base Field references and works for both host
         * and device-space fields. It performs the same validation as the
         * typed version but without requiring a specific memory space.
         *
         * @param nodal_field The nodal field (base class reference)
         * @param quad_field The quadrature point field (base class reference)
         * @param is_transpose If true, swap ghost requirements
         * @return Reference to the GlobalFieldCollection
         * @throws RuntimeError if validation fails
         */
        const GlobalFieldCollection &
        validate_fields_generic(const Field & nodal_field,
                                const Field & quad_field,
                                bool is_transpose = false) const;

        /**
         * @brief Get or create sparse operator for apply operation
         * @param nb_grid_pts number of process-local (subdomain) grid points
         * with ghosts
         * @param nb_nodal_components number of components in nodal field
         * @return Reference to the sparse operator (cached, row-major order)
         */
        const SparseOperatorSoA<HostSpace> &
        get_apply_operator(const DynGridIndex & nb_grid_pts,
                           const Index_t nb_nodal_components) const;

        /**
         * @brief Get or create sparse operator for transpose operation
         * @param nb_grid_pts number of process-local (subdomain) grid points
         * with ghosts
         * @param nb_nodal_components number of components in nodal field
         * @return Reference to the sparse operator (cached, column-major order)
         */
        const SparseOperatorSoA<HostSpace> &
        get_transpose_operator(const DynGridIndex & nb_grid_pts,
                               const Index_t nb_nodal_components) const;

        /**
         * @brief Create sparse operator with row-major ordering
         * @tparam storage_order Storage order for index computation
         * @param nb_grid_pts number of process-local (subdomain) grid points
         * with ghosts
         * @param nb_nodal_components number of components in nodal field
         * @return The sparse operator in SoA format (row-major order)
         *
         * Row-major order groups entries by quad index, providing write
         * locality for apply_increment (scatter to quad_data).
         *
         * The storage_order template parameter determines how indices are
         * computed:
         * - ArrayOfStructures: indices = (pixel_offset * pts + pt) * comps +
         * comp
         * - StructureOfArrays: indices = comp * total_elements + pixel_offset *
         * pts + pt
         */
        template <StorageOrder storage_order>
        SparseOperatorSoA<HostSpace>
        create_apply_operator(const DynGridIndex & nb_grid_pts,
                              const Index_t nb_nodal_components) const;

        /**
         * @brief Create sparse operator with column-major ordering
         * @tparam storage_order Storage order for index computation
         * @param nb_grid_pts number of process-local (subdomain) grid points
         * with ghosts
         * @param nb_nodal_components number of components in nodal field
         * @return The sparse operator in SoA format (column-major order)
         *
         * Column-major order groups entries by nodal index, providing write
         * locality for transpose_increment (scatter to nodal_data).
         *
         * The storage_order template parameter determines how indices are
         * computed:
         * - ArrayOfStructures: indices = (pixel_offset * pts + pt) * comps +
         * comp
         * - StructureOfArrays: indices = comp * total_elements + pixel_offset *
         * pts + pt
         */
        template <StorageOrder storage_order>
        SparseOperatorSoA<HostSpace>
        create_transpose_operator(const DynGridIndex & nb_grid_pts,
                                  const Index_t nb_nodal_components) const;

        /**
         * @brief Compute grid traversal parameters
         * @tparam storage_order Storage order for stride computation
         * @param collection The field collection
         * @param nb_nodal_components Number of nodal components
         * @param nb_quad_components Number of quadrature components
         * @param is_transpose If true, use transpose stencil requirements
         *        (left/right swapped)
         * @return GridTraversalParams structure with all computed values
         *
         * The computable region is determined by stencil requirements, not
         * ghost size. This allows computing results in ghost regions where
         * the stencil has valid input data.
         *
         * For ArrayOfStructures: stride_x = elems_per_pixel
         * For StructureOfArrays: stride_x = 1 (pixels consecutive per
         * component)
         */
        template <StorageOrder storage_order>
        GridTraversalParams
        compute_traversal_params(const GlobalFieldCollection & collection,
                                 Index_t nb_nodal_components,
                                 Index_t nb_quad_components,
                                 bool is_transpose) const;

        /**
         * Apply convolution on device memory fields.
         * Data must already be in device memory space.
         *
         * @tparam DeviceSpace Target device memory space (CUDASpace, ROCmSpace)
         * @param nodal_data Pointer to nodal field data in device memory
         * @param quad_data Pointer to quadrature field data in device memory
         * @param alpha Scaling factor
         * @param params Grid traversal parameters
         */
        template <typename DeviceSpace>
        void apply_on_device(const Real * nodal_data, Real * quad_data,
                             const Real alpha,
                             const GridTraversalParams & params) const;

        /**
         * Transpose convolution on device memory fields.
         * Data must already be in device memory space.
         *
         * @tparam DeviceSpace Target device memory space (CUDASpace, ROCmSpace)
         * @param quad_data Pointer to quadrature field data in device memory
         * @param nodal_data Pointer to nodal field data in device memory
         * @param alpha Scaling factor
         * @param params Grid traversal parameters
         */
        template <typename DeviceSpace>
        void transpose_on_device(const Real * quad_data, Real * nodal_data,
                                 const Real alpha,
                                 const GridTraversalParams & params) const;

        /**
         * Get or create device-space sparse operator for apply operation.
         * Lazily copies from host cache to device.
         *
         * @tparam DeviceSpace Target device memory space
         * @param nb_grid_pts Grid points with ghosts
         * @param nb_nodal_components Number of nodal components
         * @return Reference to device sparse operator
         */
        template <typename DeviceSpace>
        const SparseOperatorSoA<DeviceSpace> &
        get_device_apply_operator(const DynGridIndex & nb_grid_pts,
                                  Index_t nb_nodal_components) const;

        /**
         * Get or create device-space sparse operator for transpose operation.
         * Lazily copies from host cache to device.
         *
         * @tparam DeviceSpace Target device memory space
         * @param nb_grid_pts Grid points with ghosts
         * @param nb_nodal_components Number of nodal components
         * @return Reference to device sparse operator
         */
        template <typename DeviceSpace>
        const SparseOperatorSoA<DeviceSpace> &
        get_device_transpose_operator(const DynGridIndex & nb_grid_pts,
                                      Index_t nb_nodal_components) const;
    };

    // Kernel functions and deep_copy_sparse_operator are now in
    // operators/convolution_kernels.hh

    /**
     * @brief Pad a shape vector to 3D by appending fill_value
     * @param shape Input shape (can be 1D, 2D, or 3D)
     * @param fill_value Value to use for padding (default: 1)
     * @return 3D shape vector
     */
    inline Shape_t pad_shape_to_3d(const Shape_t & shape,
                                   Index_t fill_value = 1) {
        Shape_t result = shape;
        while (result.size() < 3) {
            result.push_back(fill_value);
        }
        return result;
    }

    /**
     * @brief Pad a DynCoord to 3D Shape_t by appending fill_value
     * @param coord Input coordinate (can be 1D, 2D, or 3D)
     * @param fill_value Value to use for padding (default: 1)
     * @return 3D shape vector
     */
    template <size_t MaxDim, typename T>
    inline Shape_t pad_shape_to_3d(const DynCoord<MaxDim, T> & coord,
                                   Index_t fill_value = 1) {
        Shape_t result(coord.begin(), coord.end());
        while (result.size() < 3) {
            result.push_back(fill_value);
        }
        return result;
    }

    /* ---------------------------------------------------------------------- */
    /* Template method implementations for GenericLinearOperator            */
    /* ---------------------------------------------------------------------- */

    template <typename DeviceSpace>
    void GenericLinearOperator::apply_on_device(
        const Real * nodal_data, Real * quad_data, const Real alpha,
        const GridTraversalParams & params) const {
        // Get device sparse operator (lazily copies from host if needed)
        const auto & sparse_op = this->cached_device_apply_op_.value();

        // Use the KernelDispatcher for backend-agnostic kernel execution
        KernelDispatcher<DeviceSpace>::apply_convolution(
            nodal_data, quad_data, alpha, params, sparse_op);
    }

    template <typename DeviceSpace>
    void GenericLinearOperator::transpose_on_device(
        const Real * quad_data, Real * nodal_data, const Real alpha,
        const GridTraversalParams & params) const {
        // Get device sparse operator (lazily copies from host if needed)
        const auto & sparse_op = this->cached_device_transpose_op_.value();

        // Use the KernelDispatcher for backend-agnostic kernel execution
        KernelDispatcher<DeviceSpace>::transpose_convolution(
            quad_data, nodal_data, alpha, params, sparse_op);
    }

    template <typename DeviceSpace>
    const SparseOperatorSoA<DeviceSpace> &
    GenericLinearOperator::get_device_apply_operator(
        const DynGridIndex & nb_grid_pts, Index_t nb_nodal_components) const {
        // Check if device cache needs update
        if (!this->cached_device_apply_op_.has_value()) {
            // Create operator with SoA indices (device storage order) and copy
            // to device Device spaces use StructureOfArrays for optimal memory
            // coalescence
            auto host_soa_op =
                this->create_apply_operator<DeviceSpace::storage_order>(
                    nb_grid_pts, nb_nodal_components);
            this->cached_device_apply_op_ =
                muGrid::deep_copy_sparse_operator<DeviceSpace, HostSpace>(
                    host_soa_op);
        }

        return this->cached_device_apply_op_.value();
    }

    template <typename DeviceSpace>
    const SparseOperatorSoA<DeviceSpace> &
    GenericLinearOperator::get_device_transpose_operator(
        const DynGridIndex & nb_grid_pts, Index_t nb_nodal_components) const {
        // Check if device cache needs update
        if (!this->cached_device_transpose_op_.has_value()) {
            // Create operator with SoA indices (device storage order) and copy
            // to device Device spaces use StructureOfArrays for optimal memory
            // coalescence
            auto host_soa_op =
                this->create_transpose_operator<DeviceSpace::storage_order>(
                    nb_grid_pts, nb_nodal_components);
            this->cached_device_transpose_op_ =
                muGrid::deep_copy_sparse_operator<DeviceSpace, HostSpace>(
                    host_soa_op);
        }

        return this->cached_device_transpose_op_.value();
    }

}  // namespace muGrid
#endif  // SRC_LIBMUGRID_GENERIC_HH_
