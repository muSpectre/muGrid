/**
 * @file   gradient_operator_default.hh
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

#ifndef SRC_LIBMUGRID_CONVOLUTION_OPERATOR_HH_
#define SRC_LIBMUGRID_CONVOLUTION_OPERATOR_HH_

#include "convolution_operator_base.hh"
#include "kokkos_types.hh"

#include "Eigen/Dense"
#include <Kokkos_Core.hpp>

#include <vector>
#include <optional>

namespace muGrid {

    // Forward declaration
    class GlobalFieldCollection;

    /**
     * @brief Structure-of-Arrays representation of sparse operator for better
     *        cache locality and GPU compatibility.
     *
     * @tparam MemorySpace Kokkos memory space (HostSpace, CudaSpace, etc.)
     */
    template<typename MemorySpace = HostSpace>
    struct SparseOperatorSoA {
        using ViewIndex = Kokkos::View<Index_t*, MemorySpace>;
        using ViewReal = Kokkos::View<Real*, MemorySpace>;

        ViewIndex quad_indices{};   ///< Output indices into quadrature field
        ViewIndex nodal_indices{};  ///< Input indices into nodal field
        ViewReal values{};          ///< Operator values
        Index_t size{0};            ///< Number of non-zero entries

        //! Default constructor
        SparseOperatorSoA() = default;

        //! Constructor with size allocation
        explicit SparseOperatorSoA(Index_t n)
            : quad_indices("quad_indices", n),
              nodal_indices("nodal_indices", n),
              values("values", n),
              size(n) {}

        //! Check if the operator is empty
        bool empty() const { return size == 0; }
    };

    /**
     * @brief Parameters for grid traversal, precomputed for efficiency
     */
    struct GridTraversalParams {
        Index_t nx{1}, ny{1}, nz{1};           ///< Grid dimensions without ghosts
        Index_t total_pixels{0};               ///< Total number of pixels
        Index_t start_pixel_index{0};          ///< Index of first non-ghost pixel
        Index_t nodal_elems_per_pixel{0};      ///< Elements per pixel in nodal field
        Index_t quad_elems_per_pixel{0};       ///< Elements per pixel in quad field
        Index_t nodal_stride_x{0};             ///< Stride in x for nodal field
        Index_t nodal_stride_y{0};             ///< Stride in y for nodal field
        Index_t nodal_stride_z{0};             ///< Stride in z for nodal field
        Index_t quad_stride_x{0};              ///< Stride in x for quad field
        Index_t quad_stride_y{0};              ///< Stride in y for quad field
        Index_t quad_stride_z{0};              ///< Stride in z for quad field
        Index_t row_width{0};                  ///< Full row width including ghosts
    };

    /**
     * @brief Cache key for sparse operator memoization
     */
    struct SparseOperatorCacheKey {
        IntCoord_t nb_grid_pts;
        Index_t nb_nodal_components;

        bool operator==(const SparseOperatorCacheKey& other) const {
            return nb_grid_pts == other.nb_grid_pts &&
                   nb_nodal_components == other.nb_nodal_components;
        }
    };

    /**
     * @class ConvolutionOperator
     * @brief Implements convolution operations that can be applied pixel-wise to
     * the field.
     *
     * This class extends ConvolutionOperatorBase to provide specific implementations
     * for gradient and divergence operations based on the shape function
     * gradients for each quadrature point. It is designed to work with fields
     * defined on nodal points and quadrature points, facilitating the evaluation
     * of gradients and the discretised divergence.
     *
     * The implementation uses Kokkos for portable parallelization across CPU
     * and GPU architectures. The sparse operator representation uses a
     * Structure-of-Arrays (SoA) layout for optimal memory access patterns.
     *
     * @note This class cannot be instantiated directly and does not support copy
     *       construction or copy assignment.
     */
    class ConvolutionOperator : public ConvolutionOperatorBase {
    public:
        using Parent = ConvolutionOperatorBase;

        //! Default constructor is deleted to prevent instantiation.
        ConvolutionOperator() = delete;

        /**
         * @brief Constructs a ConvolutionOperator object. It initializes
         * the convolution operator with the provided pixel-wise operator,
         * and necessary information to indicate its shape.
         *
         * @param pixel_offset Stencil offset in number of pixels
         * @param pixel_operator The pixel-wise operator raveled as an array.
         * @param conv_pts_shape Shape of the stencil.
         * @param nb_pixelnodal_pts Number of nodal points per pixel.
         * @param nb_quad_pts Number of quadrature points per pixel.
         * @param nb_operators Number of operators in the stencil.
         */
        ConvolutionOperator(
            const Shape_t &pixel_offset,
            std::span<const Real> pixel_operator,
            const Shape_t &conv_pts_shape,
            const Index_t &nb_pixelnodal_pts,
            const Index_t &nb_quad_pts,
            const Index_t &nb_operators);

        //! Copy constructor
        ConvolutionOperator(const ConvolutionOperator &other) = delete;

        //! Move constructor
        ConvolutionOperator(ConvolutionOperator &&other) = default;

        //! Destructor
        ~ConvolutionOperator() override = default;

        //! Copy assignment operator
        ConvolutionOperator &
        operator=(const ConvolutionOperator &other) = delete;

        //! Move assignment operator
        ConvolutionOperator &
        operator=(ConvolutionOperator &&other) = default;

        /**
         * Evaluates the gradient of nodal_field into quadrature_point_field
         *
         * @param nodal_field input field of which to take gradient. Defined on
         * nodal points
         * @param quadrature_point_field output field to write gradient into.
         * Defined on quadrature points
         */
        void
        apply(const TypedFieldBase<Real> &nodal_field,
              TypedFieldBase<Real> &quadrature_point_field) const final;

        /**
         * Evaluates the gradient of nodal_field and adds it to
         * quadrature_point_field
         *
         * @param nodal_field input field of which to take gradient. Defined on
         * nodal points
         * @param alpha scaling factor for the increment
         * @param quadrature_point_field output field to increment by the gradient
         * field. Defined on quadrature points
         */
        void apply_increment(
            const TypedFieldBase<Real> &nodal_field, const Real &alpha,
            TypedFieldBase<Real> &quadrature_point_field) const override;

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
        void transpose(const TypedFieldBase<Real> &quadrature_point_field,
                       TypedFieldBase<Real> &nodal_field,
                       const std::vector<Real> &weights = {}) const final;

        /**
         * Evaluates the discretised divergence of quadrature_point_field and adds
         * the result to nodal_field, weights corresponds to Gaussian quadrature
         * weights. If weights are omitted, this returns some scaled version of
         * discretised divergence.
         * @param quadrature_point_field input field of which to take
         * the divergence. Defined on quadrature points.
         * @param alpha scaling factor for the increment
         * @param nodal_field output field to be incremented by the divergence
         * @param weights Gaussian quadrature weights
         */
        void transpose_increment(
            const TypedFieldBase<Real> &quadrature_point_field, const Real &alpha,
            TypedFieldBase<Real> &nodal_field,
            const std::vector<Real> &weights = {}) const final;

        /**
         * Return the operator array linking the nodal degrees of freedom to their
         * quadrature-point values.
         */
        const std::vector<Real> &get_pixel_operator() const;

        /**
         * returns the number of quadrature points are associated with any
         * pixel/voxel (i.e., the sum of the number of quadrature points associated
         * with each element belonging to any pixel/voxel.
         */
        Index_t get_nb_quad_pts() const final;

        /**
         * returns the number of operators
         */
        Index_t get_nb_operators() const final;

        /**
         * returns the number of nodal points associated with any pixel/voxel.
         * (Every node belonging to at least one of the elements belonging to any
         * pixel/voxel, without recounting nodes that appear multiple times)
         */
        Index_t get_nb_nodal_pts() const final;

        /**
         * return the spatial dimension of this gradient operator
         */
        Index_t get_spatial_dim() const final;

        /**
         * Clear the cached sparse operator (useful if memory is a concern)
         */
        void clear_cache() const;

    private:
        /**
         * stencil offset in number of pixels
         */
        Shape_t pixel_offset{};
        /**
         * array linking the nodal degrees of freedom to their quadrature-point
         * values.
         */
        std::vector<Real> pixel_operator{};
        /**
         * number of convolution points, i.e., number of nodal points that is
         * involved in the convolution of one pixel.
         */
        Shape_t conv_pts_shape;
        /**
         * number of pixel nodal points. When the grid gets complicated,
         * it shall be divided into sub-grids, where each of them is a
         * regular grid. Hence the name "pixel nodal".
         */
        Index_t nb_pixelnodal_pts;
        /**
         * number of quadrature points per pixel (e.g.. 4 for linear
         * quadrilateral)
         */
        Index_t nb_quad_pts;
        /**
         * number of nodal points that is involved in the convolution of this pixel.
         */
        Index_t nb_operators;
        /**
         * the spatial dimension & number of the nodal points involved in the
         * convolution of one pixel.
         */
        Index_t spatial_dim;
        Index_t nb_conv_pts;

        //! Tolerance for considering operator values as zero
        static constexpr Real zero_tolerance = 1e-14;

        //! Cached sparse operators for reuse (two orderings for optimal access)
        //! Row-major: groups by quad index (optimal for apply - write locality)
        mutable std::optional<SparseOperatorSoA<HostSpace>> cached_apply_op{};
        //! Column-major: groups by nodal index (optimal for transpose - write locality)
        mutable std::optional<SparseOperatorSoA<HostSpace>> cached_transpose_op{};
        mutable std::optional<SparseOperatorCacheKey> cached_key{};

        //! Device-space cached operators (lazily copied from host)
        mutable std::optional<SparseOperatorSoA<DefaultDeviceSpace>> cached_device_apply_op{};
        mutable std::optional<SparseOperatorSoA<DefaultDeviceSpace>> cached_device_transpose_op{};

        /**
         * @brief Validate that fields are compatible with this operator
         * @param nodal_field The nodal field
         * @param quad_field The quadrature point field
         * @return Reference to the GlobalFieldCollection
         * @throws RuntimeError if validation fails
         */
        const GlobalFieldCollection& validate_fields(
            const TypedFieldBase<Real> &nodal_field,
            const TypedFieldBase<Real> &quad_field) const;

        /**
         * @brief Get or create sparse operator for apply operation
         * @param nb_grid_pts number of process-local (subdomain) grid points
         * with ghosts
         * @param nb_nodal_components number of components in nodal field
         * @return Reference to the sparse operator (cached, row-major order)
         */
        const SparseOperatorSoA<HostSpace>&
        get_apply_operator(const IntCoord_t & nb_grid_pts,
                           const Index_t nb_nodal_components) const;

        /**
         * @brief Get or create sparse operator for transpose operation
         * @param nb_grid_pts number of process-local (subdomain) grid points
         * with ghosts
         * @param nb_nodal_components number of components in nodal field
         * @return Reference to the sparse operator (cached, column-major order)
         */
        const SparseOperatorSoA<HostSpace>&
        get_transpose_operator(const IntCoord_t & nb_grid_pts,
                               const Index_t nb_nodal_components) const;

        /**
         * @brief Create sparse operator with row-major ordering
         * @param nb_grid_pts number of process-local (subdomain) grid points
         * with ghosts
         * @param nb_nodal_components number of components in nodal field
         * @return The sparse operator in SoA format (row-major order)
         *
         * Row-major order groups entries by quad index, providing write
         * locality for apply_increment (scatter to quad_data).
         */
        SparseOperatorSoA<HostSpace>
        create_apply_operator(const IntCoord_t & nb_grid_pts,
                              const Index_t nb_nodal_components) const;

        /**
         * @brief Create sparse operator with column-major ordering
         * @param nb_grid_pts number of process-local (subdomain) grid points
         * with ghosts
         * @param nb_nodal_components number of components in nodal field
         * @return The sparse operator in SoA format (column-major order)
         *
         * Column-major order groups entries by nodal index, providing write
         * locality for transpose_increment (scatter to nodal_data).
         */
        SparseOperatorSoA<HostSpace>
        create_transpose_operator(const IntCoord_t & nb_grid_pts,
                                  const Index_t nb_nodal_components) const;

        /**
         * @brief Compute grid traversal parameters
         * @param collection The field collection
         * @param nb_nodal_components Number of nodal components
         * @param nb_quad_components Number of quadrature components
         * @return GridTraversalParams structure with all computed values
         */
        GridTraversalParams compute_traversal_params(
            const GlobalFieldCollection& collection,
            Index_t nb_nodal_components,
            Index_t nb_quad_components) const;

        /**
         * Apply convolution on device memory fields.
         * Data must already be in device memory space.
         *
         * @tparam DeviceSpace Target device memory space (CudaSpace, HIPSpace)
         * @param nodal_data Pointer to nodal field data in device memory
         * @param quad_data Pointer to quadrature field data in device memory
         * @param alpha Scaling factor
         * @param params Grid traversal parameters
         */
        template<typename DeviceSpace>
        void apply_on_device(
            const Real* nodal_data,
            Real* quad_data,
            const Real alpha,
            const GridTraversalParams& params) const;

        /**
         * Transpose convolution on device memory fields.
         * Data must already be in device memory space.
         *
         * @tparam DeviceSpace Target device memory space (CudaSpace, HIPSpace)
         * @param quad_data Pointer to quadrature field data in device memory
         * @param nodal_data Pointer to nodal field data in device memory
         * @param alpha Scaling factor
         * @param params Grid traversal parameters
         */
        template<typename DeviceSpace>
        void transpose_on_device(
            const Real* quad_data,
            Real* nodal_data,
            const Real alpha,
            const GridTraversalParams& params) const;

        /**
         * Get or create device-space sparse operator for apply operation.
         * Lazily copies from host cache to device.
         *
         * @tparam DeviceSpace Target device memory space
         * @param nb_grid_pts Grid points with ghosts
         * @param nb_nodal_components Number of nodal components
         * @return Reference to device sparse operator
         */
        template<typename DeviceSpace>
        const SparseOperatorSoA<DeviceSpace>&
        get_device_apply_operator(const IntCoord_t& nb_grid_pts,
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
        template<typename DeviceSpace>
        const SparseOperatorSoA<DeviceSpace>&
        get_device_transpose_operator(const IntCoord_t& nb_grid_pts,
                                      Index_t nb_nodal_components) const;
    };

    /**
     * @brief Apply convolution kernel - templated on execution space for GPU portability
     *
     * @tparam ExecutionSpace Kokkos execution space (Serial, OpenMP, Cuda, etc.)
     * @param nodal_data Pointer to nodal field data
     * @param quad_data Pointer to quadrature field data (output)
     * @param alpha Scaling factor
     * @param params Grid traversal parameters
     * @param sparse_op Sparse operator (must be in compatible memory space)
     */
    template<typename ExecutionSpace, typename MemorySpace>
    void apply_convolution_kernel(
        const Real* nodal_data,
        Real* quad_data,
        const Real alpha,
        const GridTraversalParams& params,
        const SparseOperatorSoA<MemorySpace>& sparse_op) {
        const Index_t nx = params.nx;
        const Index_t ny = params.ny;
        const Index_t nz = params.nz;
        const Index_t nodal_base = params.start_pixel_index *
                                   params.nodal_elems_per_pixel;
        const Index_t quad_base = params.start_pixel_index *
                                  params.quad_elems_per_pixel;
        const Index_t nodal_stride_x = params.nodal_stride_x;
        const Index_t nodal_stride_y = params.nodal_stride_y;
        const Index_t nodal_stride_z = params.nodal_stride_z;
        const Index_t quad_stride_x = params.quad_stride_x;
        const Index_t quad_stride_y = params.quad_stride_y;
        const Index_t quad_stride_z = params.quad_stride_z;
        const Index_t nnz = sparse_op.size;

        // Get raw pointers for kernel access
        const Index_t* quad_indices = sparse_op.quad_indices.data();
        const Index_t* nodal_indices = sparse_op.nodal_indices.data();
        const Real* op_values = sparse_op.values.data();

        using MDPolicy = Kokkos::MDRangePolicy<
            ExecutionSpace,
            Kokkos::Rank<3>,
            Kokkos::IndexType<Index_t>>;

        Kokkos::parallel_for(
            "ConvolutionOperator::apply_kernel",
            MDPolicy({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(const Index_t z, const Index_t y, const Index_t x) {
                const Index_t nodal_offset = nodal_base +
                    z * nodal_stride_z + y * nodal_stride_y + x * nodal_stride_x;
                const Index_t quad_offset = quad_base +
                    z * quad_stride_z + y * quad_stride_y + x * quad_stride_x;

                for (Index_t i = 0; i < nnz; ++i) {
                    quad_data[quad_offset + quad_indices[i]] +=
                        alpha * nodal_data[nodal_offset + nodal_indices[i]] *
                        op_values[i];
                }
            });
    }

    /**
     * @brief Transpose convolution kernel - templated on execution space for GPU portability
     *
     * @tparam ExecutionSpace Kokkos execution space (Serial, OpenMP, Cuda, etc.)
     * @param quad_data Pointer to quadrature field data (input)
     * @param nodal_data Pointer to nodal field data (output)
     * @param alpha Scaling factor
     * @param params Grid traversal parameters
     * @param sparse_op Sparse operator (must be in compatible memory space)
     *
     * @note Uses atomic operations for thread-safe accumulation since multiple
     *       pixels may write to the same nodal DOF (overlapping stencils).
     */
    template<typename ExecutionSpace, typename MemorySpace>
    void transpose_convolution_kernel(
        const Real* quad_data,
        Real* nodal_data,
        const Real alpha,
        const GridTraversalParams& params,
        const SparseOperatorSoA<MemorySpace>& sparse_op) {
        const Index_t nx = params.nx;
        const Index_t ny = params.ny;
        const Index_t nz = params.nz;
        const Index_t nodal_base = params.start_pixel_index *
                                   params.nodal_elems_per_pixel;
        const Index_t quad_base = params.start_pixel_index *
                                  params.quad_elems_per_pixel;
        const Index_t nodal_stride_x = params.nodal_stride_x;
        const Index_t nodal_stride_y = params.nodal_stride_y;
        const Index_t nodal_stride_z = params.nodal_stride_z;
        const Index_t quad_stride_x = params.quad_stride_x;
        const Index_t quad_stride_y = params.quad_stride_y;
        const Index_t quad_stride_z = params.quad_stride_z;
        const Index_t nnz = sparse_op.size;

        // Get raw pointers for kernel access
        const Index_t* quad_indices = sparse_op.quad_indices.data();
        const Index_t* nodal_indices = sparse_op.nodal_indices.data();
        const Real* op_values = sparse_op.values.data();

        using MDPolicy = Kokkos::MDRangePolicy<
            ExecutionSpace,
            Kokkos::Rank<3>,
            Kokkos::IndexType<Index_t>>;

        Kokkos::parallel_for(
            "ConvolutionOperator::transpose_kernel",
            MDPolicy({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(const Index_t z, const Index_t y, const Index_t x) {
                const Index_t nodal_offset = nodal_base +
                    z * nodal_stride_z + y * nodal_stride_y + x * nodal_stride_x;
                const Index_t quad_offset = quad_base +
                    z * quad_stride_z + y * quad_stride_y + x * quad_stride_x;

                for (Index_t i = 0; i < nnz; ++i) {
                    Kokkos::atomic_add(
                        &nodal_data[nodal_offset + nodal_indices[i]],
                        alpha * quad_data[quad_offset + quad_indices[i]] *
                        op_values[i]);
                }
            });
    }

    /**
     * @brief Deep copy sparse operator between memory spaces
     *
     * @tparam DstSpace Destination memory space
     * @tparam SrcSpace Source memory space
     * @param src Source sparse operator
     * @return Copy in destination memory space
     */
    template<typename DstSpace, typename SrcSpace>
    SparseOperatorSoA<DstSpace> deep_copy_sparse_operator(
        const SparseOperatorSoA<SrcSpace>& src) {
        SparseOperatorSoA<DstSpace> dst(src.size);
        Kokkos::deep_copy(dst.quad_indices, src.quad_indices);
        Kokkos::deep_copy(dst.nodal_indices, src.nodal_indices);
        Kokkos::deep_copy(dst.values, src.values);
        return dst;
    }

    /**
     * @brief Pad a shape vector to 3D by appending fill_value
     * @param shape Input shape (can be 1D, 2D, or 3D)
     * @param fill_value Value to use for padding (default: 1)
     * @return 3D shape vector
     */
    inline Shape_t pad_shape_to_3d(const Shape_t& shape, Index_t fill_value = 1) {
        Shape_t result = shape;
        while (result.size() < 3) {
            result.push_back(fill_value);
        }
        return result;
    }

    /**
     * @brief Pad a DynCcoord to 3D Shape_t by appending fill_value
     * @param coord Input coordinate (can be 1D, 2D, or 3D)
     * @param fill_value Value to use for padding (default: 1)
     * @return 3D shape vector
     */
    template<size_t MaxDim, typename T>
    inline Shape_t pad_shape_to_3d(const DynCcoord<MaxDim, T>& coord,
                                   Index_t fill_value = 1) {
        Shape_t result(coord.begin(), coord.end());
        while (result.size() < 3) {
            result.push_back(fill_value);
        }
        return result;
    }

    /* ---------------------------------------------------------------------- */
    /* Template method implementations for ConvolutionOperator                */
    /* ---------------------------------------------------------------------- */

    template<typename DeviceSpace>
    void ConvolutionOperator::apply_on_device(
        const Real* nodal_data,
        Real* quad_data,
        const Real alpha,
        const GridTraversalParams& params) const {
        // Get execution space type corresponding to memory space
        using ExecutionSpace = typename DeviceSpace::execution_space;

        // Get device sparse operator (lazily copies from host if needed)
        // Note: For non-default device spaces, this would need specialization
        const auto& sparse_op = this->cached_device_apply_op.value();

        apply_convolution_kernel<ExecutionSpace, DeviceSpace>(
            nodal_data, quad_data, alpha, params, sparse_op);
    }

    template<typename DeviceSpace>
    void ConvolutionOperator::transpose_on_device(
        const Real* quad_data,
        Real* nodal_data,
        const Real alpha,
        const GridTraversalParams& params) const {
        // Get execution space type corresponding to memory space
        using ExecutionSpace = typename DeviceSpace::execution_space;

        // Get device sparse operator (lazily copies from host if needed)
        const auto& sparse_op = this->cached_device_transpose_op.value();

        transpose_convolution_kernel<ExecutionSpace, DeviceSpace>(
            quad_data, nodal_data, alpha, params, sparse_op);
    }

    template<typename DeviceSpace>
    const SparseOperatorSoA<DeviceSpace>&
    ConvolutionOperator::get_device_apply_operator(
        const IntCoord_t& nb_grid_pts,
        Index_t nb_nodal_components) const {
        // Ensure host operator is cached first
        this->get_apply_operator(nb_grid_pts, nb_nodal_components);

        // Check if device cache needs update
        if (!this->cached_device_apply_op.has_value()) {
            // Copy from host to device
            this->cached_device_apply_op = deep_copy_sparse_operator<
                DeviceSpace, HostSpace>(this->cached_apply_op.value());
        }

        return this->cached_device_apply_op.value();
    }

    template<typename DeviceSpace>
    const SparseOperatorSoA<DeviceSpace>&
    ConvolutionOperator::get_device_transpose_operator(
        const IntCoord_t& nb_grid_pts,
        Index_t nb_nodal_components) const {
        // Ensure host operator is cached first
        this->get_transpose_operator(nb_grid_pts, nb_nodal_components);

        // Check if device cache needs update
        if (!this->cached_device_transpose_op.has_value()) {
            // Copy from host to device
            this->cached_device_transpose_op = deep_copy_sparse_operator<
                DeviceSpace, HostSpace>(this->cached_transpose_op.value());
        }

        return this->cached_device_transpose_op.value();
    }

} // namespace muGrid
#endif  // SRC_LIBMUGRID_CONVOLUTION_OPERATOR_HH_
