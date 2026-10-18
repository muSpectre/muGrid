/**
 * @file   generic.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   16 Jun 2020
 *
 * @brief  Generic stencil (convolution) operator mapping nodal values to
 * quadrature-point values, e.g. shape-function gradients
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

#include <algorithm>
#include <cmath>
#include <vector>
#include <optional>
#include <span>

namespace muGrid {

    // Forward declaration
    class GlobalFieldCollection;

    // SparseOperatorSoA and GridTraversalParams are defined in
    // operators/convolution_kernels_cpu.hh

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
     * This class extends LinearOperator with an arbitrary, user-supplied
     * stencil, e.g. the shape-function gradients evaluated at each
     * quadrature point. For a nodal field u with n_c components (each
     * component is transformed independently) it computes
     *
     *     apply:     y_{p,q,o,c} = sum_s sum_i B_{o,q,i}(s) u_{p+s,i,c}
     *     transpose: u_{p,i,c}   = sum_s sum_{q,o} w_q B_{o,q,i}(s)
     *                                              y_{p-s,q,o,c}
     *
     * where p is a pixel, s an offset in the stencil box
     * [offset, offset + stencil_shape - 1], i a nodal sub-point (input
     * component), q a quadrature point, o an output component and w_q the
     * optional quadrature weights (w_q = 1 when none are given). transpose()
     * is therefore the exact matrix transpose of apply() (weighted by W).
     *
     * The stencil is converted once per grid into a list of its non-zero
     * entries (coefficients with |c| <= 1e-14 max|c| are dropped), each with
     * precomputed input/output index offsets. The kernels loop over these
     * entries and, for each, sweep the computable region with a unit-stride
     * inner x loop, so the cost is O(nnz x n_c x pixels). The list is cached
     * per (grid, n_c) and rebuilt when either changes.
     *
     * The implementation supports portable execution across CPU and GPU
     * (CUDA/HIP) architectures through backend-specific kernel implementations.
     * The sparse operator is stored as a Structure-of-Arrays (separate index,
     * value and quadrature-point arrays), so each kernel streams contiguous
     * arrays and GPU threads reading the same field load consecutive words.
     *
     * @note There is no default constructor, and copy construction and copy
     * assignment are disabled (the operator owns its sparse caches).
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
         * @param coefficients The stencil coefficients raveled as an array
         * in column-major order with the output component fastest: entry
         * ((s * nb_pixel_input_components + i) * nb_quad_pts + q)
         * * nb_output_components + o, where s is the column-major (x
         * fastest) index of the stencil point. Its size must be
         * nb_output_components * nb_quad_pts * nb_pixel_input_components *
         * prod(stencil_shape).
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

        //! Single-precision (Real32) overload of apply().
        void apply(const TypedFieldBase<Real32> & nodal_field,
                   TypedFieldBase<Real32> & quadrature_point_field) const final;

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

        //! Single-precision (Real32) overload of apply_increment().
        void apply_increment(
            const TypedFieldBase<Real32> & nodal_field, const Real32 & alpha,
            TypedFieldBase<Real32> & quadrature_point_field) const override;

        /**
         * Evaluates the transpose of apply() into nodal_field, with each
         * quadrature point scaled by its entry of `weights` (one per
         * quadrature point). Without weights every quadrature point counts
         * once: the result is the bare matrix transpose Bᵀ. That differs from
         * FEMGradientOperator::transpose(), whose default is the
         * quadrature-weighted Bᵀ W with W = Wfrac · cell volume -- this
         * operator knows a stencil but no grid spacing, so it cannot supply W.
         * Pass the FEM operator's quadrature weights here to get its result.
         * @param quadrature_point_field input field of which to take
         * the divergence. Defined on quadrature points.
         * @param nodal_field output field into which divergence is written
         * @param weights Gaussian quadrature weights
         */
        void transpose(const TypedFieldBase<Real> & quadrature_point_field,
                       TypedFieldBase<Real> & nodal_field,
                       const std::vector<Real> & weights = {}) const final;

        //! Single-precision (Real32) overload of transpose().
        void transpose(const TypedFieldBase<Real32> & quadrature_point_field,
                       TypedFieldBase<Real32> & nodal_field,
                       const std::vector<Real> & weights = {}) const final;

        /**
         * As transpose(), scaled by alpha and added to nodal_field (Bᵀ with
         * unit weights by default; see transpose()).
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

        //! Single-precision (Real32) overload of transpose_increment().
        void transpose_increment(
            const TypedFieldBase<Real32> & quadrature_point_field,
            const Real32 & alpha, TypedFieldBase<Real32> & nodal_field,
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
         * @param weights Gaussian quadrature weights (copied to the device and
         * applied by the kernel)
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
         * @param weights Gaussian quadrature weights (copied to the device and
         * applied by the kernel)
         */
        void transpose_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> &
                quadrature_point_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights = {}) const;

        //! Single-precision (Real32) device overloads.
        void apply(
            const TypedFieldBase<Real32, DefaultDeviceSpace> & nodal_field,
            TypedFieldBase<Real32, DefaultDeviceSpace> & quadrature_point_field)
            const;
        void apply_increment(
            const TypedFieldBase<Real32, DefaultDeviceSpace> & nodal_field,
            const Real32 & alpha,
            TypedFieldBase<Real32, DefaultDeviceSpace> & quadrature_point_field)
            const;
        void transpose(const TypedFieldBase<Real32, DefaultDeviceSpace> &
                           quadrature_point_field,
                       TypedFieldBase<Real32, DefaultDeviceSpace> & nodal_field,
                       const std::vector<Real> & weights = {}) const;
        void transpose_increment(
            const TypedFieldBase<Real32, DefaultDeviceSpace> &
                quadrature_point_field,
            const Real32 & alpha,
            TypedFieldBase<Real32, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights = {}) const;
#endif

        /**
         * Return the stencil coefficients.
         */
        const std::vector<Real> & get_coefficients() const;

        /**
         * Return the stencil offset in number of pixels.
         */
        Shape_t get_offset() const final;

        /**
         * Return the shape of the stencil.
         */
        Shape_t get_stencil_shape() const final;

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
         * returns the number of input components (nodal sub-points) per
         * pixel/voxel, i.e. the nodes owned by one pixel/voxel, each counted
         * once even if it belongs to several elements.
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
         * values into a multiplication with a complex number in Fourier space:
         * for u_p = exp(2πi q·p), (B u)_p = fourier(q) u_p with
         *
         *     fourier(q) = sum_s c_s exp(2πi q·s).
         *
         * Only defined for scalar operators (one output component,
         * quadrature point and input component); throws otherwise. Cost
         * O(prod(stencil_shape)).
         *
         * @param phase The frequency q in cycles per grid point per axis
         * (the wavevector times the grid spacing, lacking the factor 2π)
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

        /**
         * Coefficients at most this fraction of the largest one are dropped
         * from the sparse operator. Relative, so it removes rounding noise in
         * the coefficient tables without depending on the units: an absolute
         * threshold of 1e-14 would drop every coefficient of an operator in
         * SI units at the micrometre scale (or of the identity times 1e-15).
         */
        static constexpr Real relative_zero_tolerance = 1e-14;

        //! Absolute threshold below which a coefficient counts as zero.
        Real zero_threshold() const {
            Real max_abs{0};
            for (const Real & value : this->coefficients_) {
                max_abs = std::max(max_abs, std::abs(value));
            }
            return relative_zero_tolerance * max_abs;
        }

        //! Cached sparse operators for reuse (two orderings, one per
        //! direction). Row-major: groups by quad index, so consecutive
        //! entries write the same output location in apply
        mutable std::optional<SparseOperatorSoA<HostSpace>> cached_apply_op_{};
        //! Column-major: groups by nodal index, so consecutive entries write
        //! the same output location in transpose
        mutable std::optional<SparseOperatorSoA<HostSpace>>
            cached_transpose_op_{};
        mutable std::optional<SparseOperatorCacheKey> cached_key_{};

        //! Device-space cached operators (lazily copied from host)
        mutable std::optional<SparseOperatorSoA<DefaultDeviceSpace>>
            cached_device_apply_op_{};
        mutable std::optional<SparseOperatorSoA<DefaultDeviceSpace>>
            cached_device_transpose_op_{};
        //! Grid/components key the device caches were built for; the device
        //! caches must be rebuilt when it changes (e.g. the operator is reused
        //! on a different grid), just like the host cache key.
        mutable std::optional<SparseOperatorCacheKey> cached_device_key_{};

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

        //! Host apply/transpose with increment, templated on the field scalar
        //! type T (Real or Real32). The cached sparse operator stays in double
        //! and its coefficients are cast to T inside the kernels.
        template <typename T>
        void apply_increment_impl(
            const TypedFieldBase<T> & nodal_field, T alpha,
            TypedFieldBase<T> & quadrature_point_field) const;
        template <typename T>
        void transpose_increment_impl(
            const TypedFieldBase<T> & quadrature_point_field, T alpha,
            TypedFieldBase<T> & nodal_field,
            const std::vector<Real> & weights) const;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        //! Device counterparts of the above (defined in generic_gpu.cc).
        template <typename T>
        void apply_increment_device_impl(
            const TypedFieldBase<T, DefaultDeviceSpace> & nodal_field, T alpha,
            TypedFieldBase<T, DefaultDeviceSpace> & quadrature_point_field)
            const;
        template <typename T>
        void transpose_increment_device_impl(
            const TypedFieldBase<T, DefaultDeviceSpace> & quadrature_point_field,
            T alpha, TypedFieldBase<T, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights) const;
#endif

        /**
         * Apply convolution on device memory fields.
         * Data must already be in device memory space, and the device apply
         * operator must already be cached (get_device_apply_operator()).
         *
         * @tparam DeviceSpace Target device memory space (CUDASpace, ROCmSpace)
         * @param nodal_data Pointer to nodal field data in device memory
         * @param quad_data Pointer to quadrature field data in device memory
         * @param alpha Scaling factor
         * @param params Grid traversal parameters
         */
        template <typename DeviceSpace, typename T>
        void apply_on_device(const T * nodal_data, T * quad_data,
                             const T alpha,
                             const GridTraversalParams & params) const;

        /**
         * Transpose convolution on device memory fields.
         * Data must already be in device memory space, and the device
         * transpose operator must already be cached
         * (get_device_transpose_operator()).
         *
         * @tparam DeviceSpace Target device memory space (CUDASpace, ROCmSpace)
         * @param quad_data Pointer to quadrature field data in device memory
         * @param nodal_data Pointer to nodal field data in device memory
         * @param alpha Scaling factor
         * @param params Grid traversal parameters
         * @param weights Device pointer to per-quad-point weights; nullptr for
         *   unweighted (all weights treated as 1)
         */
        template <typename DeviceSpace, typename T>
        void transpose_on_device(const T * quad_data, T * nodal_data,
                                 const T alpha,
                                 const GridTraversalParams & params,
                                 const Real * weights = nullptr) const;

        /**
         * Get or create device-space sparse operator for apply operation.
         * On a miss (or a different grid/component count) the operator is
         * built on the host with DeviceSpace::storage_order indices and
         * deep-copied to the device; the host cache is not involved.
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
         * Built and copied like get_device_apply_operator().
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

    // The kernel functions are in operators/convolution_kernels_cpu.hh and
    // operators/convolution_kernels_gpu.hh; deep_copy_sparse_operator is in
    // operators/kernel_dispatcher.hh

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

    template <typename DeviceSpace, typename T>
    void GenericLinearOperator::apply_on_device(
        const T * nodal_data, T * quad_data, const T alpha,
        const GridTraversalParams & params) const {
        // Device sparse operator, cached beforehand by
        // get_device_apply_operator()
        const auto & sparse_op = this->cached_device_apply_op_.value();

        // Use the KernelDispatcher for backend-agnostic kernel execution
        KernelDispatcher<DeviceSpace>::template apply_convolution<T>(
            nodal_data, quad_data, alpha, params, sparse_op);
    }

    template <typename DeviceSpace, typename T>
    void GenericLinearOperator::transpose_on_device(
        const T * quad_data, T * nodal_data, const T alpha,
        const GridTraversalParams & params, const Real * weights) const {
        // Device sparse operator, cached beforehand by
        // get_device_transpose_operator()
        const auto & sparse_op = this->cached_device_transpose_op_.value();

        // Use the KernelDispatcher for backend-agnostic kernel execution
        KernelDispatcher<DeviceSpace>::template transpose_convolution<T>(
            quad_data, nodal_data, alpha, params, sparse_op, weights);
    }

    template <typename DeviceSpace>
    const SparseOperatorSoA<DeviceSpace> &
    GenericLinearOperator::get_device_apply_operator(
        const DynGridIndex & nb_grid_pts, Index_t nb_nodal_components) const {
        // Rebuild when the cache is empty or was built for a different grid /
        // number of components (the precomputed indices are grid-specific).
        const SparseOperatorCacheKey key{nb_grid_pts, nb_nodal_components};
        const bool stale = this->cached_device_key_.has_value() &&
                           !(this->cached_device_key_.value() == key);
        if (!this->cached_device_apply_op_.has_value() || stale) {
            // Create operator with SoA indices (device storage order) and copy
            // it to the device. Device fields are StructureOfArrays so that
            // consecutive threads (consecutive pixels) access consecutive
            // addresses and loads coalesce.
            auto host_soa_op =
                this->create_apply_operator<DeviceSpace::storage_order>(
                    nb_grid_pts, nb_nodal_components);
            this->cached_device_apply_op_ =
                muGrid::deep_copy_sparse_operator<DeviceSpace, HostSpace>(
                    host_soa_op);
            if (stale) {
                // Grid changed: the transpose cache is stale as well.
                this->cached_device_transpose_op_.reset();
            }
            this->cached_device_key_ = key;
        }

        return this->cached_device_apply_op_.value();
    }

    template <typename DeviceSpace>
    const SparseOperatorSoA<DeviceSpace> &
    GenericLinearOperator::get_device_transpose_operator(
        const DynGridIndex & nb_grid_pts, Index_t nb_nodal_components) const {
        // Rebuild when the cache is empty or was built for a different grid /
        // number of components (the precomputed indices are grid-specific).
        const SparseOperatorCacheKey key{nb_grid_pts, nb_nodal_components};
        const bool stale = this->cached_device_key_.has_value() &&
                           !(this->cached_device_key_.value() == key);
        if (!this->cached_device_transpose_op_.has_value() || stale) {
            // Create operator with SoA indices (device storage order) and copy
            // it to the device (see get_device_apply_operator()).
            auto host_soa_op =
                this->create_transpose_operator<DeviceSpace::storage_order>(
                    nb_grid_pts, nb_nodal_components);
            this->cached_device_transpose_op_ =
                muGrid::deep_copy_sparse_operator<DeviceSpace, HostSpace>(
                    host_soa_op);
            if (stale) {
                // Grid changed: the apply cache is stale as well.
                this->cached_device_apply_op_.reset();
            }
            this->cached_device_key_ = key;
        }

        return this->cached_device_transpose_op_.value();
    }

}  // namespace muGrid
#endif  // SRC_LIBMUGRID_GENERIC_HH_
