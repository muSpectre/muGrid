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
         * @param pixel_operator The pixel-wise operator raveled as a matrix.
         * @param conv_pts_shape Shape of the stencil.
         * @param nb_pixelnodal_pts Number of nodal points per pixel.
         * @param nb_quad_pts Number of quadrature points per pixel.
         * @param nb_operators Number of operators in the stencil.
         */
        ConvolutionOperator(
            const Shape_t &pixel_offset,
            const Eigen::MatrixXd &pixel_operator,
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
         * Return the operator matrix linking the nodal degrees of freedom to their
         * quadrature-point values.
         */
        const Eigen::MatrixXd &get_pixel_operator() const;

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

    protected:
        /**
         * stencil offset in number of pixels
         */
        Shape_t pixel_offset{};
        /**
         * matrix linking the nodal degrees of freedom to their quadrature-point
         * values.
         */
        Eigen::MatrixXd pixel_operator{};
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

    private:
        //! Tolerance for considering operator values as zero
        static constexpr Real zero_tolerance = 1e-14;

        //! Cached sparse operator for reuse
        mutable std::optional<SparseOperatorSoA<HostSpace>> cached_sparse_op{};
        mutable std::optional<SparseOperatorCacheKey> cached_key{};

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
         * @brief Get or create the sparse operator representation
         * @param nb_grid_pts number of process-local (subdomain) grid points
         * with ghosts
         * @param nb_nodal_components number of components in nodal field
         * @return Reference to the sparse operator (cached)
         */
        const SparseOperatorSoA<HostSpace>&
        get_sparse_operator(const IntCoord_t & nb_grid_pts,
                            const Index_t nb_nodal_components) const;

        /**
         * @brief Create a new sparse operator representation
         * @param nb_grid_pts number of process-local (subdomain) grid points
         * with ghosts
         * @param nb_nodal_components number of components in nodal field
         * @return The sparse operator in SoA format
         */
        SparseOperatorSoA<HostSpace>
        create_sparse_operator(const IntCoord_t & nb_grid_pts,
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
    };

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

} // namespace muGrid
#endif  // SRC_LIBMUGRID_CONVOLUTION_OPERATOR_HH_
