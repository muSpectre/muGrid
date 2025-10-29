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

#include "convolution_operator_base.hh"

#include "Eigen/Dense"

#include <vector>

#ifndef SRC_LIBMUGRID_CONVOLUTION_OPERATOR_HH_
#define SRC_LIBMUGRID_CONVOLUTION_OPERATOR_HH_

namespace muGrid {
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
         * @param quadrature_point_field output field to increment by the gradient
         * field. Defined on quadrature points
         */
        void apply_increment(
            const TypedFieldBase<Real> &nodal_field, const Real &alpha,
            TypedFieldBase<Real> &quadrature_point_field) const override;

        /**
         * Evaluates the discretised divergence of quadrature_point_field into
         * nodal_field,  weights corrensponds to Gaussian quadrature weights. If
         * weights are omitted, this returns some scaled version of discretised
         * divergence.
         * @param quadrature_point_field input field of which to take
         * the divergence. Defined on quadrature points.
         * @param nodal_field ouput field into which divergence is written
         * @param weights Gaussian quadrature weigths
         */
        void transpose(const TypedFieldBase<Real> &quadrature_point_field,
                       TypedFieldBase<Real> &nodal_field,
                       const std::vector<Real> &weights = {}) const final;

        /**
         * Evaluates the discretised divergence of quadrature_point_field and adds
         * the result to nodal_field,  weights corrensponds to Gaussian quadrature
         * weights. If weights are omitted, this returns some scaled version of
         * discretised divergence.
         * @param quadrature_point_field input field of which to take
         * the divergence. Defined on quadrature points.
         * @param nodal_field ouput field to be incremented by theh divergence
         * @param weights Gaussian quadrature weigths
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
         * invovled in the convolution of one pixel.
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
         * number of nodal points that is invovled in the convolution of this pixel.
         */
        Index_t nb_operators;
        /**
         * the spatial dimension & number of the nodal points involved in the
         * convolution of one pixel.
         */
        Index_t spatial_dim;
        Index_t nb_conv_pts;
    };
} // namespace muGrid
#endif  // SRC_LIBMUGRID_CONVOLUTION_OPERATOR_HH_
