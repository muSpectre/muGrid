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

#include "gradient_operator_base.hh"

#include <Eigen/Dense>

#include <vector>

#ifndef SRC_LIBMUGRID_GRADIENT_OPERATOR_DEFAULT_HH_
#define SRC_LIBMUGRID_GRADIENT_OPERATOR_DEFAULT_HH_

namespace muGrid {

  class GradientOperatorDefault : public GradientOperatorBase {
   public:
    using Parent = GradientOperatorBase;
    //! Default constructor
    GradientOperatorDefault() = delete;

    /**
     * constructor
     *
     * @param spatial_dimension spatial dimension of the stencil
     * @param nb_quad_pts number of quadrature points per element
     * @param nb_elements number of elements per pixel
     * @param nb_elemenodal_pts number of nodal points per element
     * @param nb_pixelnodal_pts number of nodal points per pixelo
     * @param shape_fn_gradients per quadrature point and element, one matrix
     * of shape function gradients (evaluated on the quadrature point)
     * @param nodal_pts nodal point indices composed of nodal point index
     * within a pixel and pixel coordinate offset. E.g. the second nodal point
     * in pixel (i+1, j) gets (1, (1, 0))
     */
    GradientOperatorDefault(
        const Index_t & spatial_dim, const Index_t & nb_quad_pts,
        const Index_t & nb_elements, const Index_t & nb_elemnodal_pts,
        const Index_t & nb_pixelnodal_pts,
        const std::vector<std::vector<Eigen::MatrixXd>> & shape_fn_gradients,
        const std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>>
            nodal_pts);

    //! Copy constructor
    GradientOperatorDefault(const GradientOperatorDefault & other) = delete;

    //! Move constructor
    GradientOperatorDefault(GradientOperatorDefault && other) = default;

    //! Destructor
    virtual ~GradientOperatorDefault() = default;

    //! Copy assignment operator
    GradientOperatorDefault &
    operator=(const GradientOperatorDefault & other) = delete;

    //! Move assignment operator
    GradientOperatorDefault &
    operator=(GradientOperatorDefault && other) = default;

    /**
     * Evaluates the gradient of nodal_field into quadrature_point_field
     *
     * @param nodal_field input field of which to take gradient. Defined on
     * nodal points
     * @param quadrature_point_field output field to write gradient into.
     * Defined on quadrature points
     */
    void
    apply_gradient(const TypedFieldBase<Real> & nodal_field,
                   TypedFieldBase<Real> & quadrature_point_field) const final;

    /**
     * Evaluates the gradient of nodal_field and adds it to
     * quadrature_point_field
     *
     * @param nodal_field input field of which to take gradient. Defined on
     * nodal points
     * @param quadrature_point_field output field to increment by the gradient
     * field. Defined on quadrature points
     */
    void apply_gradient_increment(
        const TypedFieldBase<Real> & nodal_field, const Real & alpha,
        TypedFieldBase<Real> & quadrature_point_field) const override;

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
    void apply_transpose(const TypedFieldBase<Real> & quadrature_point_field,
                         TypedFieldBase<Real> & nodal_field,
                         const std::vector<Real> & weights = {}) const final;

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
    void apply_transpose_increment(
        const TypedFieldBase<Real> & quadrature_point_field, const Real & alpha,
        TypedFieldBase<Real> & nodal_field,
        const std::vector<Real> & weights = {}) const final;

    /**
     * Return the gradient matrix linking the nodal degrees of freedom to their
     * quadrature-point derivatives.
     */
    const Eigen::MatrixXd & get_pixel_gradient() const;

    /**
     * returns the number of quadrature points are associated with any
     * pixel/voxel (i.e., the sum of the number of quadrature points associated
     * with each element belonging to any pixel/voxel.
     */
    Index_t get_nb_pixel_quad_pts() const final;
    /**
     * returns the number of nodal points associated with any pixel/voxel.
     * (Every node belonging to at least one of the elements belonging to any
     * pixel/voxel, without recounting nodes that appear multiple times)
     */
    Index_t get_nb_pixel_nodal_pts() const final;
    /**
     * return the spatial dimension of this gradient operator
     */
    Index_t get_spatial_dim() const final;

    /**
     * return the number of quadrature points per element
     */
    const Index_t & get_nb_quad_pts_per_element() const;

    /**
     * return the number of elements per pixel
     */
    const Index_t & get_nb_elements() const;

   protected:
    /**
     * matrix linking the nodal degrees of freedom to their quadrature-point
     * derivatives.
     */
    Eigen::MatrixXd pixel_gradient{};
    Index_t spatial_dim;
    /**
     * number of quadrature points per element (e.g.. 4 for linear
     * quadrilateral)
     */
    Index_t nb_quad_pts;
    //! number of elements per pixel
    Index_t nb_elements;
    //! number of nodal points per element (e.g., 3 for triangles)
    Index_t nb_elemnodal_pts;
    //! number of nodal points per pixel
    Index_t nb_pixelnodal_pts;
    /**
     * number of nodal points that could possibly have an influnce on gradient
     * values in this pixel. This corresponds to the number of nodal points per
     * pixel  for this pixel plus the upper neighbour plus the right neighbour
     * plus the frontal neighbour for a three-dimensional problem.
     */
    Index_t nb_possible_nodal_contribution;
    // TODO(junge): Check with Martin whether this can be true. Why does it not
    // depend on rank?
    Index_t nb_grad_component_per_pixel;
  };

}  // namespace muGrid
#endif  // SRC_LIBMUGRID_GRADIENT_OPERATOR_DEFAULT_HH_
