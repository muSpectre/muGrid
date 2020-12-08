/**
 * @file   gradient_operator_base.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   16 Jun 2020
 *
 * @brief  Interface for gradient (and divergence) operators
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

#include "grid_common.hh"
#include "field_typed.hh"

#ifndef SRC_LIBMUGRID_GRADIENT_OPERATOR_BASE_HH_
#define SRC_LIBMUGRID_GRADIENT_OPERATOR_BASE_HH_

namespace muGrid {

  /**
   * Base class defining the interface for gradient and divergence operations
   * (in the integral, finite-element sense
   */
  class GradientOperatorBase {
   public:
    //! Default constructor
    GradientOperatorBase() = default;

    //! Copy constructor
    GradientOperatorBase(const GradientOperatorBase & other) = delete;

    //! Move constructor
    GradientOperatorBase(GradientOperatorBase && other) = default;

    //! Destructor
    virtual ~GradientOperatorBase() = default;

    //! Copy assignment operator
    GradientOperatorBase &
    operator=(const GradientOperatorBase & other) = delete;

    //! Move assignment operator
    GradientOperatorBase & operator=(GradientOperatorBase && other) = default;

    /**
     * Evaluates the gradient of nodal_field into quadrature_point_field
     *
     * @param nodal_field input field of which to take gradient. Defined on
     * nodal points
     * @param quadrature_point_field output field to write gradient into.
     * Defined on quadrature points
     */
    virtual void
    apply_gradient(const TypedFieldBase<Real> & nodal_field,
                   TypedFieldBase<Real> & quadrature_point_field) const = 0;

    /**
     * Evaluates the gradient of nodal_field and adds it to
     * quadrature_point_field
     *
     * @param nodal_field input field of which to take gradient. Defined on
     * nodal points
     * @param quadrature_point_field output field to increment by the gradient
     * field. Defined on quadrature points
     */
    virtual void apply_gradient_increment(
        const TypedFieldBase<Real> & nodal_field, const Real & alpha,
        TypedFieldBase<Real> & quadrature_point_field) const = 0;

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
    virtual void
    apply_transpose(const TypedFieldBase<Real> & quadrature_point_field,
                    TypedFieldBase<Real> & nodal_field,
                    const std::vector<Real> & weights = {}) const = 0;

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
    virtual void apply_transpose_increment(
        const TypedFieldBase<Real> & quadrature_point_field, const Real & alpha,
        TypedFieldBase<Real> & nodal_field,
        const std::vector<Real> & weights = {}) const = 0;

    /**
     * returns the number of quadrature points are associated with any
     * pixel/voxel (i.e., the sum of the number of quadrature points associated
     * with each element belonging to any pixel/voxel.
     */
    virtual Index_t get_nb_pixel_quad_pts() const = 0;
    /**
     * returns the number of nodal points associated with any pixel/voxel.
     * (Every node belonging to at least one of the elements belonging to any
     * pixel/voxel, without recounting nodes that appear multiple times)
     */
    virtual Index_t get_nb_pixel_nodal_pts() const = 0;

    /**
     * return the spatial dimension of this gradient operator
     */
    virtual Index_t get_spatial_dim() const = 0;

   protected:
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_GRADIENT_OPERATOR_BASE_HH_
