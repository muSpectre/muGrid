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
   * @class GradientOperatorBase
   * @brief Base class for gradient and divergence operations.
   *
   * This class defines the interface for performing gradient and divergence
   * operations in the context of finite element analysis. It provides the
   * foundational structure for implementing these operations on various types
   * of fields and supports both nodal and quadrature point evaluations.
   *
   * @details The class is designed to be inherited by specific implementations
   * that define the actual computational logic for gradient and divergence
   * operations. It includes constructors, a destructor, and assignment
   * operators to manage object lifecycle and ensure proper resource management.
   */
  class GradientOperatorBase {
   public:
    /**
     * @brief Default constructor.
     *
     * Initializes a new instance of the GradientOperatorBase class. This
     * constructor is defaulted, indicating that it performs no special
     * actions other than initializing the object.
     */
    GradientOperatorBase() = default;

    /**
     * @brief Copy constructor (deleted).
     *
     * Disables the copy construction of GradientOperatorBase instances.
     * This ensures that a GradientOperatorBase object cannot be copied,
     * enforcing unique ownership of its resources.
     */
    GradientOperatorBase(const GradientOperatorBase & other) = delete;

    /**
     * @brief Move constructor.
     *
     * Enables the move semantics for GradientOperatorBase instances. This
     * allows the efficient transfer of resources from one object to another
     * without copying.
     */
    GradientOperatorBase(GradientOperatorBase && other) = default;

    /**
     * @brief Virtual destructor.
     *
     * Ensures that derived classes can be properly cleaned up through pointers
     * to the base class. This destructor is defaulted.
     */
    virtual ~GradientOperatorBase() = default;

    /**
     * @brief Copy assignment operator (deleted).
     *
     * Disables the copy assignment of GradientOperatorBase instances.
     * This prevents the accidental or intentional copying of an instance,
     * enforcing unique ownership of its resources.
     */
    GradientOperatorBase &
    operator=(const GradientOperatorBase & other) = delete;

    /**
     * @brief Move assignment operator.
     *
     * Enables the move assignment of GradientOperatorBase instances, allowing
     * resources to be transferred between objects without copying.
     */
    GradientOperatorBase & operator=(GradientOperatorBase && other) = default;

    /**
     * @brief Applies the gradient operation.
     *
     * This method evaluates the gradient of a field defined at nodal points and
     * writes the result into a field defined at quadrature points.
     *
     * @param nodal_field The input field from which the gradient is computed.
     *                    Defined on nodal points.
     * @param quadrature_point_field The output field where the gradient is
     *                               written. Defined on quadrature points.
     */
    virtual void
    apply_gradient(const TypedFieldBase<Real> & nodal_field,
                   TypedFieldBase<Real> & quadrature_point_field) const = 0;

    /**
     * @brief Applies the gradient operation with increment.
     *
     * Evaluates the gradient of a field defined at nodal points and adds the
     * result to a field defined at quadrature points.
     *
     * @param nodal_field The input field from which the gradient is computed.
     *                    Defined on nodal points.
     * @param alpha A scaling factor applied to the gradient before adding it to
     *              the quadrature_point_field.
     * @param quadrature_point_field The field to which the scaled gradient is
     *                               added. Defined on quadrature points.
     */
    virtual void apply_gradient_increment(
        const TypedFieldBase<Real> & nodal_field, const Real & alpha,
        TypedFieldBase<Real> & quadrature_point_field) const = 0;

    /**
     * @brief Applies the discretised divergence operation.
     *
     * Evaluates the discretised divergence of a field defined at quadrature
     * points and writes the result into a field defined at nodal points.
     *
     * @param quadrature_point_field The input field from which the divergence
     * is computed. Defined on quadrature points.
     * @param nodal_field The output field where the divergence is written.
     *                    Defined on nodal points.
     * @param weights Optional Gaussian quadrature weights. If omitted, a scaled
     *                version of the discretised divergence is returned.
     */
    virtual void
    apply_transpose(const TypedFieldBase<Real> & quadrature_point_field,
                    TypedFieldBase<Real> & nodal_field,
                    const std::vector<Real> & weights = {}) const = 0;

    /**
     * @brief Applies the discretised divergence operation with increment.
     *
     * Evaluates the discretised divergence of a field defined at quadrature
     * points and adds the result to a field defined at nodal points.
     *
     * @param quadrature_point_field The input field from which the divergence
     * is computed. Defined on quadrature points.
     * @param alpha A scaling factor applied to the divergence before adding it
     * to the nodal_field.
     * @param nodal_field The field to which the scaled divergence is added.
     *                    Defined on nodal points.
     * @param weights Optional Gaussian quadrature weights. If omitted, a scaled
     *                version of the discretised divergence is returned.
     */
    virtual void apply_transpose_increment(
        const TypedFieldBase<Real> & quadrature_point_field, const Real & alpha,
        TypedFieldBase<Real> & nodal_field,
        const std::vector<Real> & weights = {}) const = 0;

    /**
     * @brief Returns the number of quadrature points per pixel/voxel.
     *
     * Calculates the total number of quadrature points associated with each
     * pixel/voxel, summing the quadrature points of all elements belonging to
     * the pixel/voxel.
     *
     * @return The total number of quadrature points per pixel/voxel.
     */
    virtual Index_t get_nb_pixel_quad_pts() const = 0;

    /**
     * @brief Returns the number of nodal points per pixel/voxel.
     *
     * Calculates the total number of nodal points associated with each
     * pixel/voxel, without recounting nodes that appear in multiple elements.
     *
     * @return The total number of nodal points per pixel/voxel.
     */
    virtual Index_t get_nb_pixel_nodal_pts() const = 0;

    /**
     * @brief Returns the spatial dimension of the gradient operator.
     *
     * @return The spatial dimensionality of the operations performed by this
     *         gradient operator.
     */
    virtual Index_t get_spatial_dim() const = 0;

   protected:
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_GRADIENT_OPERATOR_BASE_HH_
