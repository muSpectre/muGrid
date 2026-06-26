/**
 * @file   convolution_operator_base.hh
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

#include "core/exception.hh"
#include "core/types.hh"
#include "collection/field_collection_global.hh"
#include "field/field_typed.hh"

#include <algorithm>
#include <sstream>
#include <string>

#ifndef SRC_LIBMUGRID_LINEAR_OPERATOR_BASE_HH_
#define SRC_LIBMUGRID_LINEAR_OPERATOR_BASE_HH_

namespace muGrid {

  /**
   * @struct GhostRequirement
   * @brief Number of ghost layers an operator needs on each side of the
   *        subdomain in each spatial direction.
   *
   * Used by stencil operators to report their ghost-buffer needs, so that
   * domain decompositions and FFT engines can be constructed directly from
   * the operators that will run on them instead of hand-written ghost
   * counts.
   */
  struct GhostRequirement {
    Shape_t left{};   //!< ghost layers on the low-index side, per direction
    Shape_t right{};  //!< ghost layers on the high-index side, per direction

    /**
     * Elementwise maximum of two requirements; use to size one
     * decomposition serving several operators.
     */
    static GhostRequirement max(const GhostRequirement & a,
                                const GhostRequirement & b) {
      if (a.left.size() != b.left.size()) {
        std::stringstream err_msg{};
        err_msg << "Dimension mismatch: cannot combine a ghost requirement "
                << "for " << a.left.size() << "D with one for "
                << b.left.size() << "D";
        throw RuntimeError{err_msg.str()};
      }
      GhostRequirement combined{a};
      for (std::size_t direction{0}; direction < a.left.size(); ++direction) {
        combined.left[direction] =
            std::max(a.left[direction], b.left[direction]);
        combined.right[direction] =
            std::max(a.right[direction], b.right[direction]);
      }
      return combined;
    }
  };

  /**
   * @class LinearOperator
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
  class LinearOperator {
   public:
    /**
     * @brief Default constructor.
     *
     * Initializes a new instance of the GradientOperator class. This
     * constructor is defaulted, indicating that it performs no special
     * actions other than initializing the object.
     */
    LinearOperator() = default;

    /**
     * @brief Copy constructor (deleted).
     *
     * Disables the copy construction of GradientOperator instances.
     * This ensures that a GradientOperator object cannot be copied,
     * enforcing unique ownership of its resources.
     */
    LinearOperator(const LinearOperator & other) = delete;

    /**
     * @brief Move constructor.
     *
     * Enables the move semantics for GradientOperator instances. This
     * allows the efficient transfer of resources from one object to another
     * without copying.
     */
    LinearOperator(LinearOperator && other) = default;

    /**
     * @brief Virtual destructor.
     *
     * Ensures that derived classes can be properly cleaned up through pointers
     * to the base class. This destructor is defaulted.
     */
    virtual ~LinearOperator() = default;

    /**
     * @brief Copy assignment operator (deleted).
     *
     * Disables the copy assignment of GradientOperator instances.
     * This prevents the accidental or intentional copying of an instance,
     * enforcing unique ownership of its resources.
     */
    LinearOperator &
    operator=(const LinearOperator & other) = delete;

    /**
     * @brief Move assignment operator.
     *
     * Enables the move assignment of GradientOperator instances, allowing
     * resources to be transferred between objects without copying.
     */
    LinearOperator & operator=(LinearOperator && other) = default;

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
    apply(const TypedFieldBase<Real> & nodal_field,
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
    virtual void apply_increment(
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
    transpose(const TypedFieldBase<Real> & quadrature_point_field,
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
    virtual void transpose_increment(
        const TypedFieldBase<Real> & quadrature_point_field, const Real & alpha,
        TypedFieldBase<Real> & nodal_field,
        const std::vector<Real> & weights = {}) const = 0;

      /**
       * @brief Returns the number of output components per pixel/voxel.
       *
       * For gradient operators, this is the number of gradient components
       * (typically equal to spatial dimension).
       *
       * @return The number of output components.
       */
      virtual Index_t get_nb_output_components() const = 0;

   /**
     * @brief Returns the number of quadrature points per pixel/voxel.
     *
     * Calculates the total number of quadrature points associated with each
     * pixel/voxel, summing the quadrature points of all elements belonging to
     * the pixel/voxel.
     *
     * @return The total number of quadrature points per pixel/voxel.
     */
    virtual Index_t get_nb_quad_pts() const = 0;

    /**
     * @brief Returns the number of input components per pixel/voxel.
     *
     * For gradient operators, this is typically 1 (one scalar value per grid
     * point, with neighbors accessed via ghost communication).
     *
     * @return The number of input components per pixel/voxel.
     */
    virtual Index_t get_nb_input_components() const = 0;

    /**
     * @brief Returns the spatial dimension of the gradient operator.
     *
     * @return The spatial dimensionality of the operations performed by this
     *         gradient operator.
     */
    virtual Dim_t get_spatial_dim() const = 0;

    /**
     * @brief Returns the stencil offset in pixels.
     *
     * The offset is the position of the first stencil entry relative to the
     * pixel the operator is applied at; apply() reads input values in the
     * range [offset, offset + stencil_shape - 1] around each pixel.
     *
     * @return The stencil offset, one entry per spatial direction.
     */
    virtual Shape_t get_offset() const = 0;

    /**
     * @brief Returns the shape of the stencil in pixels.
     *
     * @return The stencil shape, one entry per spatial direction.
     */
    virtual Shape_t get_stencil_shape() const = 0;

    /**
     * @brief Ghost layers required by apply().
     *
     * Derived from the stencil geometry: apply() reads input values at
     * [offset, offset + stencil_shape - 1] around each pixel, requiring
     * max(0, -offset) ghost layers on the left and
     * max(0, offset + stencil_shape - 1) on the right.
     *
     * @return The minimum ghost layers for apply(), per direction.
     */
    GhostRequirement get_apply_ghost_requirement() const {
      const Shape_t offset{this->get_offset()};
      const Shape_t stencil_shape{this->get_stencil_shape()};
      GhostRequirement requirement{};
      for (std::size_t direction{0}; direction < offset.size(); ++direction) {
        requirement.left.push_back(
            std::max(Index_t{0}, -offset[direction]));
        requirement.right.push_back(std::max(
            Index_t{0}, offset[direction] + stencil_shape[direction] - 1));
      }
      return requirement;
    }

    /**
     * @brief Ghost layers required by transpose().
     *
     * The default assumes a gather-style transpose that reads at mirrored
     * stencil offsets, i.e. the mirror image of the apply() requirement.
     * Operators with a scatter-style transpose (writing into the same ghost
     * buffers that apply() reads, followed by ghost reduction) override
     * this method.
     *
     * @return The minimum ghost layers for transpose(), per direction.
     */
    virtual GhostRequirement get_transpose_ghost_requirement() const {
      GhostRequirement requirement{this->get_apply_ghost_requirement()};
      std::swap(requirement.left, requirement.right);
      return requirement;
    }

    /**
     * @brief Ghost layers covering both apply() and transpose().
     *
     * The elementwise maximum of the apply() and transpose() requirements.
     * This is the safe default for sizing a domain decomposition or FFT
     * engine that this operator will run on.
     *
     * @return Ghost layers sufficient for all operations, per direction.
     */
    GhostRequirement get_ghost_requirement() const {
      return GhostRequirement::max(this->get_apply_ghost_requirement(),
                                   this->get_transpose_ghost_requirement());
    }

   protected:
    /**
     * @brief Common field validation shared by all stencil operators.
     *
     * Verifies that the input and output fields share a single
     * GlobalFieldCollection whose spatial dimension matches this operator's,
     * and that its ghost buffers satisfy this operator's requirement. Returns
     * the collection so callers can read its grid extents. Operators with
     * extra requirements (e.g. component-count checks) run those after calling
     * this.
     *
     * Lifting this here removes the per-operator `validate_fields` copy that
     * each Laplace / FEM-gradient / stiffness operator carried, and keeps the
     * runtime check tied to get_spatial_dim()/the ghost requirement so it
     * cannot drift from what the operator reports.
     *
     * @param input          Operator input field.
     * @param output         Operator output field.
     * @param operator_name  Name used in error messages.
     * @param is_transpose   Check the transpose ghost requirement (default:
     *                       the apply requirement).
     * @return The validated GlobalFieldCollection shared by both fields.
     * @throws RuntimeError if any check fails.
     */
    const GlobalFieldCollection &
    check_fields(const Field & input, const Field & output,
                 const std::string & operator_name,
                 const bool is_transpose = false) const {
      const auto & input_collection = input.get_collection();
      if (&input_collection != &output.get_collection()) {
        throw RuntimeError(operator_name +
                           ": input and output fields must belong to the same "
                           "field collection");
      }
      const auto * global_fc =
          dynamic_cast<const GlobalFieldCollection *>(&input_collection);
      if (global_fc == nullptr) {
        throw RuntimeError(operator_name + " requires GlobalFieldCollection");
      }
      if (global_fc->get_spatial_dim() != this->get_spatial_dim()) {
        throw RuntimeError(
            "Field collection dimension (" +
            std::to_string(global_fc->get_spatial_dim()) +
            ") does not match operator dimension (" +
            std::to_string(this->get_spatial_dim()) + ")");
      }
      this->check_ghost_requirement(*global_fc, is_transpose, operator_name);
      return *global_fc;
    }

    /**
     * @brief Throws if a field collection's ghost buffers are too small for
     *        this operator.
     *
     * Checks against get_apply_ghost_requirement() or
     * get_transpose_ghost_requirement(), so the runtime check can never
     * diverge from the reported requirement.
     *
     * @param collection The field collection holding the operator's fields.
     * @param is_transpose Check the transpose requirement.
     * @param operator_name Name used in the error message.
     */
    void check_ghost_requirement(const GlobalFieldCollection & collection,
                                 const bool & is_transpose,
                                 const std::string & operator_name) const {
      const GhostRequirement requirement{
          is_transpose ? this->get_transpose_ghost_requirement()
                       : this->get_apply_ghost_requirement()};
      const auto & nb_ghosts_left{collection.get_nb_ghosts_left()};
      const auto & nb_ghosts_right{collection.get_nb_ghosts_right()};
      for (Dim_t direction{0}; direction < collection.get_spatial_dim();
           ++direction) {
        if (nb_ghosts_left[direction] < requirement.left[direction] ||
            nb_ghosts_right[direction] < requirement.right[direction]) {
          std::stringstream err_msg{};
          err_msg << operator_name << " requires at least "
                  << requirement.left[direction] << " ghost layer(s) on the "
                  << "left and " << requirement.right[direction]
                  << " on the right of axis " << direction << " for the "
                  << (is_transpose ? "transpose" : "apply")
                  << " operation, but the field collection has "
                  << nb_ghosts_left[direction] << " on the left and "
                  << nb_ghosts_right[direction] << " on the right.";
          throw RuntimeError{err_msg.str()};
        }
      }
    }
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_LINEAR_OPERATOR_BASE_HH_
