/**
 * @file   derivative.hh
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *         Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   05 June 2019
 *
 * @brief  Representation of finite-differences stencils
 *
 * Copyright © 2019 Lars Pastewka
 *
 * µFFT is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µFFT is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µFFT; see the file COPYING. If not, write to the
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

#ifndef SRC_LIBMUFFT_DERIVATIVE_HH_
#define SRC_LIBMUFFT_DERIVATIVE_HH_

#include <memory>

#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/exception.hh>
#include <libmugrid/field_map.hh>
#include <libmugrid/field_typed.hh>
#include <libmugrid/field_collection_global.hh>

#include "mufft_common.hh"

namespace muFFT {
  /**
   * base class for projection related exceptions
   */
  class DerivativeError : public RuntimeError {
   public:
    //! constructor
    explicit DerivativeError(const std::string & what)
        : RuntimeError(what) {}
    //! constructor
    explicit DerivativeError(const char * what) : RuntimeError(what) {}
  };

  /**
   * Representation of a derivative
   */
  class DerivativeBase {
   public:
    //! convenience alias
    using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    //! Deleted default constructor
    DerivativeBase() = delete;

    //! constructor with spatial dimension
    explicit DerivativeBase(Index_t spatial_dimension);

    //! Copy constructor
    DerivativeBase(const DerivativeBase & other) = default;

    //! Move constructor
    DerivativeBase(DerivativeBase && other) = default;

    //! Destructor
    virtual ~DerivativeBase() = default;

    //! Copy assignment operator
    DerivativeBase & operator=(const DerivativeBase & other) = delete;

    //! Move assignment operator
    DerivativeBase & operator=(DerivativeBase && other) = delete;

    /**
     * Return Fourier representation of the derivative as a function of the
     * phase. The phase is the wavevector times cell dimension, but lacking a
     * factor of 2 π.
     */
    virtual Complex fourier(const Vector & phase) const = 0;

   protected:
    //! spatial dimension of the problem
    Index_t spatial_dimension;
  };

  /**
   * Representation of a derivative computed by Fourier interpolation
   */
  class FourierDerivative : public DerivativeBase {
   public:
    using Parent = DerivativeBase;  //!< base class
    //! convenience alias
    using Vector = typename Parent::Vector;

    //! Default constructor
    FourierDerivative() = delete;

    //! Constructor with raw FourierDerivative information
    explicit FourierDerivative(Index_t spatial_dimension, Index_t direction);

    //! Constructor with raw FourierDerivative information and shift info
    explicit FourierDerivative(Index_t spatial_dimension, Index_t direction,
                               const Eigen::ArrayXd & shift);

    //! Copy constructor
    FourierDerivative(const FourierDerivative & other) = default;

    //! Move constructor
    FourierDerivative(FourierDerivative && other) = default;

    //! Destructor
    virtual ~FourierDerivative() = default;

    //! Copy assignment operator
    FourierDerivative & operator=(const FourierDerivative & other) = delete;

    //! Move assignment operator
    FourierDerivative & operator=(FourierDerivative && other) = delete;

    /**
     * Return Fourier representation of the Fourier interpolated derivative
     * shifted to the new position of the derivative. This here simply returns
     * I*2*pi*phase * e^(I*2*pi*shift*phase). (I*2*pi*wavevector is the
     * Fourier representation of the derivative and e^(I*2*pi*shift*phase)
     * shifts the derivative to its new position.)
     **/
    virtual Complex fourier(const Vector & phase) const {
      return Complex(0, 2 * muGrid::pi * phase[this->direction]) *
             std::exp(
                 Complex(0, 2 * muGrid::pi * this->shift.matrix().dot(phase)));
    }

   protected:
    //! spatial direction in which to perform differentiation
    Index_t direction;
    //! real space shift from the position of the center of the cell.
    const Eigen::ArrayXd shift;
  };

  /**
   * Representation of a finite-differences stencil
   */
  class DiscreteDerivative : public DerivativeBase {
   public:
    using Parent = DerivativeBase;  //!< base class
    //! convenience alias
    using Vector = typename Parent::Vector;

    //! Default constructor
    DiscreteDerivative() = delete;

    /**
     * Constructor with raw stencil information
     * @param nb_pts: stencil size
     * @param lbounds: relative starting point of stencil
     * @param stencil: stencil coefficients
     */
    DiscreteDerivative(DynCcoord_t nb_pts, DynCcoord_t lbounds,
                       const std::vector<Real> & stencil);

    //! Constructor with raw stencil information
    DiscreteDerivative(DynCcoord_t nb_pts, DynCcoord_t lbounds,
                       const Eigen::ArrayXd & stencil);

    //! Copy constructor
    DiscreteDerivative(const DiscreteDerivative & other) = default;

    //! Move constructor
    DiscreteDerivative(DiscreteDerivative && other) = default;

    //! Destructor
    virtual ~DiscreteDerivative() = default;

    //! Copy assignment operator
    DiscreteDerivative & operator=(const DiscreteDerivative & other) = delete;

    //! Move assignment operator
    DiscreteDerivative & operator=(DiscreteDerivative && other) = delete;

    //! Return stencil value
    Real operator()(const DynCcoord_t & dcoord) const {
      return this->stencil[this->pixels.get_index(dcoord)];
    }

    //! Return number of grid points in stencil
    const DynCcoord_t & get_nb_pts() const {
      return this->pixels.get_nb_subdomain_grid_pts();
    }

    //! Return lower stencil bound
    const DynCcoord_t & get_lbounds() const {
      return this->pixels.get_subdomain_locations();
    }

    //! Return the pixels class that allows to iterate over pixels
    const muGrid::CcoordOps::DynamicPixels & get_pixels() const {
      return this->pixels;
    }

    /**
     * Apply the "stencil" to a component (degree-of-freedom) of a field and
     * store the result to a select component of a second field. Note that the
     * compiler should have opportunity to inline this function to optimize
     * loops over DOFs.
     * TODO: This presently only works *without* MPI parallelization! If you
     * need parallelization, apply the stencil in Fourier space using the
     * `fourier` method. Currently this method is only used in the serial tests.
     */
    template <typename T>
    void apply(const muGrid::TypedField<T> & in_field, Index_t in_dof,
               muGrid::TypedField<T> & out_field, Index_t out_dof,
               Real fac = 1.0) const {
      // check whether fields are global
      if (!in_field.is_global()) {
        throw DerivativeError("Input field must be a global field.");
      }
      if (!out_field.is_global()) {
        throw DerivativeError("Output field must be a global field.");
      }
      // check whether specified dofs are in range
      if (in_dof < 0 or in_dof >= in_field.get_nb_dof_per_pixel()) {
        std::stringstream ss{};
        ss << "Component " << in_dof << " of input field does not exist."
           << "(Input field has " << in_field.get_nb_dof_per_pixel()
           << " components.)";
        throw DerivativeError(ss.str());
      }
      if (out_dof < 0 or out_dof >= out_field.get_nb_dof_per_pixel()) {
        std::stringstream ss{};
        ss << "Component " << out_dof << " of output field does not exist."
           << "(Input field has " << out_field.get_nb_dof_per_pixel()
           << " components.)";
        throw DerivativeError(ss.str());
      }
      // get global field collections
      const auto & in_collection{
          dynamic_cast<const muGrid::GlobalFieldCollection &>(
              in_field.get_collection())};
      const auto & out_collection{
          dynamic_cast<const muGrid::GlobalFieldCollection &>(
              in_field.get_collection())};
      if (in_collection.get_nb_pixels() != out_collection.get_nb_pixels()) {
        std::stringstream ss{};
        ss << "Input fields lives on a " << in_collection.get_nb_pixels()
           << " grid, but output fields lives on an incompatible "
           << out_collection.get_nb_pixels() << " grid.";
        throw DerivativeError(ss.str());
      }

      // construct maps
      muGrid::FieldMap<Real, Mapping::Const> in_map{in_field,
                                                    muGrid::IterUnit::Pixel};
      muGrid::FieldMap<Real, Mapping::Mut> out_map{out_field,
                                                   muGrid::IterUnit::Pixel};
      // loop over field pixel iterator
      Index_t ndim{in_collection.get_spatial_dim()};
      auto & nb_grid_pts{
          in_collection.get_pixels().get_nb_subdomain_grid_pts()};
      in_collection.get_pixels().get_nb_subdomain_grid_pts();
      for (const auto && coord : in_collection.get_pixels()) {
        T derivative{};
        // loop over stencil
        for (const auto && dcoord : this->pixels) {
          auto coord2{coord + dcoord};
          // TODO(pastewka): This only works in serial. For this to work
          //  properly in (MPI) parallel, we need ghost buffers (which will
          //  affect large parts of the code).
          for (Index_t dim{0}; dim < ndim; ++dim) {
            coord2[dim] =
                muGrid::CcoordOps::modulo(coord2[dim], nb_grid_pts[dim]);
          }
          derivative += this->stencil[this->pixels.get_index(dcoord)] *
                        in_map[in_collection.get_index(coord2)](in_dof);
        }
        out_map[out_collection.get_index(coord)](out_dof) = fac * derivative;
      }
    }

    /**
     * Any translationally invariant linear combination of grid values (as
     * expressed through the "stencil") becomes a multiplication with a number
     * in Fourier space. This method returns the Fourier representation of
     * this stencil.
     */
    virtual Complex fourier(const Vector & phase) const {
      Complex s{0, 0};
      for (auto && dcoord : muGrid::CcoordOps::DynamicPixels(
               this->pixels.get_nb_subdomain_grid_pts(),
               this->pixels.get_subdomain_locations())) {
        const Real arg{phase.matrix().dot(eigen(dcoord).template cast<Real>())};
        s += this->operator()(dcoord) *
             std::exp(Complex(0, 2 * muGrid::pi * arg));
      }
      return s;
    }

    /**
     * Return a new stencil rolled axes. Given a stencil on a
     * three-dimensional grid with axes (x, y, z), the stencil
     * that has been "rolled" by distance one has axes (z, x, y).
     * This is a simple implementation of a rotation operation.
     * For example, given a stencil that described the derivative in
     * the x-direction, rollaxes(1) gives the derivative in the
     * y-direction and rollaxes(2) gives the derivative in the
     * z-direction.
     */
    DiscreteDerivative rollaxes(int distance = 1) const;

    //! return the stencil data
    const Eigen::ArrayXd & get_stencil() const { return this->stencil; }

   protected:
    muGrid::CcoordOps::DynamicPixels pixels{};  //!< iterate over the stencil
    const Eigen::ArrayXd stencil;               //!< Finite-differences stencil
  };

  /**
   * Allows inserting `muFFT::DiscreteDerivative`s into `std::ostream`s
   */
  std::ostream & operator<<(std::ostream & os,
                            const DiscreteDerivative & derivative);

  //! convenience alias
  using Gradient_t = std::vector<std::shared_ptr<DerivativeBase>>;

  /**
   * convenience function to build a spatial_dimension-al gradient operator
   * using exact Fourier differentiation
   *
   * @param spatial_dimension number of spatial dimensions
   */
  Gradient_t make_fourier_gradient(const Index_t & spatial_dimension);
}  // namespace muFFT

#endif  // SRC_LIBMUFFT_DERIVATIVE_HH_
