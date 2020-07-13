/**
 * @file   projection_base.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Dec 2017
 *
 * @brief  Base class for Projection operators
 *
 * Copyright © 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
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

#ifndef SRC_PROJECTION_PROJECTION_BASE_HH_
#define SRC_PROJECTION_PROJECTION_BASE_HH_

#include <libmugrid/exception.hh>
#include <libmugrid/field_collection.hh>
#include <libmugrid/field_typed.hh>

#include <libmufft/fft_engine_base.hh>

#include "common/muSpectre_common.hh"

#include <memory>

namespace muSpectre {

  //! convenience alias
  using MatrixXXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

  template <class Projection>
  struct Projection_traits {};

  /**
   * base class for projection related exceptions
   */
  class ProjectionError : public muGrid::RuntimeError {
   public:
    //! constructor
    explicit ProjectionError(const std::string & what)
        : muGrid::RuntimeError(what) {}
    //! constructor
    explicit ProjectionError(const char * what) : muGrid::RuntimeError(what) {}
  };

  /**
   * defines the interface which must be implemented by projection operators
   */
  class ProjectionBase {
   public:
    //! Eigen type to replace fields
    using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    //! global FieldCollection
    using GFieldCollection_t =
        typename muFFT::FFTEngineBase::GFieldCollection_t;
    //! Field type on which to apply the projection
    using Field_t = muGrid::TypedFieldBase<Real>;
    /**
     * iterator over all pixels. This is taken from the FFT engine,
     * because depending on the real-to-complex FFT employed, only
     * roughly half of the pixels are present in Fourier space
     * (because of the hermitian nature of the transform)
     */
    using iterator = typename muFFT::FFTEngineBase::iterator;

    //! Default constructor
    ProjectionBase() = delete;

    //! Constructor with cell sizes
    ProjectionBase(muFFT::FFTEngine_ptr engine,
                   const DynRcoord_t & domain_lengths,
                   const Index_t & nb_quad_pts,
                   const Index_t & nb_components, const Formulation & form);

    //! Copy constructor
    ProjectionBase(const ProjectionBase & other) = delete;

    //! Move constructor
    ProjectionBase(ProjectionBase && other) = default;

    //! Destructor
    virtual ~ProjectionBase() = default;

    //! Copy assignment operator
    ProjectionBase & operator=(const ProjectionBase & other) = delete;

    //! Move assignment operator
    ProjectionBase & operator=(ProjectionBase && other) = delete;

    //! initialises the fft engine (plan the transform)
    virtual void initialise();

    //! apply the projection operator to a field
    virtual void apply_projection(Field_t & field) = 0;

    /**
     * returns the process-local number of grid points in each direction of the
     * cell
     */
    const DynCcoord_t & get_nb_subdomain_grid_pts() const;

    //! returns the process-local locations of the cell
    const DynCcoord_t & get_subdomain_locations() const {
      return this->fft_engine->get_subdomain_locations();
    }
    //! returns the global number of grid points in each direction of the cell
    const DynCcoord_t & get_nb_domain_grid_pts() const;

    //! returns the physical sizes of the cell
    const DynRcoord_t & get_domain_lengths() const {
      return this->domain_lengths;
    }

    //! returns the physical sizes of the pixles of the cell
    const DynRcoord_t get_pixel_lengths() const;

    /**
     * return the `muSpectre::Formulation` that is used in solving
     * this cell. This allows tho check whether a projection is
     * compatible with the chosen formulation
     */
    const Formulation & get_formulation() const { return this->form; }

    //! return the raw projection operator. This is mainly intended
    //! for maintenance and debugging and should never be required in
    //! regular use
    // virtual Eigen::Map<MatrixXXc> get_operator() = 0;

    //! return the communicator object
    const auto & get_communicator() const {
      return this->fft_engine->get_communicator();
    }

    /**
     * returns the number of rows and cols for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetric storage, it is a column
     * vector)
     */
    virtual std::array<Index_t, 2> get_strain_shape() const = 0;

    //! get number of components to project per pixel
    virtual Index_t get_nb_dof_per_pixel() const = 0;

    //! return the number of spatial dimensions
    const Index_t & get_dim() const;

    /**
     * returns the number of quadrature points
     */
    const Index_t & get_nb_quad_pts() const;

    /**
     * returns the number of nodal points
     */
    const Index_t & get_nb_nodal_pts() const;

    //! return a reference to the fft_engine
    muFFT::FFTEngineBase & get_fft_engine();

    //! return a reference to the fft_engine
    const muFFT::FFTEngineBase & get_fft_engine() const;

    //! perform a deep copy of the projector (this should never be necessary in
    //! c++)
    virtual std::unique_ptr<ProjectionBase> clone() const = 0;

   protected:
    //! handle on the fft_engine used
    muFFT::FFTEngine_ptr fft_engine;
    DynRcoord_t domain_lengths;  //!< physical sizes of the cell
    Index_t nb_quad_pts;
    Index_t nb_components;
    /**
     * formulation this projection can be applied to (determines
     * whether the projection enforces gradients, small strain tensor
     * or symmetric smal strain tensor
     */
    Formulation form;
    /**
     * A local `muSpectre::Field` to store the Fourier space representation of
     * the projected field per k-space point. This field is obtained from the
     * FFT engine, since the pixels considered depend on the FFT implementation.
     * See
     * http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data
     * for an example
     */
    muGrid::TypedFieldBase<Complex> & work_space;

    bool initialised{false};  //! has the projection been initialised?
  };

}  // namespace muSpectre

#endif  // SRC_PROJECTION_PROJECTION_BASE_HH_
