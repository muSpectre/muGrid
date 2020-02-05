/**
 * @file   fft_engine_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Dec 2017
 *
 * @brief  Interface for FFT engines
 *
 * Copyright © 2017 Till Junge
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#ifndef SRC_LIBMUFFT_FFT_ENGINE_BASE_HH_
#define SRC_LIBMUFFT_FFT_ENGINE_BASE_HH_

#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/field_collection_global.hh>
#include <libmugrid/field_typed.hh>

#include "communicator.hh"
#include "mufft_common.hh"

namespace muFFT {

  /**
   * Virtual base class for FFT engines. To be implemented by all
   * FFT_engine implementations.
   */
  class FFTEngineBase {
   public:
    //! global FieldCollection
    using GFieldCollection_t = muGrid::GlobalFieldCollection;
    //! pixel iterator
    using Pixels = typename GFieldCollection_t::DynamicPixels;
    /**
     * Field type on which to apply the projection.
     * This is a TypedFieldBase because it need to be able to hold
     * either TypedField or a WrappedField.
     */
    using Field_t = muGrid::TypedFieldBase<Real>;
    /**
     * Field type holding a Fourier-space representation of a
     * real-valued second-order tensor field
     */
    using Workspace_t = muGrid::ComplexField;
    /**
     * iterator over Fourier-space discretisation point
     */
    using iterator = typename GFieldCollection_t::DynamicPixels::iterator;

    //! Default constructor
    FFTEngineBase() = delete;

    /**
     * Constructor with the domain's number of grid points in each direciton,
     * the number of components to transform, and the communicator
     */
    FFTEngineBase(DynCcoord_t nb_grid_pts, Dim_t nb_dof_per_pixel,
                  Communicator comm = Communicator());

    //! Copy constructor
    FFTEngineBase(const FFTEngineBase & other) = delete;

    //! Move constructor
    FFTEngineBase(FFTEngineBase && other) = delete;

    //! Destructor
    virtual ~FFTEngineBase() = default;

    //! Copy assignment operator
    FFTEngineBase & operator=(const FFTEngineBase & other) = delete;

    //! Move assignment operator
    FFTEngineBase & operator=(FFTEngineBase && other) = delete;

    //! compute the plan, etc
    virtual void initialise(FFT_PlanFlags /*plan_flags*/);

    //! forward transform (dummy for interface)
    virtual Workspace_t & fft(Field_t & /*field*/) = 0;

    //! inverse transform (dummy for interface)
    virtual void ifft(Field_t & /*field*/) const = 0;

    //! return whether this engine is active
    virtual bool is_active() const { return true; }

    /**
     * iterators over only those pixels that exist in frequency space
     * (i.e. about half of all pixels, see rfft)
     */
    const Pixels & get_pixels() const;

    //! nb of pixels (mostly for debugging)
    size_t size() const;
    //! nb of pixels in Fourier space
    size_t fourier_size() const;
    //! nb of pixels in the work space (may contain a padding region)
    size_t workspace_size() const;

    //! return the communicator object
    const Communicator & get_communicator() const { return this->comm; }

    /**
     * returns the process-local number of grid points in each direction of the
     * cell
     */
    const DynCcoord_t & get_nb_subdomain_grid_pts() const {
      return this->nb_subdomain_grid_pts;
    }
    /**
     * returns the process-local number of grid points in each direction of the
     * cell
     */
    const DynCcoord_t & get_nb_domain_grid_pts() const {
      return this->nb_domain_grid_pts;
    }
    //! returns the process-local locations of the cell
    const DynCcoord_t & get_subdomain_locations() const {
      return this->subdomain_locations;
    }
    /**
     * returns the process-local number of grid points in each direction of the
     * cell in Fourier space
     */
    const DynCcoord_t & get_nb_fourier_grid_pts() const {
      return this->nb_fourier_grid_pts;
    }
    //! returns the process-local locations of the cell in Fourier space
    const DynCcoord_t & get_fourier_locations() const {
      return this->fourier_locations;
    }

    //! only required for testing and debugging
    GFieldCollection_t & get_field_collection() {
      return this->work_space_container;
    }
    //! only required for testing and debugging
    Workspace_t & get_work_space() { return this->work; }

    //! factor by which to multiply projection before inverse transform (this is
    //! typically 1/nb_pixels for so-called unnormalized transforms (see,
    //! e.g.
    //! http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data
    //! or https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.fft.html
    //! . Rather than scaling the inverse transform (which would cost one more
    //! loop), FFT engines provide this value so it can be used in the
    //! projection operator (where no additional loop is required)
    inline Real normalisation() const { return norm_factor; }

    //! return the number of components per pixel
    const Dim_t & get_nb_dof_per_pixel() const;

    //! return the number of spatial dimensions
    const Dim_t & get_spatial_dim() const;

    /**
     * returns the number of quadrature points
     */
    const Dim_t & get_nb_quad_pts() const;

    //! has this engine been initialised?
    bool is_initialised() const { return this->initialised; }

    //! perform a deep copy of the engine (this should never be necessary in
    //! c++)
    virtual std::unique_ptr<FFTEngineBase> clone() const = 0;

   protected:
    //! spatial dimension of the grid
    Dim_t spatial_dimension;
    /**
     * Field collection in which to store fields associated with
     * Fourier-space points
     */
    Communicator comm;  //!< communicator
    //! Field collection to store the fft workspace
    GFieldCollection_t work_space_container;
    DynCcoord_t nb_subdomain_grid_pts;  //!< nb_grid_pts of the process-local
                                        //!< (subdomain) portion of the cell
    DynCcoord_t subdomain_locations;    //!< location of the process-local
                                        //!< (subdomain) portion of the cell
    DynCcoord_t
        nb_fourier_grid_pts;  //!< nb_grid_pts of the process-local (subdomain)
                              //!< portion of the Fourier transformed data
    DynCcoord_t
        fourier_locations;  //!< location of the process-local (subdomain)
                            //!< portion of the Fourier transformed data
    const DynCcoord_t
        nb_domain_grid_pts;  //!< nb_grid_pts of the full domain of the cell
    Workspace_t & work;      //!< field to store the Fourier transform of P
    const Real norm_factor;  //!< normalisation coefficient of fourier transform
    //! number of degrees of freedom per pixel. Corresponds to the number of
    //! quadrature points per pixel multiplied by the number of components per
    //! quadrature point
    Dim_t nb_dof_per_pixel;
    bool initialised{false};  //!< to prevent double initialisation
  };

  //! reference to fft engine is safely managed through a `std::shared_ptr`
  using FFTEngine_ptr = std::shared_ptr<FFTEngineBase>;

}  // namespace muFFT

#endif  // SRC_LIBMUFFT_FFT_ENGINE_BASE_HH_
