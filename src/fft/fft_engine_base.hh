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
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#ifndef FFT_ENGINE_BASE_H
#define FFT_ENGINE_BASE_H

#include "common/common.hh"
#include "common/communicator.hh"
#include "common/field_collection.hh"

namespace muSpectre {

  /**
   * Virtual base class for FFT engines. To be implemented by all
   * FFT_engine implementations.
   */
  template <Dim_t DimS>
  class FFTEngineBase
  {
  public:
    constexpr static Dim_t sdim{DimS}; //!< spatial dimension of the cell
    //! cell coordinates type
    using Ccoord = Ccoord_t<DimS>;
    //! global FieldCollection
    using GFieldCollection_t = GlobalFieldCollection<DimS>;
    //! local FieldCollection (for Fourier-space pixels)
    using LFieldCollection_t = LocalFieldCollection<DimS>;
    //! Field type on which to apply the projection
    using Field_t = TypedField<GFieldCollection_t, Real>;
    /**
     * Field type holding a Fourier-space representation of a
     * real-valued second-order tensor field
     */
    using Workspace_t = TypedField<LFieldCollection_t, Complex>;
    /**
     * iterator over Fourier-space discretisation point
     */
    using iterator = typename LFieldCollection_t::iterator;

    //! Default constructor
    FFTEngineBase() = delete;

    //! Constructor with cell resolutions
    FFTEngineBase(Ccoord resolutions, Dim_t nb_components,
                  Communicator comm=Communicator());

    //! Copy constructor
    FFTEngineBase(const FFTEngineBase &other) = delete;

    //! Move constructor
    FFTEngineBase(FFTEngineBase &&other) = default;

    //! Destructor
    virtual ~FFTEngineBase() = default;

    //! Copy assignment operator
    FFTEngineBase& operator=(const FFTEngineBase &other) = delete;

    //! Move assignment operator
    FFTEngineBase& operator=(FFTEngineBase &&other) = default;

    //! compute the plan, etc
    virtual void initialise(FFT_PlanFlags /*plan_flags*/);

    //! forward transform (dummy for interface)
    virtual Workspace_t & fft(Field_t & /*field*/) = 0;

    //! inverse transform (dummy for interface)
    virtual void ifft(Field_t & /*field*/) const = 0;

    /**
     * iterators over only those pixels that exist in frequency space
     * (i.e. about half of all pixels, see rfft)
     */
    //! returns an iterator to the first pixel in Fourier space
    inline iterator begin() {return this->work_space_container.begin();}
    //! returns an iterator past to the last pixel in Fourier space
    inline iterator end()  {return this->work_space_container.end();}

    //! nb of pixels (mostly for debugging)
    size_t size() const;
    //! nb of pixels in Fourier space
    size_t workspace_size() const;

    //! return the communicator object
    const Communicator & get_communicator() const {return this->comm;}

    //! returns the process-local resolutions of the cell
    const Ccoord & get_subdomain_resolutions() const {
      return this->subdomain_resolutions;}
    //! returns the process-local locations of the cell
    const Ccoord & get_subdomain_locations() const {
      return this->subdomain_locations;}
    //! returns the process-local resolutions of the cell in Fourier space
    const Ccoord & get_fourier_resolutions() const {return this->fourier_resolutions;}
    //! returns the process-local locations of the cell in Fourier space
    const Ccoord & get_fourier_locations() const {return this->fourier_locations;}
    //! returns the resolutions of the cell
    const Ccoord & get_domain_resolutions() const {return this->domain_resolutions;}

    //! only required for testing and debugging
    LFieldCollection_t & get_field_collection() {
      return this->work_space_container;}
    //! only required for testing and debugging
    Workspace_t& get_work_space() {return this->work;}

    //! factor by which to multiply projection before inverse transform (this is
    //! typically 1/nb_pixels for so-called unnormalized transforms (see,
    //! e.g. http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data
    //! or https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.fft.html
    //! . Rather than scaling the inverse transform (which would cost one more
    //! loop), FFT engines provide this value so it can be used in the
    //! projection operator (where no additional loop is required)
    inline Real normalisation() const {return norm_factor;};

    //! return the number of components per pixel
    Dim_t get_nb_components() const {return nb_components;}

  protected:
    /**
     * Field collection in which to store fields associated with
     * Fourier-space points
     */
    Communicator comm; //!< communicator
    LFieldCollection_t work_space_container{};
    Ccoord subdomain_resolutions; //!< resolutions of the process-local (subdomain) portion of the cell
    Ccoord subdomain_locations; // !< location of the process-local (subdomain) portion of the cell
    Ccoord fourier_resolutions; //!< resolutions of the process-local (subdomain) portion of the Fourier transformed data
    Ccoord fourier_locations; // !< location of the process-local (subdomain) portion of the Fourier transformed data
    const Ccoord domain_resolutions; //!< resolutions of the full domain of the cell
    Workspace_t & work; //!< field to store the Fourier transform of P
    const Real norm_factor; //!< normalisation coefficient of fourier transform
    Dim_t nb_components;
  private:
  };

}  // muSpectre

#endif /* FFT_ENGINE_BASE_H */
