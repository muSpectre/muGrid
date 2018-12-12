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

#ifndef SRC_FFT_PROJECTION_BASE_HH_
#define SRC_FFT_PROJECTION_BASE_HH_

#include "common/common.hh"
#include "common/field.hh"
#include "common/field_collection.hh"
#include "fft/fft_engine_base.hh"

#include <memory>

namespace muSpectre {

  /**
   * base class for projection related exceptions
   */
  class ProjectionError : public std::runtime_error {
   public:
    //! constructor
    explicit ProjectionError(const std::string &what)
        : std::runtime_error(what) {}
    //! constructor
    explicit ProjectionError(const char *what) : std::runtime_error(what) {}
  };

  template <class Projection> struct Projection_traits {};

  /**
   * defines the interface which must be implemented by projection operators
   */
  template <Dim_t DimS, Dim_t DimM> class ProjectionBase {
   public:
    //! Eigen type to replace fields
    using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    //! type of fft_engine used
    using FFTEngine = FFTEngineBase<DimS>;
    //! reference to fft engine is safely managed through a `std::unique_ptr`
    using FFTEngine_ptr = std::unique_ptr<FFTEngine>;
    //! cell coordinates type
    using Ccoord = typename FFTEngine::Ccoord;
    //! spatial coordinates type
    using Rcoord = Rcoord_t<DimS>;
    //! global FieldCollection
    using GFieldCollection_t = typename FFTEngine::GFieldCollection_t;
    //! local FieldCollection (for Fourier-space pixels)
    using LFieldCollection_t = typename FFTEngine::LFieldCollection_t;
    //! Field type on which to apply the projection
    using Field_t = TypedField<GFieldCollection_t, Real>;
    /**
     * iterator over all pixels. This is taken from the FFT engine,
     * because depending on the real-to-complex FFT employed, only
     * roughly half of the pixels are present in Fourier space
     * (because of the hermitian nature of the transform)
     */
    using iterator = typename FFTEngine::iterator;

    //! Default constructor
    ProjectionBase() = delete;

    //! Constructor with cell sizes
    ProjectionBase(FFTEngine_ptr engine, Rcoord domain_lengths,
                   Formulation form);

    //! Copy constructor
    ProjectionBase(const ProjectionBase &other) = delete;

    //! Move constructor
    ProjectionBase(ProjectionBase &&other) = default;

    //! Destructor
    virtual ~ProjectionBase() = default;

    //! Copy assignment operator
    ProjectionBase &operator=(const ProjectionBase &other) = delete;

    //! Move assignment operator
    ProjectionBase &operator=(ProjectionBase &&other) = default;

    //! initialises the fft engine (plan the transform)
    virtual void initialise(FFT_PlanFlags flags = FFT_PlanFlags::estimate);

    //! apply the projection operator to a field
    virtual void apply_projection(Field_t &field) = 0;

    //! returns the process-local resolutions of the cell
    const Ccoord &get_subdomain_resolutions() const {
      return this->fft_engine->get_subdomain_resolutions();
    }
    //! returns the process-local locations of the cell
    const Ccoord &get_subdomain_locations() const {
      return this->fft_engine->get_subdomain_locations();
    }
    //! returns the resolutions of the cell
    const Ccoord &get_domain_resolutions() const {
      return this->fft_engine->get_domain_resolutions();
    }
    //! returns the physical sizes of the cell
    const Rcoord &get_domain_lengths() const { return this->domain_lengths; }

    /**
     * return the `muSpectre::Formulation` that is used in solving
     * this cell. This allows tho check whether a projection is
     * compatible with the chosen formulation
     */
    const Formulation &get_formulation() const { return this->form; }

    //! return the raw projection operator. This is mainly intended
    //! for maintenance and debugging and should never be required in
    //! regular use
    virtual Eigen::Map<Eigen::ArrayXXd> get_operator() = 0;

    //! return the communicator object
    const Communicator &get_communicator() const {
      return this->fft_engine->get_communicator();
    }

    /**
     * returns the number of rows and cols for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetriy storage, it is a column
     * vector)
     */
    virtual std::array<Dim_t, 2> get_strain_shape() const = 0;

    //! get number of components to project per pixel
    virtual Dim_t get_nb_components() const { return DimM * DimM; }

   protected:
    //! handle on the fft_engine used
    FFTEngine_ptr fft_engine;
    const Rcoord domain_lengths;  //!< physical sizes of the cell
    /**
     * formulation this projection can be applied to (determines
     * whether the projection enforces gradients, small strain tensor
     * or symmetric smal strain tensor
     */
    const Formulation form;
    /**
     * A local `muSpectre::FieldCollection` to store the projection
     * operator per k-space point. This is a local rather than a
     * global collection, since the pixels considered depend on the
     * FFT implementation. See
     * http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data
     * for an example
     */
    LFieldCollection_t &projection_container{};

   private:
  };

}  // namespace muSpectre

#endif  // SRC_FFT_PROJECTION_BASE_HH_
