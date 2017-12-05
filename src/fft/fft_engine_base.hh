/**
 * file   fft_engine_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Dec 2017
 *
 * @brief  Interface for FFT engines
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "common/common.hh"
#include "common/field_collection.hh"


#ifndef FFT_ENGINE_BASE_H
#define FFT_ENGINE_BASE_H

namespace muSpectre {

  enum class FFT_PlanFlags {estimate, measure, patient};

  template <Dim_t DimS, Dim_t DimM>
  class FFT_Engine_base
  {
  public:

    using Ccoord = Ccoord_t<DimS>;
    using GFieldCollection_t = FieldCollection<DimS, DimM, true>;
    using LFieldCollection_t = FieldCollection<DimS, DimM, false>;
    using Field_t = TensorField<GFieldCollection_t, Real, 2, DimM>;
    using Workspace_t = TensorField<LFieldCollection_t, Complex, 2, DimM>;
    using iterator = typename LFieldCollection_t::iterator;

    //! Default constructor
    FFT_Engine_base() = delete;

    //! Constructor with system sizes
    FFT_Engine_base(Ccoord sizes);

    //! Copy constructor
    FFT_Engine_base(const FFT_Engine_base &other) = delete;

    //! Move constructor
    FFT_Engine_base(FFT_Engine_base &&other) noexcept = default;

    //! Destructor
    virtual ~FFT_Engine_base() noexcept = default;

    //! Copy assignment operator
    FFT_Engine_base& operator=(const FFT_Engine_base &other) = delete;

    //! Move assignment operator
    FFT_Engine_base& operator=(FFT_Engine_base &&other) noexcept = default;

    // compute the plan, etc
    void initialise(FFT_PlanFlags /*plan_flags*/);

    //! forward transform (dummy for interface)
    Workspace_t & fft(const Field_t & /*field*/);

    //! inverse transform (dummy for interface)
    void ifft(Field_t & /*field*/) const;

    /**
     * iterators over only thos pixels that exist in frequency space
     * (i.e. about half of all pixels, see rfft)
     */
    iterator begin();
    iterator end();

    //! nb of pixels (mostly for debugging)
    size_t size() const;
    size_t workspace_size() const;

  protected:
    LFieldCollection_t work_space_container{};
    const Ccoord sizes;
    Workspace_t & work;
  private:
  };

}  // muSpectre

#endif /* FFT_ENGINE_BASE_H */
