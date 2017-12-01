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
#include "common/field.hh"

#ifndef FFT_ENGINE_BASE_H
#define FFT_ENGINE_BASE_H


namespace muSpectre {

  enum class FFT_PlanFlags {estimate, measure, patient};

  template<class Projection>
  struct Projection_traits {
  };

  template <Dim_t DimS, Dim_t DimM>
  class ProjectionBase
  {
  public:
    using Ccoord = Ccoord_t<DimS>;
    using GFieldCollection_t = FieldCollection<DimS, DimM, true>;
    using t = TensorField<GFieldCollection_t, Real, 2, DimM>;
    using FFT_out_t = TensorField<GFieldCollection_t, Real, 2, DimM>;
    //! Default constructor
    ProjectionBase() = delete;

    //! Constructor with system sizes
    ProjectionBase(Ccoord sizes);

    //! Copy constructor
    ProjectionBase(const ProjectionBase &other) = delete;

    //! Move constructor
    ProjectionBase(ProjectionBase &&other) noexcept = default;

    //! Destructor
    virtual ~ProjectionBase() noexcept = default;

    //! Copy assignment operator
    ProjectionBase& operator=(const ProjectionBase &other) = delete;

    //! Move assignment operator
    ProjectionBase& operator=(ProjectionBase &&other) noexcept = default;


  protected:
    Ccoord sizes;
  private:
  };

}  // muSpectre


#endif /* FFT_ENGINE_BASE_H */
