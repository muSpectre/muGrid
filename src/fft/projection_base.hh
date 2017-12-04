/**
 * file   projection_base.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Dec 2017
 *
 * @brief  Base class for Projection operators
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

#ifndef PROJECTION_BASE_H
#define PROJECTION_BASE_H

#include "common/common.hh"
#include "common/field_collection.hh"
#include "common/field.hh"
#include "fft/fft_engine_base.hh"

namespace muSpectre {


  template<class Projection>
  struct Projection_traits {
  };

  template <Dim_t DimS, Dim_t DimM, class FFT_Engine>
  class ProjectionBase
  {
  public:
    using Ccoord = FFT_Engine::Ccoord;
    using GFieldCollection_t = FFT_Engine::GFieldCollection_t;
    using Field_t = FFT_Engine::Field_t;
    using iterator = FFT_Engine::iterator;

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

    //! initialises the fft engine (plan the transform)
    void initialise(FFT_PlanFlags flags = FFT_PlanFlags::estimate);

    //! apply the projection operator to a field
    void apply_projection(Field_t & field) const;

    /**
     * get iterator from fft_engine (the number of pixels in the
     * frequency domain depends on the fft implentation)
     */
    iterator begin();
    iterator end();

  protected:
    FFT_Engine fft_engine;
    Ccoord sizes;
    bool initialised{false};

  private:
  };

}  // muSpectre



#endif /* PROJECTION_BASE_H */