/**
 * file   projection_base.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Dec 2017
 *
 * @brief  Base class for Projection operators
 *
 * @section LICENSE
 *
 * Copyright © 2017 Till Junge
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

#include <memory>

namespace muSpectre {

  template<class Projection>
  struct Projection_traits {
  };

  template <Dim_t DimS, Dim_t DimM>
  class ProjectionBase
  {
  public:
    using FFT_Engine = FFT_Engine_base<DimS, DimM>;
    using FFT_Engine_ptr = std::unique_ptr<FFT_Engine>;
    using Ccoord = typename FFT_Engine::Ccoord;
    using Rcoord = typename FFT_Engine::Rcoord;
    using GFieldCollection_t = typename FFT_Engine::GFieldCollection_t;
    using LFieldCollection_t = typename FFT_Engine::LFieldCollection_t;
    using Field_t = typename FFT_Engine::Field_t;
    using iterator = typename FFT_Engine::iterator;

    //! Default constructor
    ProjectionBase() = delete;

    //! Constructor with system sizes
    ProjectionBase(FFT_Engine_ptr engine, Formulation form);

    //! Copy constructor
    ProjectionBase(const ProjectionBase &other) = delete;

    //! Move constructor
    ProjectionBase(ProjectionBase &&other) = default;

    //! Destructor
    virtual ~ProjectionBase() = default;

    //! Copy assignment operator
    ProjectionBase& operator=(const ProjectionBase &other) = delete;

    //! Move assignment operator
    ProjectionBase& operator=(ProjectionBase &&other) = default;

    //! initialises the fft engine (plan the transform)
    virtual void initialise(FFT_PlanFlags flags = FFT_PlanFlags::estimate);

    //! apply the projection operator to a field
    virtual void apply_projection(Field_t & field) = 0;

    //!
    const Ccoord & get_resolutions() const {
      return this->fft_engine->get_resolutions();}
    const Rcoord & get_lengths() const {
      return this->fft_engine->get_lengths();}

    const Formulation & get_formulation() const {return this->form;}

    //! return the raw projection operator. This is mainly intended
    //! for maintenance and debugging and should never be required in
    //! regular use
    virtual Eigen::Map<Eigen::ArrayXXd> get_operator() = 0;

  protected:
    FFT_Engine_ptr fft_engine;
    const Formulation form;
    LFieldCollection_t & projection_container{};

  private:
  };

}  // muSpectre



#endif /* PROJECTION_BASE_H */
