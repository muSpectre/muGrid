/**
 * file   projection_finite_strain_fast.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   12 Dec 2017
 *
 * @brief  Faster alternative to ProjectionFinitestrain
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

#ifndef PROJECTION_FINITE_STRAIN_FAST_H
#define PROJECTION_FINITE_STRAIN_FAST_H

#include "fft/projection_base.hh"
#include "common/common.hh"
#include "common/field_collection.hh"
#include "common/field_map.hh"

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM>
  class ProjectionFiniteStrainFast: public ProjectionBase<DimS, DimM>
  {
  public:
    using Parent = ProjectionBase<DimS, DimM>;
    using FFT_Engine = typename Parent::FFT_Engine;
    using Ccoord = typename Parent::Ccoord;
    using GFieldCollection_t = FieldCollection<DimS, DimM, true>;
    using LFieldCollection_t = FieldCollection<DimS, DimM, false>;
    using Field_t = TensorField<GFieldCollection_t, Real, secondOrder, DimM>;
    using Proj_t = TensorField<LFieldCollection_t, Real, firstOrder, DimM>;
    using Proj_map = MatrixFieldMap<LFieldCollection_t, Real, DimM, 1>;
    using Grad_map = MatrixFieldMap<LFieldCollection_t, Complex, DimM, DimM>;



    //! Default constructor
    ProjectionFiniteStrainFast() = delete;

    //! Constructor with fft_engine
    ProjectionFiniteStrainFast(FFT_Engine & engine);

    //! Copy constructor
    ProjectionFiniteStrainFast(const ProjectionFiniteStrainFast &other) = delete;

    //! Move constructor
    ProjectionFiniteStrainFast(ProjectionFiniteStrainFast &&other) noexcept = default;

    //! Destructor
    virtual ~ProjectionFiniteStrainFast() noexcept = default;

    //! Copy assignment operator
    ProjectionFiniteStrainFast& operator=(const ProjectionFiniteStrainFast &other) = delete;

    //! Move assignment operator
    ProjectionFiniteStrainFast& operator=(ProjectionFiniteStrainFast &&other) = default;

    //! initialises the fft engine (plan the transform)
    void initialise(FFT_PlanFlags flags = FFT_PlanFlags::estimate);

    //! apply the projection operator to a field
    void apply_projection(Field_t & field);




  protected:
    Proj_map xis;
  private:
  };

}  // muSpectre

#endif /* PROJECTION_FINITE_STRAIN_FAST_H */
