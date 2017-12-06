/**
 * file   projection_finite_strain.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Dec 2017
 *
 * @brief  Class for standard finite-strain gradient projections see de Geus et
 *         al. (https://doi.org/10.1016/j.cma.2016.12.032) for derivation
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


#ifndef PROJECTION_FINITE_STRAIN_H
#define PROJECTION_FINITE_STRAIN_H

#include "fft/projection_base.hh"
#include "common/common.hh"
#include "common/field_collection.hh"

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM, class FFT_Engine>
  class ProjectionFiniteStrain: public ProjectionBase<DimS, DimM, FFT_Engine>
  {
  public:
    using Parent = ProjectionBase<DimS, DimM, FFT_Engine>;
    using Ccoord = typename Parent::Ccoord;
    using GFieldCollection_t = FieldCollection<DimS, DimM, true>;
    using LFieldCollection_t = FieldCollection<DimS, DimM, false>;
    using Field_t = TensorField<GFieldCollection_t, Real, secondOrder, DimM>;
    using Proj_t = TensorField<LFieldCollection_t, Real, fourthOrder, DimM>;

    //! Default constructor
    ProjectionFiniteStrain() = delete;

    //! Constructor with system sizes
    ProjectionFiniteStrain(Ccoord sizes);

    //! Copy constructor
    ProjectionFiniteStrain(const ProjectionFiniteStrain &other) = delete;

    //! Move constructor
    ProjectionFiniteStrain(ProjectionFiniteStrain &&other) = default;

    //! Destructor
    virtual ~ProjectionFiniteStrain() noexcept = default;

    //! Copy assignment operator
    ProjectionFiniteStrain& operator=(const ProjectionFiniteStrain &other) = delete;

    //! Move assignment operator
    ProjectionFiniteStrain& operator=(ProjectionFiniteStrain &&other) noexcept = default;

    //! initialises the fft engine (plan the transform)
    void initialise(FFT_PlanFlags flags = FFT_PlanFlags::estimate);

    //! apply the projection operator to a field
    void apply_projection(Field_t & field) const;

  protected:
    Proj_t& Ghat;
  private:
  };

}  // muSpectre

#endif /* PROJECTION_FINITE_STRAIN_H */
