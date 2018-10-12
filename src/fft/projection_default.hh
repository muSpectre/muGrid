/**
 * @file   projection_default.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   14 Jan 2018
 *
 * @brief  virtual base class for default projection implementation, where the
 *         projection operator is stored as a full fourth-order tensor per
 *         k-space point (as opposed to 'smart' faster implementations, such as
 *         ProjectionFiniteStrainFast
 *
 * Copyright (C) 2018 Till Junge
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
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef PROJECTION_DEFAULT_H
#define PROJECTION_DEFAULT_H

#include "fft/projection_base.hh"

namespace muSpectre {

  /**
   * base class to inherit from if one implements a projection
   * operator that is stored in form of a fourth-order tensor of real
   * values per k-grid point
   */
  template <Dim_t DimS, Dim_t DimM>
  class ProjectionDefault: public ProjectionBase<DimS, DimM>
  {
  public:
    using Parent = ProjectionBase<DimS, DimM>; //!< base class
    using Vector_t = typename Parent::Vector_t; //!< to represent fields
    //! polymorphic FFT pointer type
    using FFTEngine_ptr = typename Parent::FFTEngine_ptr;
    using Ccoord = typename Parent::Ccoord; //!< cell coordinates type
    using Rcoord = typename Parent::Rcoord; //!< spatial coordinates type
    //! global field collection
    using GFieldCollection_t = GlobalFieldCollection<DimS>;
    //! local field collection for Fourier-space fields
    using LFieldCollection_t = LocalFieldCollection<DimS>;
    //! Real space second order tensor fields (to be projected)
    using Field_t = TypedField<GFieldCollection_t, Real>;
    //! Fourier-space field containing the projection operator itself
    using Proj_t = TensorField<LFieldCollection_t, Real, fourthOrder, DimM>;
    //! iterable form of the operator
    using Proj_map = T4MatrixFieldMap<LFieldCollection_t, Real, DimM>;
    //! vectorized version of the Fourier-space second-order tensor field
    using Vector_map = MatrixFieldMap<LFieldCollection_t, Complex, DimM*DimM, 1>;
    //! Default constructor
    ProjectionDefault() = delete;

    //! Constructor with cell sizes and formulation
    ProjectionDefault(FFTEngine_ptr engine, Rcoord lengths, Formulation form);

    //! Copy constructor
    ProjectionDefault(const ProjectionDefault &other) = delete;

    //! Move constructor
    ProjectionDefault(ProjectionDefault &&other) = default;

    //! Destructor
    virtual ~ProjectionDefault() = default;

    //! Copy assignment operator
    ProjectionDefault& operator=(const ProjectionDefault &other) = delete;

    //! Move assignment operator
    ProjectionDefault& operator=(ProjectionDefault &&other) = delete;

    //! apply the projection operator to a field
    void apply_projection(Field_t & field) override final;

    Eigen::Map<Eigen::ArrayXXd> get_operator() override final;

    /**
     * returns the number of rows and cols for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetriy storage, it is a column
     * vector)
     */
    std::array<Dim_t, 2> get_strain_shape() const override final;

    constexpr static Dim_t NbComponents() {return ipow(DimM, 2);}

  protected:
    Proj_t & Gfield; //!< field holding the operator
    Proj_map Ghat;   //!< iterable version of operator
  private:
  };

}  // muSpectre

#endif /* PROJECTION_DEFAULT_H */
