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
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#ifndef SRC_PROJECTION_PROJECTION_DEFAULT_HH_
#define SRC_PROJECTION_PROJECTION_DEFAULT_HH_

#include <libmugrid/field_map_static.hh>

#include <libmufft/derivative.hh>

#include "projection/projection_base.hh"

namespace muSpectre {

  /**
   * base class to inherit from if one implements a projection
   * operator that is stored in form of a fourth-order tensor of real
   * values per k-grid point
   */
  template <Index_t DimS>
  class ProjectionDefault : public ProjectionBase {
   public:
    using Parent = ProjectionBase;               //!< base class
    using Vector_t = typename Parent::Vector_t;  //!< to represent fields
    //! gradient, i.e. derivatives in each Cartesian direction
    using Gradient_t = muFFT::Gradient_t;
    using Ccoord = Ccoord_t<DimS>;  //!< cell coordinates type
    using Rcoord = Rcoord_t<DimS>;  //!< spatial coordinates type
    //! global field collection
    using GFieldCollection_t = muGrid::GlobalFieldCollection;
    //! Real space second order tensor fields (to be projected)
    using Field_t = muGrid::TypedFieldBase<Real>;
    //! fourier-space field containing the projection operator itself
    using Proj_t = muGrid::ComplexField;
    //! iterable form of the operator
    using Proj_map =
        muGrid::T4FieldMap<Complex, Mapping::Mut, DimS, IterUnit::SubPt>;
    //! vectorized version of the Fourier-space second-order tensor field
    using Vector_map =
        muGrid::MatrixFieldMap<Complex, Mapping::Mut, DimS * DimS, 1,
                               IterUnit::SubPt>;
    //! Default constructor
    ProjectionDefault() = delete;

    //! Constructor with cell sizes and formulation
    ProjectionDefault(muFFT::FFTEngine_ptr engine, DynRcoord_t lengths,
                      Gradient_t gradient, Formulation form);

    //! Copy constructor
    ProjectionDefault(const ProjectionDefault & other) = delete;

    //! Move constructor
    ProjectionDefault(ProjectionDefault && other) = default;

    //! Destructor
    virtual ~ProjectionDefault() = default;

    //! Copy assignment operator
    ProjectionDefault & operator=(const ProjectionDefault & other) = delete;

    //! Move assignment operator
    ProjectionDefault & operator=(ProjectionDefault && other) = delete;

    //! apply the projection operator to a field
    void apply_projection(Field_t & field) final;

    Eigen::Map<MatrixXXc> get_operator();

    /**
     * returns the number of rows and cols for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetriy storage, it is a column
     * vector)
     */
    std::array<Index_t, 2> get_strain_shape() const final;

    //! get number of components to project per pixel
    constexpr static Index_t NbComponents() { return DimS * DimS; }

    //! get number of components to project per pixel
    virtual Index_t get_nb_dof_per_pixel() const { return NbComponents(); }

    const Gradient_t & get_gradient() const { return this->gradient; }

   protected:
    Proj_t & Gfield;  //!< field holding the operator
    Proj_map Ghat;    //!< iterable version of operator
    /**
     * gradient (nabla) operator, can be computed using Fourier interpolation
     * or through a weighted residual
     */
    Gradient_t gradient;
  };

}  // namespace muSpectre

#endif  // SRC_PROJECTION_PROJECTION_DEFAULT_HH_
