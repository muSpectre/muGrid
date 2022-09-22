/**
 * @file   projection_gradient.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *         Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   12 Dec 2017
 *
 * @brief  Gradient projection operator. In the case of a mechanics problem,
 *         this is a faster alternative to ProjectionFinitestrain
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

#ifndef SRC_PROJECTION_PROJECTION_GRADIENT_HH_
#define SRC_PROJECTION_PROJECTION_GRADIENT_HH_

#include <libmugrid/field_collection.hh>
#include <libmugrid/mapped_field.hh>

#include <libmufft/derivative.hh>

#include "common/muSpectre_common.hh"
#include "projection/projection_base.hh"

namespace muSpectre {

  namespace internal {

    /**
     * helper struct to determine the gradient map type. Default case will
     * deliberately fail to compile. If you get an error message directing you
     * here, you used an gradient of a rank that was not prepared for here.
     */
    template <Dim_t DimS, Dim_t GradientRank, Index_t NbQuadPts>
    struct GradientMapProvider {};

    /**
     * specialisation for scalar problems (vectorial gradient)
     */
    template <Dim_t DimS, Index_t NbQuadPts>
    struct GradientMapProvider<DimS, firstOrder, NbQuadPts> {
      constexpr static Index_t NbRow() { return 1; }
      constexpr static Index_t NbCol() { return DimS * NbQuadPts; }
      constexpr static Index_t NbPrimitiveRow() { return 1; }
      constexpr static Index_t NbPrimitiveCol() { return 1; }
      using type = muGrid::MatrixFieldMap<Complex, Mapping::Mut, NbRow(),
                                          NbCol(), muGrid::IterUnit::Pixel>;
    };

    /**
     * specialisation for vectorial problems (gradient is second rank tensor)
     */
    template <Dim_t DimS, Index_t NbQuadPts>
    struct GradientMapProvider<DimS, secondOrder, NbQuadPts> {
      constexpr static Index_t NbRow() { return DimS; }
      constexpr static Index_t NbCol() { return DimS * NbQuadPts; }
      constexpr static Index_t NbPrimitiveRow() { return 1; }
      constexpr static Index_t NbPrimitiveCol() { return DimS; }
      using type = muGrid::MatrixFieldMap<Complex, Mapping::Mut, NbRow(),
                                          NbCol(), muGrid::IterUnit::Pixel>;
    };

    // alias for ease of use
    template <Dim_t DimS, Dim_t GradientRank, Index_t NbQuadPts>
    using GradientMapProvider_t =
        typename GradientMapProvider<DimS, GradientRank, NbQuadPts>::type;
  }  // namespace internal
  /**
   * Performs a Helmholtz decomposition of the field and retains only the
   * gradient part. I the case of mechanics, this replaces
   * `muSpectre::ProjectionFiniteStrain` with a faster and less memory-hungry
   * alternative formulation. Use this if you don't have a very good reason not
   * to (and tell me (author) about it, I'd be interested to hear it).
   */
  template <Index_t DimS, Index_t GradientRank, Index_t NbQuadPts = OneQuadPt>
  class ProjectionGradient : public ProjectionBase {
   public:
    using Parent = ProjectionBase;  //!< base class
    //! gradient, i.e. derivatives in each Cartesian direction
    using Gradient_t = Parent::Gradient_t;
    //! weight for each quadrature point
    using Weights_t = typename Parent::Weights_t;
    using Ccoord = Ccoord_t<DimS>;  //!< cell coordinates type
    using Rcoord = Rcoord_t<DimS>;  //!< spatial coordinates type
    //! Real space second order tensor fields (to be projected)
    using Field_t = muGrid::TypedFieldBase<Real>;
    //! Fourier-space field containing the projection operator itself
    using Proj_t = muGrid::ComplexField;
    //! iterable form of the operator
    using Proj_map =
        muGrid::MatrixFieldMap<Complex, Mapping::Mut, DimS * NbQuadPts, 1,
                               muGrid::IterUnit::Pixel>;
    //! iterable Fourier-space second-order tensor field
    using Grad_map =
        internal::GradientMapProvider_t<DimS, GradientRank, NbQuadPts>;

    constexpr static Index_t NbGradRow{
        internal::GradientMapProvider<DimS, GradientRank, NbQuadPts>::NbRow()};
    constexpr static Index_t NbGradCol{
        internal::GradientMapProvider<DimS, GradientRank, NbQuadPts>::NbCol()};
    constexpr static Index_t NbPrimitiveRow{
        internal::GradientMapProvider<DimS, GradientRank,
                                      NbQuadPts>::NbPrimitiveRow()};
    constexpr static Index_t NbPrimitiveCol{
        internal::GradientMapProvider<DimS, GradientRank,
                                      NbQuadPts>::NbPrimitiveCol()};

    // Type of the matrix used as the zero frequency projection (as a mask
    // operator)
    using SingleProj_t =
        Eigen::Matrix<Complex, NbGradRow * NbGradCol, NbGradRow * NbGradCol>;

    // Types defined to cast single field entry to vector to apply a mask
    // projection as the zero frequency projection
    using PixelVec_t = Eigen ::Matrix<Complex, NbGradRow * NbGradCol, 1>;
    using PixelVec_map = Eigen::Map<PixelVec_t>;

    //! Default constructor
    ProjectionGradient() = delete;

    //! Constructor with FFT engine
    ProjectionGradient(
        muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
        const Gradient_t & gradient, const Weights_t & weights,
        const MeanControl & mean_control = MeanControl::StrainControl);

    //! Constructor with FFT engine and default (Fourier) gradient
    ProjectionGradient(
        muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
        const MeanControl & mean_control = MeanControl::StrainControl);

    //! Copy constructor
    ProjectionGradient(const ProjectionGradient & other) = delete;

    //! Move constructor
    ProjectionGradient(ProjectionGradient && other) = default;

    //! Destructor
    virtual ~ProjectionGradient() = default;

    //! Copy assignment operator
    ProjectionGradient & operator=(const ProjectionGradient & other) = delete;

    //! Move assignment operator
    ProjectionGradient & operator=(ProjectionGradient && other) = default;

    //! initialises the fft engine (plan the transform)
    void initialise() final;

    //! apply the projection operator to a field in the case of determined
    //! macroscopic average strain applied on the RVE
    void apply_projection(Field_t & field) final;

    //! compute the positions of the nodes of the pixels.
    //! This function is only applicable in serial.
    Field_t & integrate(Field_t & strain) final;

    //! compute the nonaffine displacements of the nodes of the pixels.
    //! This function is applicable in serial and parallel.
    Field_t & integrate_nonaffine_displacements(Field_t & strain) final;

    Eigen::Map<MatrixXXc> get_operator();

    /**
     * returns the number of rows and cols for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetriy storage, it is a column
     * vector)
     */
    std::array<Index_t, 2> get_strain_shape() const final;

    //! get number of components to project per pixel
    constexpr static Index_t NbComponents() { return DimS * DimS * NbQuadPts; }

    //! get number of components to project per pixel
    virtual Index_t get_nb_dof_per_pixel() const { return NbComponents(); }

    //! perform a deep copy of the projector (this should never be necessary in
    //! c++)
    std::unique_ptr<ProjectionBase> clone() const final;

   protected:
    //! field of projection operators
    muGrid::MappedT1Field<Complex, Mapping::Mut, DimS * NbQuadPts,
                          IterUnit::SubPt>
        proj_field;

    //! field of integration operators
    muGrid::MappedT1Field<Complex, Mapping::Mut, DimS * NbQuadPts,
                          IterUnit::SubPt>
        int_field;

    std::unique_ptr<SingleProj_t> zero_freq_proj_holder{
        std::make_unique<SingleProj_t>(SingleProj_t::Zero())};
    SingleProj_t zero_freq_proj{*zero_freq_proj_holder};
  };

}  // namespace muSpectre

#endif  // SRC_PROJECTION_PROJECTION_GRADIENT_HH_
