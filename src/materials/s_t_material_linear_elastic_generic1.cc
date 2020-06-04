/**
 * @file   s_t_material_linear_elastic_generic1.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   20 Jan 2020
 *
 * @brief  the implemenation of the methods of the class
 * STMateriallinearelasticgeneric1
 *
 * Copyright © 2020 Ali Falsafi
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

#include "materials/s_t_material_linear_elastic_generic1.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM, StrainMeasure StrainM, StressMeasure StressM>
  STMaterialLinearElasticGeneric1<DimM, StrainM, StressM>::
      STMaterialLinearElasticGeneric1(const std::string & name,
                                      const Index_t & spatial_dimension,
                                      const Index_t & nb_quad_pts,
                                      const CInput_t & C_voigt)
      : Parent{name, spatial_dimension, nb_quad_pts},
        C_holder{std::make_unique<Stiffness_t>()}, C{*this->C_holder},
        F_holder{std::make_unique<Strain_t>(Strain_t::Identity())},
        F{*this->F_holder}, F_is_set{false} {
    MatTB::make_C_from_C_voigt<DimM>(C_voigt, *this->C_holder);
    this->last_step_was_nonlinear = false;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM, StrainMeasure StrainM, StressMeasure StressM>
  auto STMaterialLinearElasticGeneric1<DimM, StrainM, StressM>::make_evaluator(
      const CInput_t & C_voigt)
      -> std::tuple<Material_sptr, MaterialEvaluator<DimM>> {
    constexpr Index_t SpatialDimension{DimM};
    constexpr Index_t NbQuadPts{1};
    auto mat{std::make_shared<STMaterialLinearElasticGeneric1>(
        "name", SpatialDimension, NbQuadPts, C_voigt)};

    using Ret_t = std::tuple<Material_sptr, MaterialEvaluator<DimM>>;
    return Ret_t(mat, MaterialEvaluator<DimM>{mat});
  }
  /* ---------------------------------------------------------------------- */

  //! Green-Lagrange strain and PK2 Stress
  template class STMaterialLinearElasticGeneric1<
      twoD, StrainMeasure::GreenLagrange, StressMeasure::PK2>;
  template class STMaterialLinearElasticGeneric1<
      threeD, StrainMeasure::GreenLagrange, StressMeasure::PK2>;

  /* ---------------------------------------------------------------------- */
  //! Gradient strain and PK1 Stress
  template class STMaterialLinearElasticGeneric1<twoD, StrainMeasure::Gradient,
                                                 StressMeasure::PK1>;
  template class STMaterialLinearElasticGeneric1<
      threeD, StrainMeasure::Gradient, StressMeasure::PK1>;

  /* ---------------------------------------------------------------------- */
  //! Gradient strain and Kirchhoff Stress
  template class STMaterialLinearElasticGeneric1<twoD, StrainMeasure::Gradient,
                                                 StressMeasure::Kirchhoff>;
  template class STMaterialLinearElasticGeneric1<
      threeD, StrainMeasure::Gradient, StressMeasure::Kirchhoff>;

  /* ---------------------------------------------------------------------- */
  //! Green-Lagrange strain and Kirchhoff Stress
  template class STMaterialLinearElasticGeneric1<
      twoD, StrainMeasure::GreenLagrange, StressMeasure::Kirchhoff>;
  template class STMaterialLinearElasticGeneric1<
      threeD, StrainMeasure::GreenLagrange, StressMeasure::Kirchhoff>;

}  // namespace muSpectre
