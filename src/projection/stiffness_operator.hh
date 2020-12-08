/**
 * @file   stiffness_operator.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   20 Jul 2020
 *
 * @brief  Class represents the action of the full finite element stiffness
 * matrix
 *
 * Copyright © 2020 Till Junge, Martin Ladecký
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

#include "common/muSpectre_common.hh"
#include "materials/materials_toolbox.hh"

#include <libmugrid/gradient_operator_base.hh>
#include <libmugrid/mapped_field.hh>
#include <libmugrid/exception.hh>

#ifndef SRC_PROJECTION_STIFFNESS_OPERATOR_HH_
#define SRC_PROJECTION_STIFFNESS_OPERATOR_HH_

namespace muSpectre {

  class StiffnessError : public muGrid::RuntimeError {
   public:
    //! constructor
    explicit StiffnessError(const std::string & what)
        : muGrid::RuntimeError(what) {}
    //! constructor
    explicit StiffnessError(const char * what) : muGrid::RuntimeError(what) {}
  };

  class StiffnessOperator {
   public:
    //! Ref to input/output vector
    using EigenVec_t = Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, 1>>;
    //! Ref to input vector
    using EigenCVec_t =
        Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>>;

    //! Default constructor
    StiffnessOperator() = delete;

    /**
     * Constructor from GradientOperator, weights, and (if you're solving a
     * mechanics problem) the formulation, which has to be either of
     * Formulation::finite_strain or Formulation::small_strain.
     */
    StiffnessOperator(
        const Index_t & displacement_rank,
        std::shared_ptr<muGrid::GradientOperatorBase> gradient_operator,
        const std::vector<Real> & quadrature_weights,
        const Formulation & formulation = Formulation::not_set);

    //! Copy constructor
    StiffnessOperator(const StiffnessOperator & other) = delete;

    //! Move constructor
    StiffnessOperator(StiffnessOperator && other) = default;

    //! Destructor
    virtual ~StiffnessOperator() = default;

    //! Copy assignment operator
    StiffnessOperator & operator=(const StiffnessOperator & other) = delete;

    //! Move assignment operator
    StiffnessOperator & operator=(StiffnessOperator && other) = delete;

    /**
     * computes the effect of Ku (in this formulation, K is the Hessian matrix
     * and Ku = -f)
     */
    void apply(const muGrid::TypedFieldBase<Real> & material_properties,
               const muGrid::TypedFieldBase<Real> & displacement,
               muGrid::TypedFieldBase<Real> & force);

    /**
     * computes the effect of Ku (in this formulation, K is the Hessian matrix
     * and Ku = -f)
     */
    void
    apply_increment(const muGrid::TypedFieldBase<Real> & material_properties,
                    const muGrid::TypedFieldBase<Real> & displacement,
                    const Real & alpha, muGrid::TypedFieldBase<Real> & force);

    /**
     * computes the effect of Ku (in this formulation, K is the Hessian matrix
     * and Ku = -f)
     */
    void
    apply_increment(const muGrid::TypedFieldBase<Real> & material_properties,
                    EigenCVec_t displacement, const Real & alpha,
                    EigenVec_t force);

    /**
     * computes the effect of Ku (in this formulation, K is the Hessian matrix
     * and Ku = -f)
     */
    void apply(const Eigen::Ref<const Eigen::MatrixXd> & material_properties,
               const muGrid::TypedFieldBase<Real> & displacement,
               muGrid::TypedFieldBase<Real> & force);

    /**
     * return the raw gradient operator
     */
    std::shared_ptr<muGrid::GradientOperatorBase> get_gradient_operator();

    /**
     * computes the discretised divergence (corresponds to the nodal forces)
     */
    void apply_divergence(
        const muGrid::TypedFieldBase<Real> & quadrature_point_field,
        muGrid::TypedFieldBase<Real> & nodal_field) const;

    /**
     * return the currently set formulation
     */
    const Formulation & get_formulation() const;

   protected:
    void prepare_application(const muGrid::TypedFieldBase<Real> & displacement,
                             const muGrid::TypedFieldBase<Real> & force);

    template <Formulation Form>
    void
    apply_worker(const Eigen::Ref<const Eigen::MatrixXd> & material_properties,
                 const muGrid::TypedFieldBase<Real> & displacement,
                 muGrid::TypedFieldBase<Real> & force);
    /**
     * computes the effect of Ku (in this formulation, K is the Hessian matrix
     * and Ku = -f)
     */
    template <Formulation Form>
    void apply_increment_worker(
        const muGrid::TypedFieldBase<Real> & material_properties,
        const muGrid::TypedFieldBase<Real> & displacement, const Real & alpha,
        muGrid::TypedFieldBase<Real> & force);

    Index_t displacement_rank;
    std::shared_ptr<muGrid::GradientOperatorBase> gradient_operator;
    Index_t nb_displacement_components;
    std::vector<Real> quadrature_weights;
    std::shared_ptr<muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>>>
        quad_field{nullptr};

    Formulation formulation;
  };

  namespace internal {

    template <Formulation Form>
    struct GradientTransformer {
      /**
       * in the default case, there is noting to do
       */
      template <typename T>
      static decltype(auto) transform(T && gradient) {
        return std::forward<T>(gradient);
      }
    };

    /* ---------------------------------------------------------------------- */
    template <>
    struct GradientTransformer<Formulation::small_strain> {
      /**
       * in the small strain case, we return the infinitesimal strain tensor
       */
      template <class Derived>
      static decltype(auto)
      transform(const Eigen::MatrixBase<Derived> & gradient) {
        return MatTB::convert_strain<StrainMeasure::PlacementGradient,
                                     StrainMeasure::Infinitesimal>(gradient);
      }
    };

  }  // namespace internal

}  // namespace muSpectre

#endif  // SRC_PROJECTION_STIFFNESS_OPERATOR_HH_
