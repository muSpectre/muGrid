/**
 * @file   material_linear_elastic_generic1.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   21 Sep 2018
 *
 * @brief Implementation fo a generic linear elastic material that
 *        stores the full elastic stiffness tensor. Convenient but not the
 *        most efficient
 *
 * Copyright © 2018 Till Junge
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_GENERIC1_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_GENERIC1_HH_

#include "common/muSpectre_common.hh"
#include "materials/stress_transformations_PK2.hh"
#include "materials/material_muSpectre_base.hh"

#include <libmugrid/T4_map_proxy.hh>
#include <libmugrid/field_map_static.hh>

#include <memory>

namespace muSpectre {

  /**
   * forward declaration
   */
  template <Index_t DimM>
  class MaterialLinearElasticGeneric1;

  /**
   * traits for use by MaterialMuSpectre for crtp
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElasticGeneric1<DimM>> {
    //! global field collection

    //! expected map type for strain fields
    using StrainMap_t =
        muGrid::T2FieldMap<Real, Mapping::Const, DimM, IterUnit::SubPt>;
    //! expected map type for stress fields
    using StressMap_t =
        muGrid::T2FieldMap<Real, Mapping::Mut, DimM, IterUnit::SubPt>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t =
        muGrid::T4FieldMap<Real, Mapping::Mut, DimM, IterUnit::SubPt>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};
  };
  /**
   * Linear elastic law defined by a full stiffness tensor. Very
   * generic, but not most efficient. Note: it is template by ImpMaterial to
   * make other materials to inherit form this class without any malfunctioning.
   * i.e. the typeof classes inherits from this class will be passed to
   * MaterialMuSpectre and MAterialMuSpectre will be able to access their types
   * and methods directly without any interference of
   * MaterialLinearElasticGeneric1.
   */
  template <Index_t DimM>
  class MaterialLinearElasticGeneric1
      : public MaterialMuSpectre<MaterialLinearElasticGeneric1<DimM>, DimM> {
   public:
    //! parent type
    using Parent = MaterialMuSpectre<MaterialLinearElasticGeneric1<DimM>, DimM>;
    //! generic input tolerant to python input
    using CInput_t =
        Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>, 0,
                   Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    //! Default constructor
    MaterialLinearElasticGeneric1() = delete;

    /**
     * Constructor by name and stiffness tensor.
     *
     * @param name unique material name
     * @param spatial_dimension spatial dimension of the problem. This
     * corresponds to the dimensionality of the Cell
     * @param nb_quad_pts number of quadrature points per pixel
     * @param C_voigt elastic tensor in Voigt notation
     */
    MaterialLinearElasticGeneric1(const std::string & name,
                                  const Index_t & spatial_dimension,
                                  const Index_t & nb_quad_pts,
                                  const CInput_t & C_voigt);

    //! Copy constructor
    MaterialLinearElasticGeneric1(const MaterialLinearElasticGeneric1 & other) =
        delete;

    //! Move constructor
    MaterialLinearElasticGeneric1(MaterialLinearElasticGeneric1 && other) =
        delete;

    //! Destructor
    virtual ~MaterialLinearElasticGeneric1() = default;

    //! Copy assignment operator
    MaterialLinearElasticGeneric1 &
    operator=(const MaterialLinearElasticGeneric1 & other) = delete;

    //! Move assignment operator
    MaterialLinearElasticGeneric1 &
    operator=(MaterialLinearElasticGeneric1 && other) = delete;

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor). Note: the
     * pixel index is ignored.
     */
    template <class Derived>
    inline decltype(auto) evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                                          const size_t & quad_pt_index = 0);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor). Note: the pixel index is ignored.
     */
    template <class Derived>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const size_t & quad_pt_index = 0);

    /**
     * return a reference to the stiffness tensor
     */
    const muGrid::T4Mat<Real, DimM> & get_C() const { return this->C; }

    template <class Derived1, class Derived2>
    void make_C_from_C_voigt(const Eigen::MatrixBase<Derived1> & C_voigt,
                             Eigen::MatrixBase<Derived2> & C_holder);

   protected:
    // Here, the stiffness tensor is encapsulated into a unique ptr because
    // of this bug:
    // https://eigen.tuxfamily.narkive.com/maHiFSha/fixed-size-vectorizable-members-and-std-make-shared
    // . The problem is that `std::make_shared` uses the global `::new` to
    // allocate `void *` rather than using the the object's `new` operator,
    // and therefore ignores the solution proposed by eigen (documented here
    // http://eigen.tuxfamily.org/dox-devel/group__TopicStructHavingEigenMembers.html).
    // Offloading the offending object into a heap-allocated structure who's
    // construction we control fixes this problem temporarily, until we can
    // use C++17 and guarantee alignment. This comes at the cost of a heap
    // allocation, which is not an issue here, as this happens only once per
    // material and run.
    std::unique_ptr<muGrid::T4Mat<Real, DimM>> C_holder;  //! stiffness
                                                          //! tensor
    const muGrid::T4Mat<Real, DimM> & C;
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialLinearElasticGeneric1<DimM>::evaluate_stress(
      const Eigen::MatrixBase<Derived> & E, const size_t & /*quad_pt_index*/)
      -> decltype(auto) {
    static_assert(Derived::ColsAtCompileTime == DimM, "wrong input size");
    static_assert(Derived::RowsAtCompileTime == DimM, "wrong input size");
    return Matrices::tensmult(this->C, E);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialLinearElasticGeneric1<DimM>::evaluate_stress_tangent(
      const Eigen::MatrixBase<Derived> & E, const size_t & /*quad_pt_index*/)
      -> decltype(auto) {
    using Stress_t = decltype(this->evaluate_stress(E));
    using Stiffness_t = Eigen::Map<const muGrid::T4Mat<Real, DimM>>;
    using Ret_t = std::tuple<Stress_t, Stiffness_t>;
    return Ret_t{this->evaluate_stress(E), Stiffness_t(this->C.data())};
  }
}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_GENERIC1_HH_
