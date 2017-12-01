/**
 * file   material_muSpectre_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   25 Oct 2017
 *
 * @brief  Base class for materials written for µSpectre specifically. These
 *         can take full advantage of the configuration-change utilities of
 *         µSpectre. The user can inherit from them to define new constitutive
 *         laws and is merely required to provide the methods for computing the
 *         second Piola-Kirchhoff stress and Tangent. This class uses the
 *         "curiously recurring template parameter" to avoid virtual calls.
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

#include <tuple>
#include <type_traits>
#include <iterator>
#include <stdexcept>

#include "materials/material_base.hh"
#include "materials/materials_toolbox.hh"
#include "common/field_collection.hh"
#include "common/field.hh"
#include "common//utilities.hh"


#ifndef MATERIAL_MUSPECTRE_BASE_H
#define MATERIAL_MUSPECTRE_BASE_H

namespace muSpectre {

  template <class Material>
  struct MaterialMuSpectre_traits {
  };

  template <class Material, Dim_t DimS, Dim_t DimM>
  class MaterialMuSpectre;

  template <>
  struct MaterialMuSpectre_traits<void> {
    using DefaultInternalVariables = std::tuple<>;
  };

  //! 'Material' is a CRTP
  template <class Material, Dim_t DimS, Dim_t DimM>
  class MaterialMuSpectre: public MaterialBase<DimS, DimM>
  {
  public:
    using NeedTangent = MatTB::NeedTangent;
    using Parent = MaterialBase<DimS, DimM>;
    using GFieldCollection_t = typename Parent::GFieldCollection_t;
    using MFieldCollection_t = typename Parent::MFieldCollection_t;
    using StressField_t = typename Parent::StressField_t;
    using StrainField_t = typename Parent::StrainField_t;
    using TangentField_t = typename Parent::TangentField_t;

    using DefaultInternalVariables = std::tuple<>;
    using traits = MaterialMuSpectre_traits<Material>;

    //! Default constructor
    MaterialMuSpectre() = delete;

    //! Construct by name
    MaterialMuSpectre(std::string name);

    //! Copy constructor
    MaterialMuSpectre(const MaterialMuSpectre &other) = delete;

    //! Move constructor
    MaterialMuSpectre(MaterialMuSpectre &&other) noexcept = delete;

    //! Destructor
    virtual ~MaterialMuSpectre() noexcept = default;

    //! Copy assignment operator
    MaterialMuSpectre& operator=(const MaterialMuSpectre &other) = delete;

    //! Move assignment operator
    MaterialMuSpectre& operator=(MaterialMuSpectre &&other) noexcept = delete;


    //* allocate memory, etc
    virtual void initialise(bool stiffness = false);

    using Parent::compute_stresses;
    using Parent::compute_stresses_tangent;
    //! computes stress
    virtual void compute_stresses(const StrainField_t & F,
                                  StressField_t & P,
                                  Formulation form) override final;
    //! computes stress and tangent modulus
    virtual void compute_stresses_tangent(const StrainField_t & F,
                                          StressField_t & P,
                                          TangentField_t & K,
                                          Formulation form) override final;

  protected:
    //! computes stress with the formulation available at compile time
    template <Formulation Form>
    inline void compute_stresses_worker(const StrainField_t & F,
                                        StressField_t & P);
    //! computes stress with the formulation available at compile time
    template <Formulation Form>
    inline void compute_stresses_worker(const StrainField_t & F,
                                        StressField_t & P,
                                        TangentField_t & K);
    //! this iterable class is a default for simple laws that just take a strain
    //! the iterable is just a templated wrapper to provide a range to iterate over
    //! that does or does not include tangent moduli
    template<NeedTangent need_tgt = NeedTangent::no>
    class iterable_proxy;

    /**
     * inheriting classes with internal variables need to overload this function
     */
    decltype(auto) get_internals() {
      // the default material has no internal variables
      return typename Material::InternalVariables{};}
    decltype(auto) get_internals() const {
      // the default material has no internal variables
      return typename Material::InternalVariables{};}
    typename traits::InternalVariables internal_variables{};

  private:
  };

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  MaterialMuSpectre<Material, DimS, DimM>::
  MaterialMuSpectre(std::string name):Parent(name) {
    using stress_compatible = typename Material::StressMap_t::
      template is_compatible<StressField_t>;
    using strain_compatible = typename Material::StrainMap_t::
      template is_compatible<StrainField_t>;
    using tangent_compatible = typename Material::TangentMap_t::
      template is_compatible<TangentField_t>;

    static_assert((stress_compatible::value &&
                   stress_compatible::explain()),
                  "The material's declared stress map is not compatible "
                  "with the stress field. More info in previously shown "
                  "assert.");

    static_assert((strain_compatible::value &&
                   strain_compatible::explain()),
                  "The material's declared strain map is not compatible "
                  "with the strain field. More info in previously shown "
                  "assert.");

    static_assert((tangent_compatible::value &&
                   tangent_compatible::explain()),
                  "The material's declared tangent map is not compatible "
                  "with the tangent field. More info in previously shown "
                  "assert.");
  }


  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  void MaterialMuSpectre<Material, DimS, DimM>::
  initialise(bool /*stiffness*/) {
    this->internal_fields.initialise();
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  void MaterialMuSpectre<Material, DimS, DimM>::
  compute_stresses(const StrainField_t &F, StressField_t &P,
                 muSpectre::Formulation form) {
    switch (form) {
    case Formulation::finite_strain: {
      this->template compute_stresses_worker<Formulation::finite_strain>(F, P);
      break;
    }
    case Formulation::small_strain: {
      this->template compute_stresses_worker<Formulation::small_strain>(F, P);
      break;
    }
    default:
      throw std::runtime_error("Unknown formulation");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  void MaterialMuSpectre<Material, DimS, DimM>::
  compute_stresses_tangent(const StrainField_t & F, StressField_t & P,
                           TangentField_t & K,
                           muSpectre::Formulation form) {
    switch (form) {
    case Formulation::finite_strain: {
      this->template compute_stresses_worker<Formulation::finite_strain>(F, P, K);
      break;
    }
    case Formulation::small_strain: {
      this->template compute_stresses_worker<Formulation::small_strain>(F, P, K);
      break;
    }
    default:
      throw std::runtime_error("Unknown formulation");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  template <Formulation Form>
  void MaterialMuSpectre<Material, DimS, DimM>::
  compute_stresses_worker(const StrainField_t & F,
                          StressField_t & P,
                          TangentField_t & K){

    /* These lambdas are executed for every integration point.

       F contains the transformation gradient for finite strain calculations and
       the infinitesimal strain tensor in small strain problems

       The internal_variables tuple contains whatever internal variables
       Material declared (e.g., eigenstrain, strain rate, etc.)
    */
    using Strains_t = std::tuple<typename Material::Strain_t>;
    using Stresses_t = std::tuple<typename Material::Stress_t,
                                  typename Material::Tangent_t>;
    auto constitutive_law_small_strain = [this]
      (Strains_t Strains, Stresses_t Stresses, auto && internal_variables) {
      constexpr StrainMeasure stored_strain_m{get_stored_strain_type(Form)};
      constexpr StrainMeasure expected_strain_m{
      get_formulation_strain_type(Form, Material::strain_measure)};

      auto & this_mat = static_cast<Material&>(*this);

      // Transformation gradient is first in the strains tuple
      auto && F = std::get<0>(Strains);
      auto && strain = MatTB::convert_strain<stored_strain_m, expected_strain_m>(F);
      // return value contains a tuple of rvalue_refs to both stress and tangent moduli
      Stresses =
        std::apply([&strain, &this_mat] (auto && ... internals) {
                    return
                    this_mat.evaluate_stress_tangent(std::move(strain),
                                                     internals...);},
                   internal_variables);
    };

    auto constitutive_law_finite_strain = [this]
      (Strains_t Strains, Stresses_t Stresses, auto && internal_variables) {
      constexpr StrainMeasure stored_strain_m{get_stored_strain_type(Form)};
      constexpr StrainMeasure expected_strain_m{
      get_formulation_strain_type(Form, Material::strain_measure)};
      auto & this_mat = static_cast<Material&>(*this);

      // Transformation gradient is first in the strains tuple
      auto && F = std::get<0>(Strains);
      auto && strain = MatTB::convert_strain<stored_strain_m, expected_strain_m>(F);

      // TODO: Figure this out: I can't std::move(internals...),
      // because if there are no internals, compilation fails with "no
      // matching function for call to ‘move()’'. These are tuples of
      // lvalue references, so it shouldn't be too bad, but still
      // irksome.

      // return value contains a tuple of rvalue_refs to both stress
      // and tangent moduli
      auto stress_tgt =
        std::apply([&strain, &this_mat] (auto && ... internals) {
                     return
                     this_mat.evaluate_stress_tangent(std::move(strain),
                                                      internals...);},
                   internal_variables);
      auto && stress = std::get<0>(stress_tgt);
      auto && tangent = std::get<1>(stress_tgt);
      Stresses = MatTB::PK1_stress<Material::stress_measure, Material::strain_measure>
      (std::move(F), std::move(stress), std::move(tangent));
    };

    iterable_proxy<NeedTangent::yes> fields{*this, F, P, K};
    for (auto && arglist: fields) {
      /**
       * arglist is a tuple of three tuples containing only Lvalue
       * references (see value_tye in the class definition of
       * iterable_proxy::iterator). Tuples contain strains, stresses
       * and internal variables, respectively,
       */

      //auto && stress_tgt = std::get<0>(tuples);
      //auto && inputs = std::get<1>(tuples);TODO:clean this
      static_assert(std::is_same<typename Material::Strain_t,
                    std::remove_reference_t<
                    decltype(std::get<0>(std::get<0>(arglist)))>>::value,
                    "Type mismatch for strain reference, check iterator "
                    "value_type");
      static_assert(std::is_same<typename Material::Stress_t,
                    std::remove_reference_t<
                    decltype(std::get<0>(std::get<1>(arglist)))>>::value,
                    "Type mismatch for stress reference, check iterator"
                    "value_type");
      static_assert(std::is_same<typename Material::Tangent_t,
                    std::remove_reference_t<
                    decltype(std::get<1>(std::get<1>(arglist)))>>::value,
                    "Type mismatch for tangent reference, check iterator"
                    "value_type");

      switch (Form) {
      case Formulation::small_strain: {
        std::apply(constitutive_law_small_strain, std::move(arglist));
        break;
      }
      case Formulation::finite_strain: {
        std::apply(constitutive_law_finite_strain, std::move(arglist));
        break;
      }
      }
    }
  }


  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  template <Formulation Form>
  void MaterialMuSpectre<Material, DimS, DimM>::
  compute_stresses_worker(const StrainField_t & F,
                          StressField_t & P){

    /* These lambdas are executed for every integration point.

       F contains the transformation gradient for finite strain calculations and
       the infinitesimal strain tensor in small strain problems

       The internal_variables tuple contains whatever internal variables
       Material declared (e.g., eigenstrain, strain rate, etc.)
    */

    using Strains_t = std::tuple<typename Material::Strain_t>;
    using Stresses_t = std::tuple<typename Material::Stress_t>;
    auto constitutive_law_small_strain = [this]
      (Strains_t Strains, Stresses_t Stresses, auto && internal_variables) {
      constexpr StrainMeasure stored_strain_m{get_stored_strain_type(Form)};
      constexpr StrainMeasure expected_strain_m{
      get_formulation_strain_type(Form, Material::strain_measure)};

      auto & this_mat = static_cast<Material&>(*this);

      // Transformation gradient is first in the strains tuple
      auto && F = std::get<0>(Strains);
      auto && strain = MatTB::convert_strain<stored_strain_m, expected_strain_m>(F);
      // return value contains a tuple of rvalue_refs to both stress and tangent moduli
      auto && sigma = std::get<0>(Stresses);
      sigma =
        std::apply([&strain, &this_mat] (auto && ... internals) {
                    return
                    this_mat.evaluate_stress(std::move(strain),
                                             internals...);},
                   internal_variables);
    };

    auto constitutive_law_finite_strain = [this]
      (Strains_t Strains, Stresses_t && Stresses, auto && internal_variables) {
      constexpr StrainMeasure stored_strain_m{get_stored_strain_type(Form)};
      constexpr StrainMeasure expected_strain_m{
      get_formulation_strain_type(Form, Material::strain_measure)};
      auto & this_mat = static_cast<Material&>(*this);

      // Transformation gradient is first in the strains tuple
      auto && F = std::get<0>(Strains);
      auto && strain = MatTB::convert_strain<stored_strain_m, expected_strain_m>(F);

      // TODO: Figure this out: I can't std::move(internals...),
      // because if there are no internals, compilation fails with "no
      // matching function for call to ‘move()’'. These are tuples of
      // lvalue references, so it shouldn't be too bad, but still
      // irksome.

      // return value contains a tuple of rvalue_refs to both stress
      // and tangent moduli
      auto && stress =
        std::apply([&strain, &this_mat] (auto && ... internals) {
                     return
                     this_mat.evaluate_stress(std::move(strain),
                                              internals...);},
                   internal_variables);
      auto && P = get<0>(Stresses);
      P = MatTB::PK1_stress<Material::stress_measure, Material::strain_measure>
      (F, stress);
    };

    iterable_proxy<NeedTangent::no> fields{*this, F, P};

    for (auto && arglist: fields) {
      /**
       * arglist is a tuple of three tuples containing only Lvalue
       * references (see value_tye in the class definition of
       * iterable_proxy::iterator). Tuples contain strains, stresses
       * and internal variables, respectively,
       */

      //auto && stress_tgt = std::get<0>(tuples);
      //auto && inputs = std::get<1>(tuples);TODO:clean this
      static_assert(std::is_same<typename Material::Strain_t,
                    std::remove_reference_t<
                    decltype(std::get<0>(std::get<0>(arglist)))>>::value,
                    "Type mismatch for strain reference, check iterator "
                    "value_type");
      static_assert(std::is_same<typename Material::Stress_t,
                    std::remove_reference_t<
                    decltype(std::get<0>(std::get<1>(arglist)))>>::value,
                    "Type mismatch for stress reference, check iterator"
                    "value_type");

      switch (Form) {
      case Formulation::small_strain: {
        std::apply(constitutive_law_small_strain, std::move(arglist));
        break;
      }
      case Formulation::finite_strain: {
        std::apply(constitutive_law_finite_strain, std::move(arglist));
        break;
      }
      }
    }
  }


  /* ---------------------------------------------------------------------- */
  //! this iterator class is a default for simple laws that just take a strain
  template <class Material, Dim_t DimS, Dim_t DimM>
  template <MatTB::NeedTangent NeedTgt>
  class MaterialMuSpectre<Material, DimS, DimM>::iterable_proxy {
  public:
    //! Default constructor
    iterable_proxy() = delete;
    using NeedTangent =
      typename MaterialMuSpectre<Material, DimS, DimM>::NeedTangent;
    /** Iterator uses the material's internal variables field
        collection to iterate selectively over the global fields
        (such as the transformation gradient F and first
        Piola-Kirchhoff stress P.
    **/
    template<bool DoNeedTgt=(NeedTgt == NeedTangent::yes)>
    iterable_proxy(const MaterialMuSpectre & mat,
                   const StrainField_t & F,
                   StressField_t & P,
                   std::enable_if_t<DoNeedTgt, TangentField_t> & K)
      :material(mat), strain_field(F), stress_tup{P,K},
       internals(material.internal_variables){};

    template<bool DontNeedTgt=(NeedTgt == NeedTangent::no)>
    iterable_proxy(const MaterialMuSpectre & mat,
                   const StrainField_t & F,
                   std::enable_if_t<DontNeedTgt, StressField_t> & P)
      :material(mat), strain_field(F), stress_tup{P},
       internals(material.internal_variables){};

    using StrainMap_t = typename Material::StrainMap_t;
    using StressMap_t = typename Material::StressMap_t;
    using TangentMap_t = typename Material::TangentMap_t;
    using Strain_t = typename Material::Strain_t;
    using Stress_t = typename Material::Stress_t;
    using Tangent_t = typename Material::Tangent_t;
    using InternalVariables = typename Material::InternalVariables;
    using StressFieldTup = std::conditional_t
      <(NeedTgt == NeedTangent::yes),
       std::tuple<StressField_t&, TangentField_t&>,
       std::tuple<StressField_t&>>;
    using StressMapTup = std::conditional_t
      <(NeedTgt == NeedTangent::yes),
       std::tuple<StressMap_t, TangentMap_t>,
       std::tuple<StressMap_t>>;
    using Stress_tTup = std::conditional_t<(NeedTgt == NeedTangent::yes),
                                           std::tuple<Stress_t, Tangent_t>,
                                           std::tuple<Stress_t>>;


    //! Copy constructor
    iterable_proxy(const iterable_proxy &other) = default;

    //! Move constructor
    iterable_proxy(iterable_proxy &&other) noexcept = default;

    //! Destructor
    virtual ~iterable_proxy() noexcept = default;

    //! Copy assignment operator
    iterable_proxy& operator=(const iterable_proxy &other) = default;

    //! Move assignment operator
    iterable_proxy& operator=(iterable_proxy &&other) noexcept = default;

    class iterator
    {
    public:
      using InternalReferences = MatTB::ReferenceTuple_t<InternalVariables>;
      using value_type =
        std::tuple<std::tuple<Strain_t>, Stress_tTup, InternalReferences>;
      using iterator_category = std::forward_iterator_tag;

      //! Default constructor
      iterator() = delete;

      /** Iterator uses the material's internal variables field
          collection to iterate selectively over the global fields
          (such as the transformation gradient F and first
          Piola-Kirchhoff stress P.
      **/
      iterator(const iterable_proxy & it, bool begin = true)
        : it{it}, strain_map{it.strain_field},
          stress_map {it.stress_tup},
          index{begin ? 0:it.material.internal_fields.size()}{}


      //! Copy constructor
      iterator(const iterator &other) = default;

      //! Move constructor
      iterator(iterator &&other) noexcept = default;

      //! Destructor
      virtual ~iterator() noexcept = default;

      //! Copy assignment operator
      iterator& operator=(const iterator &other) = default;

      //! Move assignment operator
      iterator& operator=(iterator &&other) noexcept = default;

      //! pre-increment
      inline iterator & operator++();
      //! dereference
      inline value_type operator*();
      //! inequality
      inline bool operator!=(const iterator & other) const;


    protected:
      const iterable_proxy & it;
      StrainMap_t strain_map;
      StressMapTup stress_map;
      size_t index;
    private:
    };

    iterator begin() {return std::move(iterator(*this));}
    iterator end() {return std::move(iterator(*this, false));}

  protected:
    const MaterialMuSpectre & material;
    const StrainField_t & strain_field;
    StressFieldTup stress_tup;
    const InternalVariables & internals;

  private:
  };

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  template <MatTB::NeedTangent NeedTgt>
  bool
  MaterialMuSpectre<Material, DimS, DimM>::iterable_proxy<NeedTgt>::iterator::
  operator!=(const iterator & other) const {
    return (this->index != other.index);
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  template <MatTB::NeedTangent NeedTgt>
  typename MaterialMuSpectre<Material, DimS, DimM>::
  template iterable_proxy<NeedTgt>::
  iterator &
  MaterialMuSpectre<Material, DimS, DimM>::iterable_proxy<NeedTgt>::iterator::
  operator++() {
    this->index++;
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  template <MatTB::NeedTangent NeedTgT>
  typename MaterialMuSpectre<Material, DimS, DimM>::
  template iterable_proxy<NeedTgT>::iterator::
  value_type
  MaterialMuSpectre<Material, DimS, DimM>::iterable_proxy<NeedTgT>::iterator::
  operator*() {

    const Ccoord_t<DimS> pixel{
      this->it.material.internal_fields.get_ccoord(this->index)};
    auto && strain = std::make_tuple(this->strain_map[pixel]);

    auto && stresses =
      std::apply([&pixel] (auto && ... stress_tgt) {
          return std::make_tuple(stress_tgt[pixel]...);},
        this->stress_map);
    const auto & internal = this->it.material.get_internals();
    const auto id{index};
    auto && internals =
      std::apply([&internal, id] (auto && ... internals) {
          return std::make_tuple(internals[id]...);},
        internal);
    return std::make_tuple(std::move(strain),
                           std::move(stresses),
                           std::move(internals));
  }

}  // muSpectre


#endif /* MATERIAL_MUSPECTRE_BASE_H */
