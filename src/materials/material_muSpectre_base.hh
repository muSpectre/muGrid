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
#include <boost/range/combine.hpp>

#include "materials/material_base.hh"
#include "materials/materials_toolbox.hh"
#include "common/field_collection.hh"
#include "common/field.hh"
#include "common//utilities.hh"


#ifndef MATERIAL_MUSPECTRE_BASE_H
#define MATERIAL_MUSPECTRE_BASE_H

namespace muSpectre {

  //! 'Material' is a CRTP
  template<class Material, Dim_t DimS, Dim_t DimM>
  class MaterialMuSpectre: public MaterialBase<DimS, DimM>
  {
  public:
    enum class NeedTangent {yes, no};
    using Parent = MaterialBase<DimS, DimM>;
    using GFieldCollection_t = typename Parent::GFieldCollection_t;
    using MFieldCollection_t = typename Parent::MFieldCollection_t;
    using StressField_t = typename Parent::StressField_t;
    using StrainField_t = typename Parent::StrainField_t;
    using TangentField_t = typename Parent::TangentField_t;

    using DefaultInternalVariables = std::tuple<>;

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
    //template<NeedTangent need_tgt = NeedTangent::no>
    //class iterable_proxy; TODO: clean this

    /**
     * get a combined range that can be used to iterate over all fields
     */
    inline decltype(auto) get_zipped_fields(const StrainField_t & F,
                                     StressField_t & P);
    inline decltype(auto) get_zipped_fields(const StrainField_t & F,
                                     StressField_t & P,
                                     TangentField_t & K);
    template <class... Args>
    inline decltype(auto) get_zipped_fields_worker(Args && ... args);

    /**
     * inheriting classes with internal variables need to overload this function
     */
    decltype(auto) get_internals() {
      return typename Material::InternalVariables{};}

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

    //! in small strain problems, we always want Cauchy stress as a function
    //! of the infinitesimal strain tensor
    // using IterableSmallStrain_t = iterable_proxy<NeedTangent::yes>;
    // using IterableFiniteStrain_t = iterable_proxy<NeedTangent::yes>;
    // using iterable =
    //   std::conditional_t
    //   <(Form==Formulation::small_strain),
    //    IterableSmallStrain_t,
    //    IterableFiniteStrain_t>;
    // TODO: CLEAN THIS

    /* These lambdas are executed for every integration point.

       F contains the transformation gradient for finite strain calculations and
       the infinitesimal strain tensor in small strain problems

       The internal_variables tuple contains whatever internal variables
       Material declared (e.g., eigenstrain, strain rate, etc.)
    */
    auto constitutive_law_small_strain = [this]
      (auto && F, auto && P, auto && K, auto && ...  internal_variables) {
      constexpr StrainMeasure stored_strain_m{get_stored_strain_type(Form)};
      constexpr StrainMeasure expected_strain_m{
      get_formulation_strain_type(Form, Material::strain_measure)};
      auto && strain = MatTB::convert_strain<stored_strain_m, expected_strain_m>(F);
      // return value contains a tuple of rvalue_refs to both stress and tangent moduli
      std::tie(P, K) = static_cast<Material&>(*this).evaluate_stress_tangent(std::move(strain), internal_variables...);
    };

    auto constitutive_law_finite_strain = [this]
      (auto && F, auto && P, auto && K, auto && ...  internal_variables) {
      constexpr StrainMeasure stored_strain_m{get_stored_strain_type(Form)};
      constexpr StrainMeasure expected_strain_m{
      get_formulation_strain_type(Form, Material::strain_measure)};

      auto && strain = MatTB::convert_strain<stored_strain_m, expected_strain_m>(F);
      // return value contains a tuple of rvalue_refs to both stress and tangent moduli
      auto && stress_tgt = static_cast<Material&>(*this).evaluate_stress_tangent(std::move(strain), internal_variables...);
      auto && stress = std::get<0>(stress_tgt);
      auto && tangent = std::get<1>(stress_tgt);
      std::tie(P, K) =  MatTB::PK1_stress<Material::stress_measure, Material::strain_measure>
      (F, stress, tangent);
    };

    auto it{this->get_zipped_fields(F, P, K)};
    for (auto && arglist: it) {
      // the iterator yields a pair of tuples. this first tuple contains
      // references to stress and stiffness in the global arrays, the second
      // contains references to the deformation gradient and internal variables
      // (some of them are const).
      //auto && stress_tgt = std::get<0>(tuples);
      //auto && inputs = std::get<1>(tuples);TODO:clean this

      switch (Form) {
      case Formulation::small_strain: {
        std::apply(constitutive_law_small_strain, asStdTuple(arglist));
        break;
      }
      case Formulation::finite_strain: {
        std::apply(constitutive_law_finite_strain, asStdTuple(arglist));
        break;
      }
      }
    }
  }


  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  decltype(auto)
  MaterialMuSpectre<Material, DimS, DimM>::
  get_zipped_fields(const StrainField_t & F, StressField_t & P) {
    typename Material::StrainMap_t Fmap(F);//TODO: think about whether F and P should be std::forward<>()'ed
    typename Material::StressMap_t Pmap(P);

    return this->get_zipped_fields_worker(std::move(Fmap),
                                          std::move(Pmap));
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  decltype(auto)
  MaterialMuSpectre<Material, DimS, DimM>::
  get_zipped_fields(const StrainField_t & F, StressField_t & P,
                    TangentField_t & K) {
    typename Material::StrainMap_t Fmap(F);
    typename Material::StressMap_t Pmap(P);
    typename Material::TangentMap_t Kmap(K);

    return this->get_zipped_fields_worker(std::move(Fmap),
                                          std::move(Pmap),
                                          std::move(Kmap));
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  template <class... Args>
  decltype(auto)
  MaterialMuSpectre<Material, DimS, DimM>::
  get_zipped_fields_worker(Args && ... args) {
    return std::apply
      ([this, &args...] (auto & ... internal_fields) {
        return boost::combine(args..., internal_fields...);},
        this->get_internals());
  }


  //TODO: CLEAN BELOW
  ///* ---------------------------------------------------------------------- */
  ////! this iterator class is a default for simple laws that just take a strain
  //template <class Material, Dim_t DimS, Dim_t DimM>
  //template <typename MaterialMuSpectre<Material, DimS, DimM>::NeedTangent NeedTgt>
  //class MaterialMuSpectre<Material, DimS, DimM>::iterable_proxy {
  //public:
  //  //! Default constructor
  //  iterable_proxy() = delete;
  //  using NeedTangent =
  //    typename MaterialMuSpectre<Material, DimS, DimM>::NeedTangent;
  //  /** Iterator uses the material's internal variables field
  //      collection to iterate selectively over the global fields
  //      (such as the transformation gradient F and first
  //      Piola-Kirchhoff stress P.
  //  **/
  //  template<bool DoNeedTgt=(NeedTgt == NeedTangent::yes)>
  //  iterable_proxy(const MaterialMuSpectre & mat,
  //                 const StrainField_t & F,
  //                 StressField_t & P,
  //                 std::enable_if_t<DoNeedTgt, TangentField_t> & K);
  //
  //  template<bool DontNeedTgt=(NeedTgt == NeedTangent::no)>
  //  iterable_proxy(const MaterialMuSpectre & mat,
  //                 const StrainField_t & F,
  //                 std::enable_if_t<DontNeedTgt, StressField_t> & P);
  //
  //  using Strain_t = typename Material::Strain_t;
  //  using Stress_t = typename Material::Stress_t;
  //  using Tangent_t = typename Material::Tangent_t;
  //
  //  //! Copy constructor
  //  iterable_proxy(const iterable_proxy &other) = default;
  //
  //  //! Move constructor
  //  iterable_proxy(iterable_proxy &&other) noexcept = default;
  //
  //  //! Destructor
  //  virtual ~iterable_proxy() noexcept = default;
  //
  //  //! Copy assignment operator
  //  iterable_proxy& operator=(const iterable_proxy &other) = default;
  //
  //  //! Move assignment operator
  //  iterable_proxy& operator=(iterable_proxy &&other) noexcept = default;
  //
  //  class iterator
  //  {
  //  public:
  //    using value_type =
  //      std::conditional_t
  //      <(NeedTgt == NeedTangent::yes),
  //       std::tuple<std::tuple<Stress_t, Tangent_t>, std::tuple<Strain_t>>,
  //       std::tuple<std::tuple<Stress_t>, std::tuple<Strain_t>>>;
  //    using iterator_category = std::forward_iterator_tag;
  //
  //    //! Default constructor
  //    iterator() = delete;
  //
  //    /** Iterator uses the material's internal variables field
  //        collection to iterate selectively over the global fields
  //        (such as the transformation gradient F and first
  //        Piola-Kirchhoff stress P.
  //    **/
  //    iterator(const iterable_proxy & it, bool begin = true)
  //      : it{it}, index{begin ? 0:it.mat.internal_fields.size()}{}
  //
  //
  //    //! Copy constructor
  //    iterator(const iterator &other) = default;
  //
  //    //! Move constructor
  //    iterator(iterator &&other) noexcept = default;
  //
  //    //! Destructor
  //    virtual ~iterator() noexcept = default;
  //
  //    //! Copy assignment operator
  //    iterator& operator=(const iterator &other) = default;
  //
  //    //! Move assignment operator
  //    iterator& operator=(iterator &&other) noexcept = default;
  //
  //    //! pre-increment
  //    inline iterator & operator++();
  //    //! dereference
  //    inline value_type operator*();
  //    //! inequality
  //    inline bool operator!=(const iterator & other) const;
  //
  //
  //  protected:
  //    const iterable_proxy & it;
  //    size_t index;
  //  private:
  //  };
  //
  //  iterator begin() {return iterator(*this);}
  //  iterator end() {return iterator(*this, false);}
  //
  //protected:
  //  using StressTup = std::conditional_t
  //    <(NeedTgt == NeedTangent::yes),
  //     std::tuple<StressField_t&, TangentField_t&>,
  //     std::tuple<StressField_t&>>;
  //  const MaterialMuSpectre & material;
  //  const StrainField_t & strain_field;
  //  auto ZippedFields;
  //  std::tupleStrainField_t
  //  StressTup stress_tup;
  //private:
  //};
  //
  ///* ---------------------------------------------------------------------- */
  //template <class Material, Dim_t DimS, Dim_t DimM>
  //template <typename MaterialMuSpectre<Material, DimS, DimM>::NeedTangent Tangent>
  //template <bool DoNeedTgt>
  //MaterialMuSpectre<Material, DimS, DimM>::iterable_proxy<Tangent>::
  //iterable_proxy(const MaterialMuSpectre<Material, DimS, DimM> & mat,
  //               const StrainField_t & F,
  //               StressField_t & P,
  //               std::enable_if_t<DoNeedTgt, TangentField_t> & K)
  //  : material{mat}, strain_field{F}, stress_tup(P, K),
  //    ZippedFields{std::apply(boost::combine,
  //                            std::tuple_cat(std::tuple<F>,
  //                                           stress_tup,
  //                                           get_internals()))}
  //{
  //  static_assert((Tangent == NeedTangent::yes),
  //                "Explicitly choosing the template parameter 'DoNeedTgt' "
  //                "breaks the SFINAE intention of thi code. Let it be deduced "
  //                "by the compiler.");
  //}
  //
  ///* ---------------------------------------------------------------------- */
  //template <class Material, Dim_t DimS, Dim_t DimM>
  //template <typename MaterialMuSpectre<Material, DimS, DimM>::NeedTangent Tangent>
  //template <bool DontNeedTgt>
  //MaterialMuSpectre<Material, DimS, DimM>::iterable_proxy<Tangent>::
  //iterable_proxy(const MaterialMuSpectre<Material, DimS, DimM> & mat,
  //               const StrainField_t & F,
  //               std::enable_if_t<DontNeedTgt, StressField_t> & P)
  //  : material{mat}, strain_field{F}, stress_tup(P),
  //    ZippedFields{std::apply(boost::combine,
  //                            std::tuple_cat(std::tuple<F>,
  //                                           stress_tup,
  //                                           get_internals()))}  {
  //  static_assert((Tangent == NeedTangent::no),
  //                "Explicitly choosing the template parameter 'DontNeedTgt' "
  //                "breaks the SFINAE intention of thi code. Let it be deduced "
  //                "by the compiler.");
  //}
  //
  ///* ---------------------------------------------------------------------- */
  ////! pre-increment
  //template<class Material, Dim_t DimS, Dim_t DimM>
  //template<typename MaterialMuSpectre<Material, DimS, DimM>::NeedTangent Tangent>
  //
  //typename MaterialMuSpectre<Material, DimS, DimM>::
  //template iterable_proxy<Tangent>::iterator &
  //
  //MaterialMuSpectre<Material, DimS, DimM>::
  //template iterable_proxy<Tangent>::iterator::
  //operator++() {
  //  ++this->index;
  //  return *this;
  //}
  //
  ///* ---------------------------------------------------------------------- */
  ////! dereference
  //template<class Material, Dim_t DimS, Dim_t DimM>
  //template<typename MaterialMuSpectre<Material, DimS, DimM>::NeedTangent Tangent>
  //
  //typename MaterialMuSpectre<Material, DimS, DimM>::
  //template iterable_proxy<Tangent>::iterator::value_type
  //
  //MaterialMuSpectre<Material, DimS, DimM>::
  //template iterable_proxy<Tangent>::iterator::
  //operator*() {
  //  static_assert(Tangent, "Template specialisation failed");
  //  return std::make_tuple
  //    (this->material.internal_fields);
  //}

}  // muSpectre


#endif /* MATERIAL_MUSPECTRE_BASE_H */
