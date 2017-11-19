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
#include "stdexcept"

#include "materials/material_base.hh"
#include "materials/materials_toolbox.hh"
#include "common/field_collection.hh"
#include "common/field.hh"


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
    using MFieldCollection_t = typename Parent::MFieldCollection_t;
    using StressMap_t = typename Parent::StressMap_t;
    using StrainMap_t = typename Parent::StrainMap_t;
    using TangentMap_t = typename Parent::TangentMap_t;
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
    virtual void compute_stresses(const StrainMap_t & F,
                                  StressMap_t & P,
                                  Formulation form);
    //! computes stress and tangent modulus
    virtual void compute_stresses_tangent(const StrainMap_t & F,
                                          StressMap_t & P,
                                          TangentMap_t & K,
                                          Formulation form);



  protected:
    //! computes stress with the formulation available at compile time
    template <Formulation Form>
    inline void compute_stresses_worker(const StrainMap_t & F,
                                      StressMap_t & P);
    //! computes stress with the formulation available at compile time
    template <Formulation Form>
    inline void compute_stresses_worker(const StrainMap_t & F,
                                      StressMap_t & P,
                                      TangentMap_t & K);
    //! this iterable class is a default for simple laws that just take a strain
    //! the iterable is just a templated wrapper to provide a range to iterate over
    //! that does or does not include tangent moduli
    template<StrainMeasure strain_measure,
             NeedTangent need_tgt = NeedTangent::no>
    class iterable_proxy;

  private:
  };

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  void MaterialMuSpectre<Material, DimS, DimM>::
  compute_stresses(const StrainMap_t &F, StressMap_t &P,
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
  compute_stresses_tangent(const StrainMap_t & F, StressMap_t & P,
                           TangentMap_t & K,
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
  compute_stresses_worker(const StrainMap_t & F,
                          StressMap_t & P,
                          TangentMap_t & K){

    //! in small strain problems, we always want Cauchy stress as a function
    //! of the infinitesimal strain tensor
    using IterableSmallStrain_t = iterable_proxy<StrainMeasure::Infinitesimal,
                                                 NeedTangent::yes>;
    using IterableFiniteStrain_t = iterable_proxy<Material::strain_measure,
                                                  NeedTangent::yes>;
    using iterable =
      std::conditional_t
      <(Form==Formulation::small_strain),
       IterableSmallStrain_t,
       IterableFiniteStrain_t>;

    constexpr auto stored_strain_m = get_stored_strain_type(Form);
    constexpr auto expected_strain_m = get_formulation_strain_type(Form, Material::strain_measure);

    /* This lambda is executed for every integration point.

       F contains the transformation gradient for finite strain calculations and
       the infinitesimal strain tensor in small strain problems

       The internal_variables tuple contains whatever internal variables
       Material declared (e.g., eigenstrain, strain rate, etc.)
    */
    auto constitutive_law = [this](auto && F, auto && ...  internal_variables) {
      auto && strain = MatTB::convert_strain<stored_strain_m, expected_strain_m>(F);
      // return value contains a tuple of rvalue_refs to both stress and tangent moduli
      auto && stress_tgt = this->evaluate_stress_tangent(std::move(strain), internal_variables...);
      if (Form == Formulation::small_strain) {
        return stress_tgt;
      } else {
        auto && stress = std::get<0>(stress_tgt);
        auto && tangent = std::get<1>(stress_tgt);
        return MatTB::PK1_stress<Material::stress_measure, Material::strain_measure>
        (F, stress, tangent);
      }
    };

    iterable it{this};
    for (auto && tuples: it) {
      // the iterator yields a pair of tuples. this first tuple contains
      // references to stress and stiffness in the global arrays, the second
      // contains references to the deformation gradient and internal variables
      // (some of them are const).
      auto && stress_tgt = std::get<0>(tuples);
      auto && inputs = std::get<1>(tuples);
      stress_tgt = std::apply(constitutive_law, std::move(inputs));
    }
  }

  /* ---------------------------------------------------------------------- */
  //! this iterator class is a default for simple laws that just take a strain
  template<class Material, Dim_t DimS, Dim_t DimM>
  template<StrainMeasure StrainMeasure,
           typename MaterialMuSpectre<Material, DimS, DimM>::NeedTangent NeedTgt>
  class MaterialMuSpectre<Material, DimS, DimM>::iterable_proxy {
  public:
    //! Default constructor
    iterable_proxy() = delete;

    /** Iterator uses the material's internal variables field
        collection to iterate selectively over the global fields
        (such as the transformation gradient F and first
        Piola-Kirchhoff stress P.
    **/
    iterable_proxy(const MaterialMuSpectre & mat);

    static constexpr auto strain_measure{StrainMeasure};
    using Strain_t = typename Material::Strain_t::const_reference;
    using Stress_t = typename Material::Stress_t::reference;
    using Tangent_t = typename Material::Tangent_t::reference;

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
      using value_type =
        std::conditional_t<(NeedTgt == NeedTangent::Yes),
                           std::tuple<Strain_t, Stress_t, Tangent_t>,
                           std::tuple<Strain_t, Stress_t>>;
      using iterator_category = std::forward_iterator_tag;

      //! Default constructor
      iterator() = delete;

      /** Iterator uses the material's internal variables field
          collection to iterate selectively over the global fields
          (such as the transformation gradient F and first
          Piola-Kirchhoff stress P.
      **/
      iterator(const MaterialMuSpectre & mat, bool begin = true);


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
      const MaterialMuSpectre & material;
      size_t index;
    private:
    };

  protected:
    const MaterialMuSpectre & material;
  private:
  };

  /* ---------------------------------------------------------------------- */
  template<class Material, Dim_t DimS, Dim_t DimM>
  template<StrainMeasure strain_measure,
           typename MaterialMuSpectre<Material, DimS, DimM>::NeedTangent Tangent>
  MaterialMuSpectre<Material, DimS, DimM>::template iterable_proxy<strain_measure, Tangent>::
  iterable_proxy(const MaterialMuSpectre<Material, DimS, DimM> & mat)
    : material{mat}
  {}

  /* ---------------------------------------------------------------------- */
  template<class Material, Dim_t DimS, Dim_t DimM>
  template<StrainMeasure strain_measure,
           typename MaterialMuSpectre<Material, DimS, DimM>::NeedTangent Tangent>
  MaterialMuSpectre<Material, DimS, DimM>::iterable_proxy<strain_measure, Tangent>::
  iterator::iterator(const MaterialMuSpectre<Material, DimS, DimM> & mat, bool begin)
    : material{mat}, index{begin ? 0:mat.internal_fields.size()}
  {}

  /* ---------------------------------------------------------------------- */
  //! pre-increment
  template<class Material, Dim_t DimS, Dim_t DimM>
  template<StrainMeasure strain_measure,
           typename MaterialMuSpectre<Material, DimS, DimM>::NeedTangent Tangent>

  typename MaterialMuSpectre<Material, DimS, DimM>::
  template iterable_proxy<strain_measure, Tangent>::iterator &

  MaterialMuSpectre<Material, DimS, DimM>::
  template iterable_proxy<strain_measure, Tangent>::iterator::
  operator++() {
    ++this->index;
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  //! dereference
  template<class Material, Dim_t DimS, Dim_t DimM>
  template<StrainMeasure strain_measure,
           typename MaterialMuSpectre<Material, DimS, DimM>::NeedTangent Tangent>

  typename MaterialMuSpectre<Material, DimS, DimM>::
  template iterable_proxy<strain_measure, Tangent>::iterator::value_type

  MaterialMuSpectre<Material, DimS, DimM>::
  template iterable_proxy<strain_measure, Tangent>::iterator::
  operator*() {
    static_assert(Tangent, "Template specialisation failed");
    return std::make_tuple
      (this->material.internal_fields);
  }

}  // muSpectre


#endif /* MATERIAL_MUSPECTRE_BASE_H */
