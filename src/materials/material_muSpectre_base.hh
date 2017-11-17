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
 *         second Piola-Kirchhoff stress and Stiffness. This class uses the
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

#include "materials/material.hh"
#include "materials/materials_toolbox.hh"
#include "common/field_collection.hh"
#include "common/field.hh"


#ifndef MATERIAL_MUSPECTRE_BASE_H
#define MATERIAL_MUSPECTRE_BASE_H

namespace muSpectre {

  //! 'Material' is a CRTP
  template<class Material>
  class MaterialMuSpectre: public MaterialBase<Material::sdim(), Material::mdim()>
  {
  public:
    using Parent = public MaterialBase<Material::sdim(), Material::mdim()>;
    using MFieldCollection_t = Parent::MFieldCollection_t;
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
    //! computes stress and tangent stiffness
    virtual void compute_stresses_stiffness(const StrainMap_t & F,
                                          StressMap_t & P,
                                          StiffnessMap_t & K,
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
                                      StiffnessMap_t & K);
    //! this iterable class is a default for simple laws that just take a strain
    //! the iterable is just a templated wrapper to provide a range to iterate over
    //! that does or does not include stiffness
    template<MatTB::StrainMeasure strain_measure,
             bool Stiffness = false>
    class iterable_proxy;

  private:
  };

  /* ---------------------------------------------------------------------- */
  template <class Material>
  void MaterialMuSpectre<Material>::
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
  template <class Material>
  void MaterialMuSpectre<Material>::
  compute_stresses_stiffness(const StrainMap_t & F, StressMap_t & P,
                           StiffnessMap_t & K
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
  template <class Material>
  template <Formulation Form>
  void MaterialMuSpectre<Material>::
  compute_stresses_worker(const StrainMap_t & F,
                        StressMap_t & P,
                        StiffnessMap_t & K){

    //! in small strain problems, we always want Cauchy stress as a function
    //! of the infinitesimal strain tensor
    using iterable =
      std::conditional_t
      <(Form==Formulation::small_strain),
       Material::iterable_proxy<StrainMeasure::Infinitesimal, true>,
       Material::iterable_proxy<Material::strain_measure, true>>;

    /* This lambda is executed for every intergration point. 

       The input_tuple contains strain (in the appropriate strain
       measure), and stress and potentially tangent moduli in the
       appropriate stress measure.

       The internals tuple is potentially empty (for laws without
       internal variables, such as e.g., simple hyperelastic
       materials), else it contains internal variables in the
       appropriate form, as specified in the iterable
    */
    auto fun = [this](auto && input_args...) {
      
      if (Material::uniform_stiffness) {
        
      }
    };
  }

  /* ---------------------------------------------------------------------- */
  //! this iterator class is a default for simple laws that just take a strain
  template<class Material>
  template<MatTB::StrainMeasure strain_measure,
           bool Stiffness = false>
  class MaterialMuSpectre<Material>::iterable_proxy {
  public:
    using Strain_t = typename Material::Strain_t::const_reference;
    using Stress_t = typename Material::Stress_t::reference;
    using Stiffness_t = typename Material::Stiffness_t::reference;
    using value_type =
      std::conditional_t<Stiffness,
                         std::tuple<Strain_t, Stress_t, Stiffness_t>,
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

  /* ---------------------------------------------------------------------- */
  template<class Material>
  template<MatTB::StrainMeasure strain_measure, bool Stiffness>
  MaterialMuSpectre<Material>::iterator<strain_measure, Stiffness>::
  iterator(const MaterialMuSpectre<Material> & mat, bool begin)
    : material{mat}, index{begin ? 0:mat.internal_fields.size()}
  {}

  /* ---------------------------------------------------------------------- */
  //! pre-increment
  template<class Material>
  template<MatTB::StrainMeasure strain_measure, bool Stiffness>
  typename MaterialMuSpectre<Material>::iterator<strain_measure, Stiffness> &
  MaterialMuSpectre<Material>::iterator<strain_measure, Stiffness>::
  operator++() {
    ++this->index;
  }

  /* ---------------------------------------------------------------------- */
  //! dereference
  template<class Material>
  template<MatTB::StrainMeasure strain_measure, bool Stiffness>
  typename MaterialMuSpectre<Material>::iterator<strain_measure, Stiffness>::value_type
  MaterialMuSpectre<Material>::iterator<strain_measure, Stiffness>::
  operator*() {
    static_assert(Stiffness, "Template specialisation failed");
    return std::make_tuple
      (this->material.internal_fields)
  }

}  // muSpectre


#endif /* MATERIAL_MUSPECTRE_BASE_H */
