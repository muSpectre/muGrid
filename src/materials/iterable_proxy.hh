/**
 * @file   iterable_proxy.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   08 Nov 2019
 *
 * @brief  transitional class for iterating over materials and their strain and
 *         stress fields
 *
 * Copyright © 2019 Till Junge
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

#ifndef SRC_MATERIALS_ITERABLE_PROXY_HH_
#define SRC_MATERIALS_ITERABLE_PROXY_HH_

#include "common/muSpectre_common.hh"

namespace muSpectre {

  namespace internal {

    template <class Dummy>
    struct StressesTComputer {};

    template <class StressMap_t>
    struct StressesTComputer<std::tuple<StressMap_t>> {
      using type = std::tuple<typename StressMap_t::reference>;
    };

    template <class StressMap_t, class TangentMap_t>
    struct StressesTComputer<std::tuple<StressMap_t, TangentMap_t>> {
      using type = std::tuple<typename StressMap_t::reference,
                              typename TangentMap_t::reference>;
    };

    template <class Dummy>
    struct StrainsTComputer {};

    template <class StrainMap_t>
    struct StrainsTComputer<std::tuple<StrainMap_t>> {
      using type = std::tuple<typename StrainMap_t::reference>;
    };

    template <class StrainMap_t>
    struct StrainsTComputer<std::tuple<StrainMap_t, StrainMap_t>> {
      using type = std::tuple<typename StrainMap_t::reference,
                              typename StrainMap_t::reference>;
    };

    template <class OutType>
    struct TupleBuilder {
      template <class... InTypes, size_t... I>
      static OutType helper(std::tuple<InTypes...> const & arg,
                            std::index_sequence<I...>) {
        return OutType(
            typename std::tuple_element<I, OutType>::type(std::get<I>(arg))...);
      }

      template <class... InTypes>
      static OutType build(std::tuple<InTypes...> const & arg) {
        return helper(arg, std::index_sequence_for<InTypes...>{});
      }
    };
  }  // namespace internal
  /* ---------------------------------------------------------------------- */
  //! this iterator class is a default for simple laws that just take a
  //! strain
  template <class Strains_t, class Stresses_t,
            SplitCell IsCellSplit = SplitCell::no>
  class iterable_proxy {
   public:
    //! Default constructor
    iterable_proxy() = delete;

    //! expected type for strain values
    using Strain_t = typename internal::StrainsTComputer<Strains_t>::type;
    //! expected type for stress values
    using Stress_t = typename internal::StressesTComputer<Stresses_t>::type;

    //! tuple containing a strain and possibly a strain-rate field
    using StrainFieldTup = std::conditional_t<
        (std::tuple_size<Strains_t>::value == 2),
        std::tuple<const muGrid::RealField &, const muGrid::RealField &>,
        std::tuple<const muGrid::RealField &>>;

    //! tuple containing a stress and possibly a tangent stiffness field
    using StressFieldTup =
        std::conditional_t<(std::tuple_size<Stresses_t>::value == 2),
                           std::tuple<muGrid::RealField &, muGrid::RealField &>,
                           std::tuple<muGrid::RealField &>>;

    /** Iterator uses the material's internal variables field
        collection to iterate selectively over the global fields
        (such as the transformation gradient F and first
        Piola-Kirchhoff stress P.
    **/
    // ! Constructors
    // with tangent and with strain rate
    template <bool DoNeedTgt = std::tuple_size<Stresses_t>::value == 2,
              bool DoNeedRate = std::tuple_size<Strain_t>::value == 2>
    iterable_proxy(
        MaterialBase & mat, const muGrid::RealField & F,
        std::enable_if_t<DoNeedRate, const muGrid::RealField> & F_rate,
        muGrid::RealField & P,
        std::enable_if_t<DoNeedTgt, muGrid::RealField> & K)
        : material{mat}, strain_field{std::cref(F), std::cref(F_rate)},
          stress_tup{P, K} {};

    // without tangent and with strain rate
    template <bool DontNeedTgt = std::tuple_size<Stresses_t>::value == 1,
              bool DoNeedRate = std::tuple_size<Strain_t>::value == 2>
    iterable_proxy(
        MaterialBase & mat, const muGrid::RealField & F,
        std::enable_if_t<DoNeedRate, const muGrid::RealField> & F_rate,
        std::enable_if_t<DontNeedTgt, muGrid::RealField> & P)
        : material{mat}, strain_field{std::cref(F), std::cref(F_rate)},
          stress_tup{P} {};

    // with tangent and without strain rate
    template <bool DoNeedTgt = std::tuple_size<Stresses_t>::value == 2,
              bool DontNeedRate = std::tuple_size<Strain_t>::value == 1>
    iterable_proxy(MaterialBase & mat,
                   std::enable_if_t<DontNeedRate, const muGrid::RealField> & F,
                   muGrid::RealField & P,
                   std::enable_if_t<DoNeedTgt, muGrid::RealField> & K)
        : material{mat}, strain_field{std::cref(F)}, stress_tup{P, K} {};

    // without tangent and without strain rate
    template <bool DontNeedTgt = std::tuple_size<Stresses_t>::value == 1,
              bool DontNeedRate = std::tuple_size<Strain_t>::value == 1>
    iterable_proxy(MaterialBase & mat,
                   std::enable_if_t<DontNeedRate, const muGrid::RealField> & F,
                   std::enable_if_t<DontNeedTgt, muGrid::RealField> & P)
        : material{mat}, strain_field{std::cref(F)}, stress_tup{P} {};

    //! Copy constructor
    iterable_proxy(const iterable_proxy & other) = default;

    //! Move constructor
    iterable_proxy(iterable_proxy && other) = default;

    //! Destructor
    virtual ~iterable_proxy() = default;

    //! Copy assignment operator
    iterable_proxy & operator=(const iterable_proxy & other) = default;

    //! Move assignment operator
    iterable_proxy & operator=(iterable_proxy && other) = default;

    /**
     * dereferences into a tuple containing strains, and internal
     * variables, as well as maps to the stress and potentially
     * stiffness maps where to write the response of a pixel
     */
    class iterator {
     public:
      //! return type contains a tuple of strain and possibly strain rate,
      //! stress and possibly stiffness, and a refererence to the pixel
      //! index
      using value_type = std::tuple<Strain_t, Stress_t, const size_t &, Real>;
      using iterator_category = std::forward_iterator_tag;  //!< stl conformance

      //! Default constructor
      iterator() = delete;

      /** Iterator uses the material's internal variables field
          collection to iterate selectively over the global fields
          (such as the transformation gradient F and first
          Piola-Kirchhoff stress P.
      **/
      explicit iterator(const iterable_proxy & proxy, bool begin = true)
          : proxy{proxy}, strain_map{internal::TupleBuilder<Strains_t>::build(
                              std::remove_cv_t<StrainFieldTup>(
                                  proxy.strain_field))},
            stress_map{internal::TupleBuilder<Stresses_t>::build(
                std::remove_cv_t<StressFieldTup>(proxy.stress_tup))},
            index{begin ? 0 : size_t(proxy.material.size())},
            quad_pt_iter{begin ? proxy.material.get_collection()
                                     .get_sub_pt_indices(QuadPtTag)
                                     .begin()
                               : proxy.material.get_collection()
                                     .get_sub_pt_indices(QuadPtTag)
                                     .end()} {}

      //! Copy constructor
      iterator(const iterator & other) = default;

      //! Move constructor
      iterator(iterator && other) = default;

      //! Destructor
      virtual ~iterator() = default;

      //! Copy assignment operator
      iterator & operator=(const iterator & other) = default;

      //! Move assignment operator
      iterator & operator=(iterator && other) = default;

      //! pre-increment
      inline iterator & operator++();
      //! dereference
      inline value_type operator*();
      //! inequality
      inline bool operator!=(const iterator & other) const;

     protected:
      const iterable_proxy & proxy;  //!< ref to the proxy
      Strains_t strain_map;          //!< map onto the global strain field
      //! map onto the global stress field and possibly tangent stiffness
      Stresses_t stress_map;
      /**
       * counter of current iterate (quad point). This value is the look-up
       * index for the local field collection
       */
      size_t index;
      //! iterator over quadrature point. This value is the look-up index for
      //! the global field collection
      muGrid::FieldCollection::IndexIterable::iterator quad_pt_iter;
    };

    //! returns iterator to first pixel if this material
    iterator begin() { return iterator(*this); }
    //! returns iterator past the last pixel in this material
    iterator end() { return iterator(*this, false); }

   protected:
    MaterialBase & material;      //!< reference to the proxied material
    StrainFieldTup strain_field;  //!< cell's global strain field
    //! references to the global stress field and perhaps tangent
    StressFieldTup stress_tup;
  };

  /* ---------------------------------------------------------------------- */
  template <class Strains_t, class Stresses_t, SplitCell IsCellSplit>
  bool iterable_proxy<Strains_t, Stresses_t, IsCellSplit>::iterator::operator!=(
      const iterator & other) const {
    return (this->index != other.index);
  }

  /* ---------------------------------------------------------------------- */
  template <class Strains_t, class Stresses_t, SplitCell IsCellSplit>
  auto
  iterable_proxy<Strains_t, Stresses_t, IsCellSplit>::iterator::operator++()
      -> iterator & {
    ++this->index;
    ++this->quad_pt_iter;
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <class Strains_t, class Stresses_t, SplitCell IsCellSplit>
  auto iterable_proxy<Strains_t, Stresses_t, IsCellSplit>::iterator::operator*()
      -> value_type {
    const auto & quad_pt_id{*this->quad_pt_iter};

    auto && strains = apply(
        [&quad_pt_id](auto &&... strain_and_rate) {
          return std::make_tuple(strain_and_rate[quad_pt_id]...);
        },
        this->strain_map);

    auto && ratio = 1.0;
    if (IsCellSplit != SplitCell::no) {
      ratio = this->proxy.material.get_assigned_ratio(quad_pt_id);
    }

    auto && stresses = apply(
        [&quad_pt_id](auto &&... stress_tgt) {
          return std::make_tuple(stress_tgt[quad_pt_id]...);
        },
        this->stress_map);
    return value_type(std::move(strains), std::move(stresses), this->index,
                      ratio);
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_ITERABLE_PROXY_HH_
