/**
 * file   field_map_scalar.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   26 Sep 2017
 *
 * @brief  maps over scalar fields
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

#ifndef FIELD_MAP_SCALAR_H
#define FIELD_MAP_SCALAR_H

#include "common/field_map_base.hh"

namespace muSpectre {

  template <class FieldCollection, typename T>
  class ScalarFieldMap
    : public internal::FieldMap<FieldCollection, T, 1>
    {
    public:
      using parent = internal::FieldMap<FieldCollection, T, 1>;
      using Ccoord = Ccoord_t<FieldCollection::spatial_dim()>;
      using value_type = T;
      using reference = value_type &;
      using const_reference = const value_type &;
      using size_type = typename parent::size_type;
      using pointer = T*;
      using Field = typename parent::Field;
      using iterator = typename parent::template iterator<ScalarFieldMap>;
      using const_iterator= typename parent::template iterator<ScalarFieldMap, true>;
      using reverse_iterator = std::reverse_iterator<iterator>;
      using const_reverse_iterator = std::reverse_iterator<const_iterator>;
      friend iterator;

      //! Default constructor
      ScalarFieldMap() = delete;

      ScalarFieldMap(Field & field);

      //! Copy constructor
      ScalarFieldMap(const ScalarFieldMap &other) = delete;

      //! Move constructor
      ScalarFieldMap(ScalarFieldMap &&other) noexcept = delete;

      //! Destructor
      virtual ~ScalarFieldMap() noexcept = default;

      //! Copy assignment operator
      ScalarFieldMap& operator=(const ScalarFieldMap &other) = delete;

      //! Move assignment operator
      ScalarFieldMap& operator=(ScalarFieldMap &&other) noexcept = delete;

      //! give human-readable field map type
      inline std::string info_string() const override final;

      //! member access
      template <class ref_t = reference>
      inline ref_t operator[](size_type index);
      inline reference operator[](const Ccoord&  ccoord);

      //! return an iterator to head of field for ranges
      inline iterator begin(){return iterator(*this);}
      inline const_iterator cbegin(){return const_iterator(*this);}
      //! return an iterator to tail of field for ranges
      inline iterator end(){return iterator(*this, false);}
      inline const_iterator cend(){return const_iterator(*this, false);}

    protected:
      //! for sad, legacy iterator use
      inline pointer ptr_to_val_t(size_type index);

      const static std::string field_info_root;
    private:
  };
  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  ScalarFieldMap<FieldCollection, T>::ScalarFieldMap(Field & field)
    :parent(field) {
    this->check_compatibility();
  }

  /* ---------------------------------------------------------------------- */
  //! human-readable field map type
  template<class FieldCollection, typename T>
  std::string
  ScalarFieldMap<FieldCollection, T>::info_string() const {
    std::stringstream info;
    info << "Scalar(" << typeid(T).name() << ")";
    return info.str();
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  const std::string ScalarFieldMap<FieldCollection, T>::field_info_root{
    "Scalar"};

  /* ---------------------------------------------------------------------- */
  //! member access
  template <class FieldCollection, typename T>
  template <class ref_t>
  ref_t
  ScalarFieldMap<FieldCollection, T>::operator[](size_type index) {
    return this->get_ref_to_entry(std::move(index));
  }

  /* ---------------------------------------------------------------------- */
  //! member access
  template <class FieldCollection, typename T>
  typename ScalarFieldMap<FieldCollection, T>::reference
  ScalarFieldMap<FieldCollection, T>::operator[](const Ccoord& ccoord) {
    auto && index = this->collection.get_index(std::move(ccoord));
    return this->get_ref_to_entry(std::move(index));
  }
}  // muSpectre

#endif /* FIELD_MAP_SCALAR_H */
