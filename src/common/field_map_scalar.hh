/**
 * file   field_map_scalar.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   26 Sep 2017
 *
 * @brief  maps over scalar fields
 *
 * @section LICENSE
 *
 * Copyright © 2017 Till Junge
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

  template <class FieldCollection, typename T, bool ConstField=false>
  class ScalarFieldMap
    : public internal::FieldMap<FieldCollection, T, 1, ConstField>
    {
    public:
      using parent = internal::FieldMap<FieldCollection, T, 1, ConstField>;
      using Ccoord = Ccoord_t<FieldCollection::spatial_dim()>;
      using value_type = T;
      using const_reference = const value_type &;
      using reference = std::conditional_t<ConstField,
                                           const_reference,
                                           value_type &>;
      using size_type = typename parent::size_type;
      using pointer = T*;
      using Field = typename parent::Field;
      using const_iterator= typename parent::template iterator<ScalarFieldMap, true>;
      using iterator = std::conditional_t
        <ConstField,
         const_iterator,
         typename parent::template iterator<ScalarFieldMap>>;
      using reverse_iterator = std::reverse_iterator<iterator>;
      using const_reverse_iterator = std::reverse_iterator<const_iterator>;
      friend iterator;

      //! Default constructor
      ScalarFieldMap() = delete;

      template <bool isntConst=!ConstField>
      ScalarFieldMap(std::enable_if_t<isntConst, Field &> field);
      template <bool isConst=ConstField>
      ScalarFieldMap(std::enable_if_t<isConst, const Field &> field);

      //! Copy constructor
      ScalarFieldMap(const ScalarFieldMap &other) = default;

      //! Move constructor
      ScalarFieldMap(ScalarFieldMap &&other) = default;

      //! Destructor
      virtual ~ScalarFieldMap() = default;

      //! Copy assignment operator
      ScalarFieldMap& operator=(const ScalarFieldMap &other) = delete;

      //! Move assignment operator
      ScalarFieldMap& operator=(ScalarFieldMap &&other) = delete;

      //! Assign a value to every entry
      ScalarFieldMap& operator=(T val);

      //! give human-readable field map type
      inline std::string info_string() const override final;

      //! member access
      inline reference operator[](size_type index);
      inline reference operator[](const Ccoord&  ccoord);

      inline const_reference operator[] (size_type index) const;
      inline const_reference operator[] (const Ccoord&  ccoord) const;

      //! return an iterator to head of field for ranges
      inline iterator begin(){return iterator(*this);}
      inline const_iterator cbegin() const {return const_iterator(*this);}
      inline const_iterator begin() const {return this->cbegin();}
      //! return an iterator to tail of field for ranges
      inline iterator end(){return iterator(*this, false);}
      inline const_iterator cend() const {return const_iterator(*this, false);}
      inline const_iterator end() const {return this->cend();}

      inline T mean() const;

    protected:
      //! for sad, legacy iterator use
      inline pointer ptr_to_val_t(size_type index);

      const static std::string field_info_root;
    private:
  };

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, bool ConstField>
  template <bool isntConst>
  ScalarFieldMap<FieldCollection, T, ConstField>::
  ScalarFieldMap(std::enable_if_t<isntConst, Field &> field)
    :parent(field) {
    static_assert((isntConst != ConstField),
                  "let the compiler deduce isntConst, this is a SFINAE "
                  "parameter");
    this->check_compatibility();
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, bool ConstField>
  template <bool isConst>
  ScalarFieldMap<FieldCollection, T, ConstField>::
  ScalarFieldMap(std::enable_if_t<isConst, const Field &> field)
    :parent(field) {
    static_assert((isConst == ConstField),
                  "let the compiler deduce isntConst, this is a SFINAE "
                  "parameter");
    this->check_compatibility();
  }

  /* ---------------------------------------------------------------------- */
  //! human-readable field map type
  template<class FieldCollection, typename T, bool ConstField>
  std::string
  ScalarFieldMap<FieldCollection, T, ConstField>::info_string() const {
    std::stringstream info;
    info << "Scalar(" << typeid(T).name() << ")";
    return info.str();
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, bool ConstField>
  const std::string ScalarFieldMap<FieldCollection, T, ConstField>::field_info_root{
    "Scalar"};

  /* ---------------------------------------------------------------------- */
  //! member access
  template <class FieldCollection, typename T, bool ConstField>
  typename ScalarFieldMap<FieldCollection, T, ConstField>::reference
  ScalarFieldMap<FieldCollection, T, ConstField>::operator[](size_type index) {
    return this->get_ptr_to_entry(std::move(index))[0];
  }

  /* ---------------------------------------------------------------------- */
  //! member access
  template <class FieldCollection, typename T, bool ConstField>
  typename ScalarFieldMap<FieldCollection, T, ConstField>::reference
  ScalarFieldMap<FieldCollection, T, ConstField>::operator[](const Ccoord& ccoord) {
    auto && index = this->collection.get_index(std::move(ccoord));
    return this->get_ptr_to_entry(std::move(index))[0];
  }

  /* ---------------------------------------------------------------------- */
  //! member access
  template <class FieldCollection, typename T, bool ConstField>
  typename ScalarFieldMap<FieldCollection, T, ConstField>::const_reference
  ScalarFieldMap<FieldCollection, T, ConstField>::
  operator[](size_type index) const {
    return this->get_ptr_to_entry(std::move(index))[0];
  }

  /* ---------------------------------------------------------------------- */
  //! member access
  template <class FieldCollection, typename T, bool ConstField>
  typename ScalarFieldMap<FieldCollection, T, ConstField>::const_reference
  ScalarFieldMap<FieldCollection, T, ConstField>::
  operator[](const Ccoord& ccoord) const {
    auto && index = this->collection.get_index(std::move(ccoord));
    return this->get_ptr_to_entry(std::move(index))[0];
  }

  /* ---------------------------------------------------------------------- */
  //! Assign a value to every entry
  template <class FieldCollection, typename T, bool ConstField>
  ScalarFieldMap<FieldCollection, T, ConstField> &
  ScalarFieldMap<FieldCollection, T, ConstField>::operator=(T val) {
    for (auto & scalar:*this) {
      scalar = val;
    }
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, bool ConstField>
  T
  ScalarFieldMap<FieldCollection, T, ConstField>::
  mean() const {
    T mean{0};
    for (auto && val: *this) {
      mean += val;
    }
    mean /= Real(this->size());
    return mean;
  }

}  // muSpectre

#endif /* FIELD_MAP_SCALAR_H */
