/**
 * @file   field_map_dynamic.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Jul 2018
 *
 * @brief  Field map for dynamically sized maps. for use in non-critical
 *         applications (i.e., i/o, postprocessing, etc, but *not* in a hot
 *         loop
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
 * General Public License for more details.
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
 */

#ifndef SRC_COMMON_FIELD_MAP_DYNAMIC_HH_
#define SRC_COMMON_FIELD_MAP_DYNAMIC_HH_

#include "common/field_map_base.hh"

namespace muSpectre {

  /**
   * Maps onto any `muSpectre::TypedField` and lets you iterate over
   * it in the form of `Eigen::Map<Eigen::Array<T, Eigen::Dynamic,
   * 1>`. This is significantly slower than the statically sized field
   * maps and should only be used in non-critical contexts.
   */
  template <class FieldCollection, typename T, bool ConstField = false>
  class TypedFieldMap : public internal::FieldMap<FieldCollection, T,
                                                  Eigen::Dynamic, ConstField> {
   public:
    //! base class
    using Parent =
        internal::FieldMap<FieldCollection, T, Eigen::Dynamic, ConstField>;
    //! sister class with all params equal, but ConstField guaranteed true
    using ConstMap = TypedFieldMap<FieldCollection, T, true>;
    //! cell coordinates type
    using Ccoord = Ccoord_t<FieldCollection::spatial_dim()>;
    //! plain Eigen type
    using Arr_t = Eigen::Array<T, Eigen::Dynamic, 1>;
    using value_type = Eigen::Map<Arr_t>;             //!< stl conformance
    using const_reference = Eigen::Map<const Arr_t>;  //!< stl conformance
    //! stl conformance
    using reference =
        std::conditional_t<ConstField, const_reference,
                           value_type>;  // since it's a resource handle
    using size_type = typename Parent::size_type;  //!< stl conformance
    using pointer = std::unique_ptr<value_type>;   //!< stl conformance

    //! polymorphic base field type (for debug and python)
    using Field = typename Parent::Field;
    //! polymorphic base field type (for debug and python)
    using Field_c = typename Parent::Field_c;
    //! stl conformance
    using const_iterator =
        typename Parent::template iterator<TypedFieldMap, true>;
    //! stl conformance
    using iterator =
        std::conditional_t<ConstField, const_iterator,
                           typename Parent::template iterator<TypedFieldMap>>;
    //! stl conformance
    using reverse_iterator = std::reverse_iterator<iterator>;
    //! stl conformance
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    //! enumerator over a constant scalar field
    using const_enumerator =
        typename Parent::template enumerator<const_iterator>;
    //! enumerator over a scalar field
    using enumerator =
        std::conditional_t<ConstField, const_enumerator,
                           typename Parent::template enumerator<iterator>>;
    //! give access to the protected fields
    friend iterator;

    //! Default constructor
    TypedFieldMap() = delete;

    //! Constructor
    explicit TypedFieldMap(Field_c &field);

    //! Copy constructor
    TypedFieldMap(const TypedFieldMap &other) = delete;

    //! Move constructor
    TypedFieldMap(TypedFieldMap &&other) = default;

    //! Destructor
    virtual ~TypedFieldMap() = default;

    //! Copy assignment operator
    TypedFieldMap &operator=(const TypedFieldMap &other) = delete;

    //! Assign a matrixlike value to every entry
    template <class Derived>
    inline TypedFieldMap &operator=(const Eigen::EigenBase<Derived> &val);

    //! Move assignment operator
    TypedFieldMap &operator=(TypedFieldMap &&other) = default;

    //! give human-readable field map type
    inline std::string info_string() const final;

    //! member access
    inline reference operator[](size_type index);
    //! member access
    inline reference operator[](const Ccoord &ccoord);

    //! member access
    inline const_reference operator[](size_type index) const;
    //! member access
    inline const_reference operator[](const Ccoord &ccoord) const;

    //! return an iterator to first entry of field
    inline iterator begin() { return iterator(*this); }
    //! return an iterator to first entry of field
    inline const_iterator cbegin() const { return const_iterator(*this); }
    //! return an iterator to first entry of field
    inline const_iterator begin() const { return this->cbegin(); }
    //! return an iterator past the last entry of field
    inline iterator end() { return iterator(*this, false); }
    //! return an iterator past the last entry of field
    inline const_iterator cend() const { return const_iterator(*this, false); }
    //! return an iterator past the last entry of field
    inline const_iterator end() const { return this->cend(); }

    /**
     * return an iterable proxy to this field that can be iterated
     * in Ccoord-value tuples
     */
    enumerator enumerate() { return enumerator(*this); }
    /**
     * return an iterable proxy to this field that can be iterated
     * in Ccoord-value tuples
     */
    const_enumerator enumerate() const { return const_enumerator(*this); }

    //! evaluate the average of the field
    inline Arr_t mean() const;

   protected:
    //! for sad, legacy iterator use
    inline pointer ptr_to_val_t(size_type index);

   private:
  };

  //----------------------------------------------------------------------------//
  template <class FieldCollection, typename T, bool ConstField>
  TypedFieldMap<FieldCollection, T, ConstField>::TypedFieldMap(Field_c &field)
      : Parent(field) {
    this->check_compatibility();
  }

  //----------------------------------------------------------------------------//
  template <class FieldCollection, typename T, bool ConstField>
  std::string
  TypedFieldMap<FieldCollection, T, ConstField>::info_string() const {
    std::stringstream info;
    info << "Dynamic(" << typeid(T).name() << ", "
         << this->field.get_nb_components() << ")";
    return info.str();
  }

  //----------------------------------------------------------------------------//
  template <class FieldCollection, typename T, bool ConstField>
  auto TypedFieldMap<FieldCollection, T, ConstField>::
  operator[](size_type index) -> reference {
    return reference{this->get_ptr_to_entry(index),
                     Dim_t(this->field.get_nb_components())};
  }

  //----------------------------------------------------------------------------//
  template <class FieldCollection, typename T, bool ConstField>
  auto TypedFieldMap<FieldCollection, T, ConstField>::
  operator[](const Ccoord &ccoord) -> reference {
    size_t index{this->collection.get_index(ccoord)};
    return (*this)[index];
  }

  //----------------------------------------------------------------------------//
  template <class FieldCollection, typename T, bool ConstField>
  auto TypedFieldMap<FieldCollection, T, ConstField>::
  operator[](size_type index) const -> const_reference {
    return const_reference{this->get_ptr_to_entry(index),
                           Dim_t(this->field.get_nb_components())};
  }

  //----------------------------------------------------------------------------//
  template <class FieldCollection, typename T, bool ConstField>
  auto TypedFieldMap<FieldCollection, T, ConstField>::
  operator[](const Ccoord &ccoord) const -> const_reference {
    size_t index{this->collection.get_index(ccoord)};
    return (*this)[index];
  }

}  // namespace muSpectre

#endif  // SRC_COMMON_FIELD_MAP_DYNAMIC_HH_
