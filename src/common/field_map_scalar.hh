/**
 * @file   field_map_scalar.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   26 Sep 2017
 *
 * @brief  maps over scalar fields
 *
 * Copyright © 2017 Till Junge
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

#ifndef SRC_COMMON_FIELD_MAP_SCALAR_HH_
#define SRC_COMMON_FIELD_MAP_SCALAR_HH_

#include "common/field_map_base.hh"

namespace muSpectre {
  /**
   * Implements maps on scalar fields (i.e. material properties,
   * temperatures, etc). Maps onto a `muSpectre::internal::TypedSizedFieldBase`
   * and lets you iterate over it in the form of the bare type of the field.
   */
  template <class FieldCollection, typename T, bool ConstField = false>
  class ScalarFieldMap
      : public internal::FieldMap<FieldCollection, T, 1, ConstField> {
   public:
    //! base class
    using Parent = internal::FieldMap<FieldCollection, T, 1, ConstField>;
    //! sister class with all params equal, but ConstField guaranteed true
    using ConstMap = ScalarFieldMap<FieldCollection, T, true>;
    //! cell coordinates type
    using Ccoord = Ccoord_t<FieldCollection::spatial_dim()>;
    using value_type = T;                        //!< stl conformance
    using const_reference = const value_type &;  //!< stl conformance
    //! stl conformance
    using reference =
        std::conditional_t<ConstField, const_reference, value_type &>;
    using size_type = typename Parent::size_type;  //!< stl conformance
    using pointer = T *;                           //!< stl conformance
    using Field = typename Parent::Field;          //!< stl conformance
    using Field_c = typename Parent::Field_c;      //!< stl conformance
    //! stl conformance
    using const_iterator =
        typename Parent::template iterator<ScalarFieldMap, true>;
    //! iterator over a scalar field
    using iterator =
        std::conditional_t<ConstField, const_iterator,
                           typename Parent::template iterator<ScalarFieldMap>>;
    using reverse_iterator =
        std::reverse_iterator<iterator>;  //!< stl conformance
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
    ScalarFieldMap() = delete;

    //! constructor
    explicit ScalarFieldMap(Field_c &field);

    //! Copy constructor
    ScalarFieldMap(const ScalarFieldMap &other) = default;

    //! Move constructor
    ScalarFieldMap(ScalarFieldMap &&other) = default;

    //! Destructor
    virtual ~ScalarFieldMap() = default;

    //! Copy assignment operator
    ScalarFieldMap &operator=(const ScalarFieldMap &other) = delete;

    //! Move assignment operator
    ScalarFieldMap &operator=(ScalarFieldMap &&other) = delete;

    //! Assign a value to every entry
    ScalarFieldMap &operator=(T val);

    //! give human-readable field map type
    inline std::string info_string() const final;

    //! member access
    inline reference operator[](size_type index);
    inline reference operator[](const Ccoord &ccoord);

    inline const_reference operator[](size_type index) const;
    inline const_reference operator[](const Ccoord &ccoord) const;

    //! return an iterator to the first pixel of the field
    iterator begin() { return iterator(*this); }
    //! return an iterator to the first pixel of the field
    const_iterator cbegin() const { return const_iterator(*this); }
    //! return an iterator to the first pixel of the field
    const_iterator begin() const { return this->cbegin(); }
    //! return an iterator to tail of field for ranges
    iterator end() { return iterator(*this, false); }
    //! return an iterator to tail of field for ranges
    const_iterator cend() const { return const_iterator(*this, false); }
    //! return an iterator to tail of field for ranges
    const_iterator end() const { return this->cend(); }

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
    inline T mean() const;

   protected:
    //! for sad, legacy iterator use
    inline pointer ptr_to_val_t(size_type index);

    //! type identifier for printing and debugging
    static const std::string field_info_root;

   private:
  };

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, bool ConstField>
  ScalarFieldMap<FieldCollection, T, ConstField>::ScalarFieldMap(Field_c &field)
      : Parent(field) {
    this->check_compatibility();
  }

  /* ---------------------------------------------------------------------- */
  //! human-readable field map type
  template <class FieldCollection, typename T, bool ConstField>
  std::string
  ScalarFieldMap<FieldCollection, T, ConstField>::info_string() const {
    std::stringstream info;
    info << "Scalar(" << typeid(T).name() << ")";
    return info.str();
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, bool ConstField>
  const std::string
      ScalarFieldMap<FieldCollection, T, ConstField>::field_info_root{"Scalar"};

  /* ---------------------------------------------------------------------- */
  //! member access
  template <class FieldCollection, typename T, bool ConstField>
  typename ScalarFieldMap<FieldCollection, T, ConstField>::reference
      ScalarFieldMap<FieldCollection, T, ConstField>::
      operator[](size_type index) {
    return this->get_ptr_to_entry(std::move(index))[0];
  }

  /* ---------------------------------------------------------------------- */
  //! member access
  template <class FieldCollection, typename T, bool ConstField>
  typename ScalarFieldMap<FieldCollection, T, ConstField>::reference
      ScalarFieldMap<FieldCollection, T, ConstField>::
      operator[](const Ccoord &ccoord) {
    auto &&index = this->collection.get_index(std::move(ccoord));
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
      operator[](const Ccoord &ccoord) const {
    auto &&index = this->collection.get_index(std::move(ccoord));
    return this->get_ptr_to_entry(std::move(index))[0];
  }

  /* ---------------------------------------------------------------------- */
  //! Assign a value to every entry
  template <class FieldCollection, typename T, bool ConstField>
  ScalarFieldMap<FieldCollection, T, ConstField> &
  ScalarFieldMap<FieldCollection, T, ConstField>::operator=(T val) {
    for (auto &scalar : *this) {
      scalar = val;
    }
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, bool ConstField>
  T ScalarFieldMap<FieldCollection, T, ConstField>::mean() const {
    T mean{0};
    for (auto &&val : *this) {
      mean += val;
    }
    mean /= Real(this->size());
    return mean;
  }

}  // namespace muSpectre

#endif  // SRC_COMMON_FIELD_MAP_SCALAR_HH_
