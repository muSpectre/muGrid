/**
 * @file   field_map.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   12 Sep 2017
 *
 * @brief  Defined a strongly defines proxy that iterates efficiently over a field
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


#ifndef FIELD_MAP_BASE_H
#define FIELD_MAP_BASE_H

#include "common/field.hh"
#include "field_collection_base.hh"
#include "common/common.hh"

#include <unsupported/Eigen/CXX11/Tensor>

#include <array>
#include <string>
#include <memory>

namespace muSpectre {

  namespace internal {
    //----------------------------------------------------------------------------//
    //! little helper to automate creation of const maps without duplication
    template<class T, bool isConst>
    struct const_corrector {
      //! non-const type
      using type = typename T::reference;
    };

    //! specialisation for constant case
    template<class T>
    struct const_corrector<T, true> {
      //! const type
      using type = typename T::const_reference;
    };

    //! convenience alias
    template<class T, bool isConst>
    using const_corrector_t = typename const_corrector<T, isConst>::type;

    //----------------------------------------------------------------------------//
    /**
     * `FieldMap` provides two central mechanisms:
     * - Map a field (that knows only about the size of the underlying object,
     * onto the mathematical object (reprensented by the respective Eigen class)
     * that provides linear algebra functionality.
     * - Provide an iterator that allows to iterate over all pixels.
     * A field is represented by `FieldBase` or a derived class.
     * `FieldMap` has the specialisations `MatrixLikeFieldMap`,
     * `ScalarFieldMap` and `TensorFieldMap`.
     */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    class FieldMap
    {
    public:
      //! Fundamental type stored
      using Scalar = T;
      //! number of scalars per entry
      constexpr static auto nb_components{NbComponents};
      using TypedField_nc = TypedSizedFieldBase
        <FieldCollection, T, NbComponents>; //!< non-constant version of field
      //! field type as seen from iterator
      using TypedField = std::conditional_t<ConstField,
                                            const TypedField_nc,
                                            TypedField_nc>;
      using Field = typename TypedField::Base; //!< iterated field type
      //! const-correct field type
      using Field_c = std::conditional_t<ConstField,
                                         const Field,
                                         Field>;
      using size_type = std::size_t;  //!< stl conformance
      using pointer = std::conditional_t<ConstField,
                                         const T*,
                                         T*>; //!< stl conformance
      //! Default constructor
      FieldMap() = delete;

      //! constructor
      FieldMap(Field_c & field);

      //! constructor with run-time cost (for python and debugging)
      template<class FC, typename T2, Dim_t NbC>
      FieldMap(TypedSizedFieldBase<FC, T2, NbC> & field);

      //! Copy constructor
      FieldMap(const FieldMap &other) = default;

      //! Move constructor
      FieldMap(FieldMap &&other) = default;

      //! Destructor
      virtual ~FieldMap() = default;

      //! Copy assignment operator
      FieldMap& operator=(const FieldMap &other) = delete;

      //! Move assignment operator
      FieldMap& operator=(FieldMap &&other) = delete;

      //! give human-readable field map type
      virtual std::string info_string() const = 0;

      //! return field name
      inline const std::string & get_name() const;

      //! return my collection (for iterating)
      inline const FieldCollection & get_collection() const;

      //! member access needs to be implemented by inheriting classes
      //inline value_type operator[](size_t index);
      //inline value_type operator[](Ccoord ccord);

      //! check compatibility (must be called by all inheriting classes at the
      //! end of their constructors
      inline void check_compatibility();

      //! convenience call to collection's size method
      inline size_t size() const;

      //! compile-time compatibility check
      template<class TypedField>
      struct is_compatible;

      /**
       * iterates over all pixels in the `muSpectre::FieldCollection`
       * and dereferences to an Eigen map to the currently used field.
       */
      template <class FullyTypedFieldMap, bool ConstIter=false>
      class iterator
      {
        static_assert(!((ConstIter==false) && (ConstField==true)),
                      "You can't have a non-const iterator over a const "
                      "field");
      public:
        //! stl conformance
        using value_type =
          const_corrector_t<FullyTypedFieldMap, ConstIter>;
        //! stl conformance
        using const_value_type =
          const_corrector_t<FullyTypedFieldMap, true>;
        //! stl conformance
        using pointer = typename FullyTypedFieldMap::pointer;
        //! stl conformance
        using difference_type = std::ptrdiff_t;
        //! stl conformance
        using iterator_category = std::random_access_iterator_tag;
        //! cell coordinates type
        using Ccoord = typename FieldCollection::Ccoord;
        //! stl conformance
        using reference = typename FullyTypedFieldMap::reference;
        //! fully typed reference as seen by the iterator
        using TypedRef = std::conditional_t<ConstIter,
                                            const FullyTypedFieldMap &,
                                            FullyTypedFieldMap>;

        //! Default constructor
        iterator() = delete;

        //! constructor
        inline iterator(TypedRef fieldmap, bool begin=true);

        //! constructor for random access
        inline iterator(TypedRef fieldmap, size_t index);

        //! Copy constructor
        iterator(const iterator &other)= default;

        //! Move constructor
        iterator(iterator &&other) = default;

        //! Destructor
        virtual ~iterator() = default;

        //! Copy assignment operator
        iterator& operator=(const iterator &other) = default;

        //! Move assignment operator
        iterator& operator=(iterator &&other) = default;

        //! pre-increment
        inline iterator & operator++();
        //! post-increment
        inline iterator operator++(int);
        //! dereference
        inline value_type operator*();
        //! dereference
        inline const_value_type operator*() const;
        //! member of pointer
        inline pointer operator->();
        //! pre-decrement
        inline iterator & operator--();
        //! post-decrement
        inline iterator operator--(int);
        //! access subscripting
        inline value_type operator[](difference_type diff);
        //! access subscripting
        inline const_value_type operator[](const difference_type diff) const;
        //! equality
        inline bool operator==(const iterator & other) const;
        //! inequality
        inline bool operator!=(const iterator & other) const;
        //! div. comparisons
        inline bool operator<(const iterator & other) const;
        //! div. comparisons
        inline bool operator<=(const iterator & other) const;
        //! div. comparisons
        inline bool operator>(const iterator & other) const;
        //! div. comparisons
        inline bool operator>=(const iterator & other) const;
        //! additions, subtractions and corresponding assignments
        inline iterator operator+(difference_type diff) const;
        //! additions, subtractions and corresponding assignments
        inline iterator operator-(difference_type diff) const;
        //! additions, subtractions and corresponding assignments
        inline iterator& operator+=(difference_type diff);
        //! additions, subtractions and corresponding assignments
        inline iterator& operator-=(difference_type diff);

        //! get pixel coordinates
        inline Ccoord get_ccoord() const;

        //! ostream operator (mainly for debug
        friend std::ostream & operator<<(std::ostream & os,
                                         const iterator& it) {
          if (ConstIter) {
            os << "const ";
          }
          os << "iterator on field '"
             << it.fieldmap.get_name()
             << "', entry " << it.index;
          return os;
        }

      protected:
        const FieldCollection & collection; //!< collection of the field
        TypedRef fieldmap; //!< ref to the field itself
        size_t index; //!< index of currently pointed-to pixel
      private:
      };

      TypedField & get_field() {return this->field;}

    protected:
      //! raw pointer to entry (for Eigen Map)
      inline pointer get_ptr_to_entry(size_t index);
      //! raw pointer to entry (for Eigen Map)
      inline const T* get_ptr_to_entry(size_t index) const;
      const FieldCollection & collection; //!< collection holding Field
      TypedField & field;  //!< mapped Field
    private:
    };
  }  // internal


  namespace internal {

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    FieldMap(Field_c& field)
      :collection(field.get_collection()), field(static_cast<TypedField&>(field)) {
      static_assert(NbComponents>0,
                    "Only fields with more than 0 components allowed");
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template <class FC, typename T2, Dim_t NbC>
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    FieldMap(TypedSizedFieldBase<FC, T2, NbC> & field)
      :collection(field.get_collection()), field(static_cast<TypedField&>(field)) {
      static_assert(std::is_same<FC, FieldCollection>::value,
                    "The field does not have the expected FieldCollection type");
      static_assert(std::is_same<T2, T>::value,
                    "The field does not have the expected Scalar type");
      static_assert((NbC == NbComponents),
                    "The field does not have the expected number of components");
    }

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    void
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    check_compatibility() {
     if (typeid(T).hash_code() !=
          this->field.get_stored_typeid().hash_code()) {
       std::string err{"Cannot create a Map of type '" +
           this->info_string() +
           "' for field '" + this->field.get_name() + "' of type '" +
           this->field.get_stored_typeid().name() + "'"};
        throw FieldInterpretationError
          (err);
      }
      //check size compatibility
      if (NbComponents != this->field.get_nb_components()) {
        throw FieldInterpretationError
          ("Cannot create a Map of type '" +
           this->info_string() +
           "' for field '" + this->field.get_name() + "' with " +
           std::to_string(this->field.get_nb_components()) + " components");
      }
    }

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    size_t
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    size() const {
      return this->collection.size();
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template <class myField>
    struct FieldMap<FieldCollection, T, NbComponents, ConstField>::is_compatible {
      //! creates a more readable compile error
      constexpr static bool explain() {
        static_assert
          (std::is_same<typename myField::collection_t, FieldCollection>::value,
           "The field does not have the expected FieldCollection type");
        static_assert
          (std::is_same<typename myField::Scalar, T>::value,
           "The // field does not have the expected Scalar type");
        static_assert((TypedField::nb_components == NbComponents),
                      "The field does not have the expected number of components");
        //The static asserts wouldn't pass in the incompatible case, so this is it
        return true;
      }
      //! evaluated compatibility
      constexpr static bool value{std::is_base_of<TypedField, myField>::value};
    };

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    const std::string &
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    get_name() const {
      return this->field.get_name();
    }

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    const FieldCollection &
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    get_collection() const {
      return this->collection;
    }

    /* ---------------------------------------------------------------------- */
    /* ---------------------------------------------------------------------- */
    // Iterator implementations
    //! constructor
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    FieldMap<FieldCollection, T, NbComponents, ConstField>::iterator
    <FullyTypedFieldMap, ConstIter>::
    iterator(TypedRef fieldmap, bool begin)
      :collection(fieldmap.get_collection()), fieldmap(fieldmap),
       index(begin ? 0 : fieldmap.field.size()) {}

    /* ---------------------------------------------------------------------- */
    //! constructor for random access
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    FieldMap<FieldCollection, T, NbComponents, ConstField>::iterator<FullyTypedFieldMap, ConstIter>::
    iterator(TypedRef fieldmap, size_t index)
      :collection(fieldmap.collection), fieldmap(fieldmap),
       index(index) {}

    /* ---------------------------------------------------------------------- */
    //! pre-increment
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FieldMap<FieldCollection, T, NbComponents, ConstField>::template
    iterator<FullyTypedFieldMap, ConstIter> &
    FieldMap<FieldCollection, T, NbComponents, ConstField>::iterator<FullyTypedFieldMap, ConstIter>::
    operator++() {
      this->index++;
      return *this;
    }

    /* ---------------------------------------------------------------------- */
    //! post-increment
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FieldMap<FieldCollection, T, NbComponents, ConstField>::template
    iterator<FullyTypedFieldMap, ConstIter>
    FieldMap<FieldCollection, T, NbComponents, ConstField>::iterator<FullyTypedFieldMap, ConstIter>::
    operator++(int) {
      iterator current = *this;
      this->index++;
      return current;
    }

    /* ---------------------------------------------------------------------- */
    //! dereference
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FieldMap<FieldCollection, T, NbComponents, ConstField>::
    template iterator<FullyTypedFieldMap, ConstIter>::value_type
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator*() {
      return this->fieldmap.operator[](this->index);
    }

    /* ---------------------------------------------------------------------- */
    //! dereference
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FieldMap<FieldCollection, T, NbComponents, ConstField>::
    template iterator<FullyTypedFieldMap, ConstIter>::const_value_type
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator*() const {
      return this->fieldmap.operator[](this->index);
    }

    /* ---------------------------------------------------------------------- */
    //! member of pointer
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FullyTypedFieldMap::pointer
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator->() {
      return this->fieldmap.ptr_to_val_t(this->index);
    }

    /* ---------------------------------------------------------------------- */
    //! pre-decrement
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FieldMap<FieldCollection, T, NbComponents, ConstField>::template
    iterator<FullyTypedFieldMap, ConstIter> &
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator--() {
      this->index--;
      return *this;
    }

    /* ---------------------------------------------------------------------- */
    //! post-decrement
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FieldMap<FieldCollection, T, NbComponents, ConstField>::template
    iterator<FullyTypedFieldMap, ConstIter>
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator--(int) {
      iterator current = *this;
      this->index--;
      return current;
    }

    /* ---------------------------------------------------------------------- */
    //! Access subscripting
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FieldMap<FieldCollection, T, NbComponents, ConstField>::
    template iterator<FullyTypedFieldMap, ConstIter>::value_type
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator[](difference_type diff) {
      return this->fieldmap[this->index+diff];
    }

    /* ---------------------------------------------------------------------- */
    //! Access subscripting
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FieldMap<FieldCollection, T, NbComponents, ConstField>::template iterator<FullyTypedFieldMap, ConstIter>::const_value_type
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator[](const difference_type diff) const {
      return this->fieldmap[this->index+diff];
    }

    /* ---------------------------------------------------------------------- */
    //! equality
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    bool
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator==(const iterator & other) const {
      return (this->index == other.index);
    }

    /* ---------------------------------------------------------------------- */
    //! inquality
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    bool
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator!=(const iterator & other) const {
      return !(*this == other);
    }

    /* ---------------------------------------------------------------------- */
    //! div. comparisons
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    bool
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator<(const iterator & other) const {
      return (this->index < other.index);
    }
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    bool
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator<=(const iterator & other) const {
      return (this->index <= other.index);
    }
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    bool
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator>(const iterator & other) const {
      return (this->index > other.index);
    }
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    bool
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator>=(const iterator & other) const {
      return (this->index >= other.index);
    }

    /* ---------------------------------------------------------------------- */
    //! additions, subtractions and corresponding assignments
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FieldMap<FieldCollection, T, NbComponents, ConstField>::template
    iterator<FullyTypedFieldMap, ConstIter>
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator+(difference_type diff) const {
      return iterator(this->fieldmap, this->index + diff);
    }
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FieldMap<FieldCollection, T, NbComponents, ConstField>::template
    iterator<FullyTypedFieldMap, ConstIter>
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator-(difference_type diff) const {
      return iterator(this->fieldmap, this->index - diff);
    }
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FieldMap<FieldCollection, T, NbComponents, ConstField>::template
    iterator<FullyTypedFieldMap, ConstIter> &
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator+=(difference_type diff) {
      this->index += diff;
      return *this;
    }
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FieldMap<FieldCollection, T, NbComponents, ConstField>::template
    iterator<FullyTypedFieldMap, ConstIter> &
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    operator-=(difference_type diff) {
      this->index -= diff;
      return *this;
    }

    /* ---------------------------------------------------------------------- */
    //! get pixel coordinates
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    template<class FullyTypedFieldMap, bool ConstIter>
    typename FieldCollection::Ccoord
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    iterator<FullyTypedFieldMap, ConstIter>::
    get_ccoord() const {
      return this->collection.get_ccoord(this->index);
    }

////----------------------------------------------------------------------------//
//template<class FieldCollection, typename T, Dim_t NbComponents, class FullyTypedFieldMap>
//std::ostream & operator <<
//(std::ostream &os,
// const typename FieldMap<FieldCollection, T, NbComponents, ConstField>::
// template iterator<FullyTypedFieldMap, ConstIter> & it) {
//  os << "iterator on field '"
//     << it.field.get_name()
//     << "', entry " << it.index;
//  return os;
//}

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    typename FieldMap<FieldCollection, T, NbComponents, ConstField>::pointer
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    get_ptr_to_entry(size_t index) {
      return this->field.get_ptr_to_entry(std::move(index));
    }

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ConstField>
    const T*
    FieldMap<FieldCollection, T, NbComponents, ConstField>::
    get_ptr_to_entry(size_t index) const {
      return this->field.get_ptr_to_entry(std::move(index));
    }

  }  // internal


}  // muSpectre

#endif /* FIELD_MAP_BASE_H */
