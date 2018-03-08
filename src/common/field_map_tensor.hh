/**
 * @file   field_map_tensor.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   26 Sep 2017
 *
 * @brief  Defines an Eigen-Tensor map over strongly typed fields
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

#ifndef FIELD_MAP_TENSOR_H
#define FIELD_MAP_TENSOR_H

#include "common/eigen_tools.hh"
#include "common/field_map_base.hh"

#include <sstream>
#include <memory>

namespace muSpectre {


  /* ---------------------------------------------------------------------- */
  /**
   * Maps onto a `muSpectre::internal::TypedSizedFieldBase` and lets
   * you iterate over it in the form of `Eigen::TensorMap<TensorFixedSize<...>>`
   */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim,
            bool ConstField=false>
  class TensorFieldMap: public
    internal::FieldMap<FieldCollection,
                       T,
                       SizesByOrder<order, dim>::Sizes::total_size,
                       ConstField>
  {
  public:
    //! base class
    using Parent = internal::FieldMap<FieldCollection, T,
                                      TensorFieldMap::nb_components, ConstField>;
    //! sister class with all params equal, but ConstField guaranteed true
    using ConstMap = TensorFieldMap<FieldCollection, T, order, dim, true>;
    //! cell coordinates type
    using Ccoord = Ccoord_t<FieldCollection::spatial_dim()>;
    //! static tensor size
    using Sizes = typename SizesByOrder<order, dim>::Sizes;
    //! plain Eigen type
    using T_t = Eigen::TensorFixedSize<T, Sizes>;
    using value_type = Eigen::TensorMap<T_t>; //!< stl conformance
    using const_reference = Eigen::TensorMap<const T_t>; //!< stl conformance
    //! stl conformance
    using reference = std::conditional_t<ConstField,
                                         const_reference,
                                         value_type>; // since it's a resource handle
    using size_type = typename Parent::size_type; //!< stl conformance
    using pointer = std::unique_ptr<value_type>; //!< stl conformance
    using TypedField = typename Parent::TypedField; //!< field to map
    //! polymorphic base field type (for debug and python)
    using Field = typename Parent::Field;
    //! polymorphic base field type (for debug and python)
    using Field_c = typename Parent::Field_c;
    //! stl conformance
    using const_iterator = typename Parent::template iterator<TensorFieldMap, true>;
    //! stl conformance
    using iterator = std::conditional_t<
      ConstField,
      const_iterator,
      typename Parent::template iterator<TensorFieldMap>>;
    //! stl conformance
    using reverse_iterator = std::reverse_iterator<iterator>;
    //! stl conformance
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    //! stl conformance
    friend iterator;

    //! Default constructor
    TensorFieldMap() = delete;

    //! constructor
    TensorFieldMap(Field_c & field);

    //! Copy constructor
    TensorFieldMap(const TensorFieldMap &other) = default;

    //! Move constructor
    TensorFieldMap(TensorFieldMap &&other) = default;

    //! Destructor
    virtual ~TensorFieldMap() = default;

    //! Copy assignment operator
    TensorFieldMap& operator=(const TensorFieldMap &other) = delete;

    //! Assign a matrixlike value to every entry
    inline TensorFieldMap & operator=(const T_t & val);

    //! Move assignment operator
    TensorFieldMap& operator=(TensorFieldMap &&other) = delete;

    //! give human-readable field map type
    inline std::string info_string() const override final;

    //! member access
    inline reference operator[](size_type index);
    //! member access
    inline reference operator[](const Ccoord & ccoord);

    //! member access
    inline const_reference operator[](size_type index) const;
    //! member access
    inline const_reference operator[](const Ccoord& ccoord) const;

    //! return an iterator to first entry of field
    inline iterator begin(){return iterator(*this);}
    //! return an iterator to first entry of field
    inline const_iterator cbegin() const {return const_iterator(*this);}
    //! return an iterator to first entry of field
    inline const_iterator begin() const {return this->cbegin();}
    //! return an iterator past the last entry of field
    inline iterator end(){return iterator(*this, false);}
    //! return an iterator past the last entry of field
    inline const_iterator cend() const {return const_iterator(*this, false);}
    //! return an iterator past the last entry of field
    inline const_iterator end() const {return this->cend();}

    //! evaluate the average of the field
    inline T_t mean() const;
  protected:
    //! for sad, legacy iterator use
    inline pointer ptr_to_val_t(size_type index);
  private:
  };

  /* ---------------------------------------------------------------------- */
  template<class FieldCollection, typename T, Dim_t order, Dim_t dim,
           bool ConstField>
  TensorFieldMap<FieldCollection, T, order, dim, ConstField>::
  TensorFieldMap(Field_c & field)
    :Parent(field) {
    this->check_compatibility();
  }

  /* ---------------------------------------------------------------------- */
  //! human-readable field map type
  template<class FieldCollection, typename T, Dim_t order, Dim_t dim, bool ConstField>
  std::string
  TensorFieldMap<FieldCollection, T, order, dim, ConstField>::info_string() const {
    std::stringstream info;
    info << "Tensor(" << typeid(T).name() << ", " << order
         << "_o, " << dim << "_d)";
    return info.str();
  }

  /* ---------------------------------------------------------------------- */
  //! member access
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim,
            bool ConstField>
  typename TensorFieldMap<FieldCollection, T, order, dim, ConstField>::reference
  TensorFieldMap<FieldCollection, T, order, dim, ConstField>::
  operator[](size_type index) {
    auto && lambda = [this, &index](auto&&...tens_sizes) {
      return reference(this->get_ptr_to_entry(index), tens_sizes...);
    };
    return call_sizes<order, dim>(lambda);
  }

  template<class FieldCollection, typename T, Dim_t order, Dim_t dim,
           bool ConstField>
  typename TensorFieldMap<FieldCollection, T, order, dim, ConstField>::reference
  TensorFieldMap<FieldCollection, T, order, dim, ConstField>::
  operator[](const Ccoord & ccoord) {
    auto && index = this->collection.get_index(ccoord);
    auto && lambda = [this, &index](auto&&...sizes) {
      return reference(this->get_ptr_to_entry(index), sizes...);
    };
    return call_sizes<order, dim>(lambda);
  }

  template <class FieldCollection, typename T, Dim_t order, Dim_t dim,
            bool ConstField>
  typename TensorFieldMap<FieldCollection, T, order, dim, ConstField>::
  const_reference
  TensorFieldMap<FieldCollection, T, order, dim, ConstField>::
  operator[](size_type index) const {
    // Warning: due to a inconsistency in Eigen's API, tensor maps
    // cannot be constructed from a const ptr, hence this nasty const
    // cast :(
    auto && lambda = [this, &index](auto&&...tens_sizes) {
      return const_reference(const_cast<T*>(this->get_ptr_to_entry(index)),
                   tens_sizes...);
    };
    return call_sizes<order, dim>(lambda);
  }

  template<class FieldCollection, typename T, Dim_t order, Dim_t dim,
           bool ConstField>
  typename TensorFieldMap<FieldCollection, T, order, dim, ConstField>::
  const_reference
  TensorFieldMap<FieldCollection, T, order, dim, ConstField>::
  operator[](const Ccoord & ccoord) const {
    auto && index = this->collection.get_index(ccoord);
    auto && lambda = [this, &index](auto&&...sizes) {
      return const_reference(const_cast<T*>(this->get_ptr_to_entry(index)),
                             sizes...);
    };
    return call_sizes<order, dim>(lambda);
  }

  /* ---------------------------------------------------------------------- */
  template<class FieldCollection, typename T, Dim_t order, Dim_t dim,
           bool ConstField>
  TensorFieldMap<FieldCollection, T, order, dim, ConstField> &
  TensorFieldMap<FieldCollection, T, order, dim, ConstField>::
  operator=(const T_t & val) {
    for (auto && tens: *this) {
      tens = val;
    }
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim,
            bool ConstField>
  typename TensorFieldMap<FieldCollection, T, order, dim, ConstField>::T_t
  TensorFieldMap<FieldCollection, T, order, dim, ConstField>::
  mean() const {
    T_t mean{T_t::Zero()};
    for (auto && val: *this) {
      mean += val;
    }
    mean *= 1./Real(this->size());
    return mean;
  }

  /* ---------------------------------------------------------------------- */
  //! for sad, legacy iterator use. Don't use unless you have to.
  template<class FieldCollection, typename T, Dim_t order, Dim_t dim,
           bool ConstField>
  typename TensorFieldMap<FieldCollection, T, order, dim, ConstField>::pointer
  TensorFieldMap<FieldCollection, T, order, dim, ConstField>::
  ptr_to_val_t(size_type index) {
    auto && lambda = [this, &index](auto&&... tens_sizes) {
      return std::make_unique<value_type>(this->get_ptr_to_entry(index),
                                       tens_sizes...);
    };
    return call_sizes<order, dim>(lambda);
  }


}  // muSpectre


#endif /* FIELD_MAP_TENSOR_H */
