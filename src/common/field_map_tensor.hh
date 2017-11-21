/**
 * file   field_map_tensor.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   26 Sep 2017
 *
 * @brief  Defines an Eigen-Tensor map over strongly typed fields
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

#ifndef FIELD_MAP_TENSOR_H
#define FIELD_MAP_TENSOR_H

#include <sstream>
#include <memory>
#include "common/eigen_tools.hh"
#include "common/field_map_base.hh"


namespace muSpectre {


  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  class TensorFieldMap: public
    internal::FieldMap<FieldCollection,
                       T,
                       SizesByOrder<order, dim>::Sizes::total_size
                       >
  {
  public:
    using Parent = internal::FieldMap<FieldCollection, T,
                                      TensorFieldMap::nb_components>;
    using Ccoord = Ccoord_t<FieldCollection::spatial_dim()>;
    using Sizes = typename SizesByOrder<order, dim>::Sizes;
    using T_t = Eigen::TensorFixedSize<T, Sizes>;
    using value_type = Eigen::TensorMap<T_t>;
    using reference = value_type; // since it's a resource handle
    using const_reference = Eigen::TensorMap<const T_t>;
    using size_type = typename Parent::size_type;
    using pointer = std::unique_ptr<value_type>;
    using TypedField = typename Parent::TypedField;
    using Field = typename TypedField::Parent;
    using iterator = typename Parent::template iterator<TensorFieldMap>;
    using const_iterator = typename Parent::template iterator<TensorFieldMap, true>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    friend iterator;

    //! Default constructor
    TensorFieldMap() = delete;

    TensorFieldMap(Field & field);

    //! Copy constructor
    TensorFieldMap(const TensorFieldMap &other) = delete;

    //! Move constructor
    TensorFieldMap(TensorFieldMap &&other) noexcept = delete;

    //! Destructor
    virtual ~TensorFieldMap() noexcept = default;

    //! Copy assignment operator
    TensorFieldMap& operator=(const TensorFieldMap &other) = delete;

    //! Move assignment operator
    TensorFieldMap& operator=(TensorFieldMap &&other) noexcept = delete;

    //! give human-readable field map type
    inline std::string info_string() const override final;

    //! member access
    template <class ref_t = reference>
    inline ref_t operator[](size_type index);
    inline reference operator[](const Ccoord & ccoord);

    //! return an iterator to head of field for ranges
    inline iterator begin(){return iterator(*this);}
    inline const_iterator cbegin(){return const_iterator(*this);}
    //! return an iterator to tail of field for ranges
    inline iterator end(){return iterator(*this, false);}
    inline const_iterator cend(){return const_iterator(*this, false);}

  protected:
    //! for sad, legacy iterator use
    inline pointer ptr_to_val_t(size_type index);
  private:
  };

  /* ---------------------------------------------------------------------- */
  template<class FieldCollection, typename T, Dim_t order, Dim_t dim>
  TensorFieldMap<FieldCollection, T, order, dim>::
  TensorFieldMap(Field & field)
    :Parent(field) {
    this->check_compatibility();
  }


  /* ---------------------------------------------------------------------- */
  //! human-readable field map type
  template<class FieldCollection, typename T, Dim_t order, Dim_t dim>
  std::string
  TensorFieldMap<FieldCollection, T, order, dim>::info_string() const {
    std::stringstream info;
    info << "Tensor(" << typeid(T).name() << ", " << order
         << "_o, " << dim << "_d)";
    return info.str();
  }

  /* ---------------------------------------------------------------------- */
  //! member access
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  template <class ref_t>
  ref_t
  TensorFieldMap<FieldCollection, T, order, dim>::operator[](size_type index) {
    auto && lambda = [this, &index](auto&&...tens_sizes) {
      return ref_t(this->get_ptr_to_entry(index), tens_sizes...);
    };
    return call_sizes<order, dim>(lambda);
  }
  template<class FieldCollection, typename T, Dim_t order, Dim_t dim>
  typename TensorFieldMap<FieldCollection, T, order, dim>::reference
  TensorFieldMap<FieldCollection, T, order, dim>::
  operator[](const Ccoord & ccoord) {
    auto && index = this->collection.get_index(ccoord);
    auto && lambda = [this, &index](auto&&...sizes) {
      return ref_t(this->get_ptr_to_entry(index), sizes...);
    };
    return call_sizes<order, dim>(lambda);
  }

  /* ---------------------------------------------------------------------- */
  //! for sad, legacy iterator use. Don't use unless you have to.
  template<class FieldCollection, typename T, Dim_t order, Dim_t dim>
  typename TensorFieldMap<FieldCollection, T, order, dim>::pointer
  TensorFieldMap<FieldCollection, T, order, dim>::
  ptr_to_val_t(size_type index) {
    auto && lambda = [this, &index](auto&&... tens_sizes) {
      return std::make_unique<value_type>(this->get_ptr_to_entry(index),
                                       tens_sizes...);
    };
    return call_sizes<order, dim>(lambda);
  }


}  // muSpectre


#endif /* FIELD_MAP_TENSOR_H */
