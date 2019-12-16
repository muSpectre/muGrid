/**
 * @file   field_typed.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   13 Aug 2019
 *
 * @brief  Implementation for typed fields
 *
 * Copyright © 2019 Till Junge
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
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

#include <sstream>

#include "field_typed.hh"
#include "field_collection.hh"
#include "field_map.hh"

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedFieldBase<T>::set_data_ptr(T * ptr) {
    this->data_ptr = ptr;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  T * TypedFieldBase<T>::data() const {
    return this->data_ptr;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedField<T> & TypedField<T>::operator=(const Parent & other) {
    Parent::operator=(other);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedField<T> & TypedField<T>::operator=(const Negative & other) {
    Parent::operator=(other);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedField<T> & TypedField<T>::operator=(const EigenRep_t & other) {
    Parent::operator=(other);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedField<T>::set_zero() {
    std::fill(this->values.begin(), this->values.end(), T{});
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedField<T>::set_pad_size(size_t pad_size) {
    this->pad_size = pad_size;
    this->resize(this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedField<T> & TypedField<T>::safe_cast(Field & other) {
    try {
      return dynamic_cast<TypedField<T> &>(other);
    } catch (const std::bad_cast &) {
      std::stringstream error{};
      error << "Cannot cast field'" << other.get_name()
            << "' to a typed field of type '" << typeid(T).name()
            << "', because it is of type '" << other.get_stored_typeid().name()
            << "'.";
      throw FieldError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  const TypedField<T> & TypedField<T>::safe_cast(const Field & other) {
    try {
      return dynamic_cast<const TypedField<T> &>(other);
    } catch (const std::bad_cast &) {
      std::stringstream error{};
      error << "Cannot cast field'" << other.get_name()
            << "' to a typed field of type '" << typeid(T).name()
            << "', because it is of type '" << other.get_stored_typeid().name()
            << "'.";
      throw FieldError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedField<T> & TypedField<T>::safe_cast(Field & other,
                                             const Dim_t & nb_components) {
    if (other.get_nb_components() != nb_components) {
      std::stringstream error{};
      error << "Cannot cast field'" << other.get_name() << "', because it has "
            << other.get_nb_components() << " compoments, rather than the "
            << nb_components << " components which are requested.";
      throw FieldError(error.str());
    }
    return TypedField::safe_cast(other);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  const TypedField<T> &
  TypedField<T>::safe_cast(const Field & other, const Dim_t & nb_components) {
    if (other.get_nb_components() != nb_components) {
      std::stringstream error{};
      error << "Cannot cast field'" << other.get_name() << "', because it has "
            << other.get_nb_components() << " compoments, rather than the "
            << nb_components << " components which are requested.";
      throw FieldError(error.str());
    }
    return TypedField::safe_cast(other);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedField<T>::resize(size_t size) {
    const auto expected_size{size * this->get_nb_components() + this->pad_size};
    if (this->values.size() != expected_size or this->current_size != size) {
      this->current_size = size;
      this->values.resize(expected_size);
    }
    this->set_data_ptr(this->values.data());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  size_t TypedField<T>::buffer_size() const {
    return this->values.size();
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedField<T>::push_back(const T & value) {
    if (this->is_global()) {
      throw FieldError(
          "push_back() makes no sense on global fields (you can't "
          "add individual pixels");
    }
    if (not this->collection.has_nb_quad()) {
      throw FieldError("Cannot push_back into a field before the number of "
                        "quadrature points has bee set for the collection");
    }
    if (this->nb_components != 1) {
      throw FieldError("This is not a scalar field. push_back an array.");
    }
    const auto & nb_quad{this->collection.get_nb_quad()};
    this->current_size += nb_quad;
    for (Dim_t quad_pt_id{0}; quad_pt_id < nb_quad; ++quad_pt_id) {
      this->values.push_back(value);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedField<T>::push_back(
      const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> &
          value) {
    if (this->is_global()) {
      throw FieldError(
          "push_back() makes no sense on global fields (you can't "
          "add individual pixels");
    }
    if (not this->collection.has_nb_quad()) {
      throw FieldError("Cannot push_back into a field before the number of "
                        "quadrature points has bee set for the collection");
    }
    if (this->nb_components != value.size()) {
      std::stringstream error{};
      error << "You are trying to push an array with " << value.size()
            << "components into a field with " << this->nb_components
            << " components.";
      throw FieldError(error.str());
    }
    const auto & nb_quad{this->collection.get_nb_quad()};
    this->current_size += nb_quad;
    for (Dim_t quad_pt_id{0}; quad_pt_id < nb_quad; ++quad_pt_id) {
      for (Dim_t i{0}; i < this->nb_components; ++i) {
        this->values.push_back(value.data()[i]);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_map(const Dim_t & nb_rows,
                                     const Dim_t & nb_cols) -> Eigen_map {
    if (not this->collection.is_initialised()) {
      std::stringstream error{};
      error << "The FieldCollection for field '" << this->name
            << "' has not been initialised";
      throw FieldError(error.str());
    }
    return Eigen_map(this->data_ptr, nb_rows, nb_cols);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_map(const Dim_t & nb_rows,
                                     const Dim_t & nb_cols) const
      -> Eigen_cmap {
    if (not this->collection.is_initialised()) {
      std::stringstream error{};
      error << "The FieldCollection for field '" << this->name
            << "' has not been initialised";
      throw FieldError(error.str());
    }
    return Eigen_cmap(this->data_ptr, nb_rows, nb_cols);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedFieldBase<T> & TypedFieldBase<T>::
  operator=(const TypedFieldBase & other) {
    this->eigen_vec() = other.eigen_vec();
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedFieldBase<T> & TypedFieldBase<T>::operator=(const Negative & other) {
    this->eigen_vec() = -other.field.eigen_vec();
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedFieldBase<T> & TypedFieldBase<T>::operator=(const EigenRep_t & other) {
    this->eigen_vec() = other;
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::operator-() const -> Negative {
    return Negative{*this};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedFieldBase<T> & TypedFieldBase<T>::
  operator+=(const TypedFieldBase & other) {
    this->eigen_vec() += other.eigen_vec();
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedFieldBase<T> & TypedFieldBase<T>::
  operator-=(const TypedFieldBase & other) {
    this->eigen_vec() -= other.eigen_vec();
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_vec() -> Eigen_map {
    return this->eigen_map(this->size() * this->nb_components, 1);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_vec() const -> Eigen_cmap {
    return this->eigen_map(this->size() * this->nb_components, 1);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_quad_pt() -> Eigen_map {
    return this->eigen_map(this->nb_components, this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_quad_pt() const -> Eigen_cmap {
    return this->eigen_map(this->nb_components, this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_pixel() -> Eigen_map {
    const auto & nb_quad{this->collection.get_nb_quad()};
    return this->eigen_map(this->nb_components * nb_quad,
                           this->size() / nb_quad);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_pixel() const -> Eigen_cmap {
    const auto & nb_quad{this->collection.get_nb_quad()};
    return this->eigen_map(this->nb_components * nb_quad,
                           this->size() / nb_quad);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::get_pixel_map(const Dim_t & nb_rows)
      -> FieldMap<T, Mapping::Mut> {
    return (nb_rows == -1)
               ? FieldMap<T, Mapping::Mut>{*this, Iteration::Pixel}
               : FieldMap<T, Mapping::Mut>{*this, nb_rows, Iteration::Pixel};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::get_pixel_map(const Dim_t & nb_rows) const
      -> FieldMap<T, Mapping::Const> {
    return (nb_rows == -1)
               ? FieldMap<T, Mapping::Const>{*this, Iteration::Pixel}
               : FieldMap<T, Mapping::Const>{*this, nb_rows, Iteration::Pixel};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::get_quad_pt_map(const Dim_t & nb_rows)
      -> FieldMap<T, Mapping::Mut> {
    return (nb_rows == -1)
               ? FieldMap<T, Mapping::Mut>{*this, Iteration::QuadPt}
               : FieldMap<T, Mapping::Mut>{*this, nb_rows, Iteration::QuadPt};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::get_quad_pt_map(const Dim_t & nb_rows) const
      -> FieldMap<T, Mapping::Const> {
    return (nb_rows == -1)
               ? FieldMap<T, Mapping::Const>{*this, Iteration::QuadPt}
               : FieldMap<T, Mapping::Const>{*this, nb_rows,
                                              Iteration::QuadPt};
  }

  template <typename T>
  WrappedField<T>::WrappedField(const std::string & unique_name,
                                  FieldCollection & collection,
                                  Dim_t nb_components, size_t size, T * ptr)
      : Parent{unique_name, collection, nb_components}, size{
                                                            static_cast<size_t>(
                                                                size)} {
    this->current_size = size / this->nb_components;

    if (size != this->nb_components * this->current_size) {
      std::stringstream error{};
      error << "Size mismatch: the provided array has a size of " << size
            << " which is not a multiple of the specified number of components "
               "(nb_components = "
            << this->nb_components << ").";
      throw FieldError(error.str());
    }
    if (this->collection.get_nb_entries() != Dim_t(this->current_size)) {
      std::stringstream error{};
      error << "Size mismatch: This field should store " << this->nb_components
            << " component(s) on " << this->collection.get_nb_pixels()
            << " pixels/voxels with " << this->collection.get_nb_quad()
            << " quadrature point(s) each, i.e. with a total of "
            << this->collection.get_nb_entries() * this->nb_components
            << " scalar values, but you supplied an array of size " << size
            << '.';
      throw FieldError(error.str());
    }
    this->set_data_ptr(ptr);
  }

  template <typename T>
  WrappedField<T>::WrappedField(const std::string & unique_name,
                                  FieldCollection & collection,
                                  Dim_t nb_components,
                                  Eigen::Ref<EigenRep_t> values)
      : Parent{unique_name, collection, nb_components},
        size{static_cast<size_t>(values.size())} {
    this->current_size = values.size() / this->nb_components;

    if (values.size() != Dim_t(this->nb_components * this->current_size)) {
      std::stringstream error{};
      error << "Size mismatch: the provided array has a size of "
            << values.size()
            << " which is not a multiple of the specified number of components "
               "(nb_components = "
            << this->nb_components << ").";
      throw FieldError(error.str());
    }
    if (this->collection.get_nb_entries() != Dim_t(this->current_size)) {
      std::stringstream error{};
      error << "Size mismatch: This field should store " << this->nb_components
            << " component(s) on " << this->collection.get_nb_pixels()
            << " pixels/voxels with " << this->collection.get_nb_quad()
            << " quadrature point(s) each, i.e. with a total of "
            << this->collection.get_nb_entries() * this->nb_components
            << " scalar values, but you supplied an array of size "
            << values.size() << '.';
      throw FieldError(error.str());
    }
    this->set_data_ptr(values.data());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto WrappedField<T>::make_const(const std::string & unique_name,
                                    FieldCollection & collection,
                                    Dim_t nb_components,
                                    Eigen::Ref<const EigenRep_t> values)
      -> std::unique_ptr<const WrappedField> {
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> map{
        const_cast<T *>(values.data()), values.rows(), values.cols()};
    return std::make_unique<WrappedField>(unique_name, collection,
                                           nb_components, map);
  }
  /* ---------------------------------------------------------------------- */
  template <typename T>
  void WrappedField<T>::set_pad_size(size_t pad_size) {
    std::stringstream error;
    error << "Setting pad size to " << pad_size << " not possible for "
          << "wrapped fields.";
    throw FieldError(error.str());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void WrappedField<T>::set_zero() {
    std::fill(static_cast<T *>(this->data_ptr),
              static_cast<T *>(this->data_ptr) + this->size, T{});
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void WrappedField<T>::resize(size_t size) {
    const auto expected_size{size * this->get_nb_components() + this->pad_size};
    if (expected_size != this->buffer_size()) {
      std::stringstream error{};
      error << "Wrapped fields cannot be resized. The current wrapped size is "
            << this->buffer_size() << ". Resize to " << expected_size
            << " was attempted.";
      throw FieldError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  size_t WrappedField<T>::buffer_size() const {
    return this->size;
  }

  /* ---------------------------------------------------------------------- */
  template class TypedFieldBase<Real>;
  template class TypedFieldBase<Complex>;
  template class TypedFieldBase<Int>;
  template class TypedFieldBase<Uint>;

  template class TypedField<Real>;
  template class TypedField<Complex>;
  template class TypedField<Int>;
  template class TypedField<Uint>;

  template class WrappedField<Real>;
  template class WrappedField<Complex>;
  template class WrappedField<Int>;
  template class WrappedField<Uint>;
}  // namespace muGrid
