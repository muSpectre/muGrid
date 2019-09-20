/**
 * @file   nfield_typed.cc
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
 * General Public License for more details.
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
 */

#include "nfield_typed.hh"
#include "nfield_collection.hh"
#include "nfield_map.hh"

#include <sstream>

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedNFieldBase<T>::set_data_ptr(T * ptr) {
    this->data_ptr = ptr;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  T * TypedNFieldBase<T>::data() const {
    return this->data_ptr;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedNField<T> & TypedNField<T>::operator=(const TypedNField & other) {
    Parent::operator=(other);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedNField<T> & TypedNField<T>::operator=(const Negative & other) {
    Parent::operator=(other);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedNField<T> & TypedNField<T>::operator=(const EigenRep_t & other) {
    Parent::operator=(other);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedNField<T>::set_zero() {
    std::fill(this->values.begin(), this->values.end(), T{});
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedNField<T>::set_pad_size(size_t pad_size) {
    this->pad_size = pad_size;
    this->resize(this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedNField<T> & TypedNField<T>::safe_cast(NField & other) {
    try {
      return dynamic_cast<TypedNField<T> &>(other);
    } catch (const std::bad_cast &) {
      std::stringstream error{};
      error << "Cannot cast field'" << other.get_name()
            << "' to a typed field of type '" << typeid(T).name()
            << "', because it is of type '" << other.get_stored_typeid().name()
            << "'.";
      throw NFieldError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  const TypedNField<T> & TypedNField<T>::safe_cast(const NField & other) {
    try {
      return dynamic_cast<const TypedNField<T> &>(other);
    } catch (const std::bad_cast &) {
      std::stringstream error{};
      error << "Cannot cast field'" << other.get_name()
            << "' to a typed field of type '" << typeid(T).name()
            << "', because it is of type '" << other.get_stored_typeid().name()
            << "'.";
      throw NFieldError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedNField<T> & TypedNField<T>::safe_cast(NField & other,
                                             const Dim_t & nb_components) {
    if (other.get_nb_components() != nb_components) {
      std::stringstream error{};
      error << "Cannot cast field'" << other.get_name() << "', because it has "
            << other.get_nb_components() << " compoments, rather than the "
            << nb_components << " components which are requested.";
      throw NFieldError(error.str());
    }
    return TypedNField::safe_cast(other);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  const TypedNField<T> &
  TypedNField<T>::safe_cast(const NField & other, const Dim_t & nb_components) {
    if (other.get_nb_components() != nb_components) {
      std::stringstream error{};
      error << "Cannot cast field'" << other.get_name() << "', because it has "
            << other.get_nb_components() << " compoments, rather than the "
            << nb_components << " components which are requested.";
      throw NFieldError(error.str());
    }
    return TypedNField::safe_cast(other);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedNField<T>::resize(size_t size) {
    const auto expected_size{size * this->get_nb_components() + this->pad_size};
    if (not(this->values.size() == expected_size)) {
      this->current_size = size;
      this->values.resize(expected_size);
    }
    this->set_data_ptr(&this->values.front());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  size_t TypedNField<T>::buffer_size() const {
    return this->values.size();
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedNField<T>::push_back(const T & value) {
    if (this->is_global()) {
      throw NFieldError(
          "push_back() makes no sense on global fields (you can't "
          "add individual pixels");
    }
    if (not this->collection.has_nb_quad()) {
      throw NFieldError("Cannot push_back into a field before the number of "
                        "quadrature points has bee set for the collection");
    }
    if (this->nb_components != 1) {
      throw NFieldError("This is not a scalar field. push_back an array.");
    }
    const auto & nb_quad{this->collection.get_nb_quad()};
    this->current_size += nb_quad;
    for (Dim_t quad_pt_id{0}; quad_pt_id < nb_quad; ++quad_pt_id) {
      this->values.push_back(value);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedNField<T>::push_back(
      const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> &
          value) {
    if (this->is_global()) {
      throw NFieldError(
          "push_back() makes no sense on global fields (you can't "
          "add individual pixels");
    }
    if (not this->collection.has_nb_quad()) {
      throw NFieldError("Cannot push_back into a field before the number of "
                        "quadrature points has bee set for the collection");
    }
    if (this->nb_components != value.size()) {
      std::stringstream error{};
      error << "You are trying to push an array with " << value.size()
            << "components into a field with " << this->nb_components
            << " components.";
      throw NFieldError(error.str());
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
  auto TypedNFieldBase<T>::eigen_map(const Dim_t & nb_rows,
                                     const Dim_t & nb_cols) -> Eigen_map {
    if (not this->collection.is_initialised()) {
      std::stringstream error{};
      error << "The FieldCollection for field '" << this->name
            << "' has not been initialised";
      throw NFieldError(error.str());
    }
    return Eigen_map(this->data_ptr, nb_rows, nb_cols);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedNFieldBase<T>::eigen_map(const Dim_t & nb_rows,
                                     const Dim_t & nb_cols) const
      -> Eigen_cmap {
    if (not this->collection.is_initialised()) {
      std::stringstream error{};
      error << "The FieldCollection for field '" << this->name
            << "' has not been initialised";
      throw NFieldError(error.str());
    }
    return Eigen_cmap(this->data_ptr, nb_rows, nb_cols);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedNFieldBase<T> & TypedNFieldBase<T>::
  operator=(const TypedNFieldBase & other) {
    this->eigen_vec() = other.eigen_vec();
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedNFieldBase<T> & TypedNFieldBase<T>::operator=(const Negative & other) {
    this->eigen_vec() = -other.field.eigen_vec();
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedNFieldBase<T> & TypedNFieldBase<T>::operator=(const EigenRep_t & other) {
    this->eigen_vec() = other;
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedNFieldBase<T>::operator-() const -> Negative {
    return Negative{*this};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedNFieldBase<T> & TypedNFieldBase<T>::
  operator+=(const TypedNFieldBase & other) {
    this->eigen_vec() += other.eigen_vec();
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedNFieldBase<T>::eigen_vec() -> Eigen_map {
    return this->eigen_map(this->size() * this->nb_components, 1);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedNFieldBase<T>::eigen_vec() const -> Eigen_cmap {
    return this->eigen_map(this->size() * this->nb_components, 1);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedNFieldBase<T>::eigen_quad_pt() -> Eigen_map {
    return this->eigen_map(this->nb_components, this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedNFieldBase<T>::eigen_quad_pt() const -> Eigen_cmap {
    return this->eigen_map(this->nb_components, this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedNFieldBase<T>::eigen_pixel() -> Eigen_map {
    const auto & nb_quad{this->collection.get_nb_quad()};
    return this->eigen_map(this->nb_components * nb_quad,
                           this->size() / nb_quad);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedNFieldBase<T>::eigen_pixel() const -> Eigen_cmap {
    const auto & nb_quad{this->collection.get_nb_quad()};
    return this->eigen_map(this->nb_components * nb_quad,
                           this->size() / nb_quad);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedNFieldBase<T>::get_pixel_map(const Dim_t & nb_rows)
      -> NFieldMap<T, Mapping::Mut> {
    auto ret_val{
        (nb_rows == -1)
            ? NFieldMap<T, Mapping::Mut>{*this, Iteration::Pixel}
            : NFieldMap<T, Mapping::Mut>{*this, nb_rows, Iteration::Pixel}};
    if (this->collection.is_initialised()) {
      ret_val.initialise();
    }
    return ret_val;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedNFieldBase<T>::get_pixel_map(const Dim_t & nb_rows) const
      -> NFieldMap<T, Mapping::Const> {
    auto ret_val{
        (nb_rows == -1)
            ? NFieldMap<T, Mapping::Const>{*this, Iteration::Pixel}
            : NFieldMap<T, Mapping::Const>{*this, nb_rows, Iteration::Pixel}};
    if (this->collection.is_initialised()) {
      ret_val.initialise();
    }
    return ret_val;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedNFieldBase<T>::get_quad_pt_map(const Dim_t & nb_rows)
      -> NFieldMap<T, Mapping::Mut> {
    auto ret_val{
        (nb_rows == -1)
            ? NFieldMap<T, Mapping::Mut>{*this, Iteration::QuadPt}
            : NFieldMap<T, Mapping::Mut>{*this, nb_rows, Iteration::QuadPt}};
    if (this->collection.is_initialised()) {
      ret_val.initialise();
    }
    return ret_val;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedNFieldBase<T>::get_quad_pt_map(const Dim_t & nb_rows) const
      -> NFieldMap<T, Mapping::Const> {
    auto ret_val{
        (nb_rows == -1)
            ? NFieldMap<T, Mapping::Const>{*this, Iteration::QuadPt}
            : NFieldMap<T, Mapping::Const>{*this, nb_rows, Iteration::QuadPt}};
    if (this->collection.is_initialised()) {
      ret_val.initialise();
    }
    return ret_val;
  }

  template <typename T>
  WrappedNField<T>::WrappedNField(const std::string & unique_name,
                                  NFieldCollection & collection,
                                  Dim_t nb_components,
                                  Eigen::Ref<EigenRep_t> values)
      : Parent{unique_name, collection, nb_components}, values{values} {
    this->current_size = values.size() / this->nb_components;

    if (values.size() != Dim_t(this->nb_components * this->current_size)) {
      std::stringstream error{};
      error << "Size mismatch: the provided array has a size of "
            << values.size()
            << " which is not a multiple of the specified number of components "
               "(nb_components = "
            << this->nb_components << ").";
      throw NFieldError(error.str());
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
      throw NFieldError(error.str());
    }
    this->set_data_ptr(values.data());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto WrappedNField<T>::make_const(const std::string & unique_name,
                                    NFieldCollection & collection,
                                    Dim_t nb_components,
                                    Eigen::Ref<const EigenRep_t> values)
      -> std::unique_ptr<const WrappedNField> {
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> map{
        const_cast<T *>(values.data()), values.rows(), values.cols()};
    return std::make_unique<WrappedNField>(unique_name, collection,
                                           nb_components, map);
  }
  /* ---------------------------------------------------------------------- */
  template <typename T>
  void WrappedNField<T>::set_pad_size(size_t pad_size) {
    std::stringstream error;
    error << "Setting pad size to " << pad_size << " not possible for "
          << "wrapped fields.";
    throw NFieldError(error.str());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void WrappedNField<T>::set_zero() {
    this->values.setZero();
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void WrappedNField<T>::resize(size_t size) {
    const auto expected_size{size * this->get_nb_components() + this->pad_size};
    if (expected_size != this->buffer_size()) {
      std::stringstream error{};
      error << "Wrapped fields cannot be resized. The current wrapped size is "
            << this->buffer_size() << ". Resize to " << expected_size
            << " was attempted.";
      throw NFieldError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  size_t WrappedNField<T>::buffer_size() const {
    return this->values.size();
  }

  /* ---------------------------------------------------------------------- */
  template class TypedNFieldBase<Real>;
  template class TypedNFieldBase<Complex>;
  template class TypedNFieldBase<Int>;
  template class TypedNFieldBase<Uint>;

  template class TypedNField<Real>;
  template class TypedNField<Complex>;
  template class TypedNField<Int>;
  template class TypedNField<Uint>;

  template class WrappedNField<Real>;
  template class WrappedNField<Complex>;
  template class WrappedNField<Int>;
  template class WrappedNField<Uint>;
}  // namespace muGrid
