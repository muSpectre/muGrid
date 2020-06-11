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
#include "raw_memory_operations.hh"

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

  template <typename T>
  WrappedField<T> & WrappedField<T>::operator=(const Parent & other) {
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
  void TypedField<T>::set_pad_size(const size_t & pad_size) {
    this->pad_size = pad_size;
    this->resize();
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
                                           const Index_t & nb_dof_per_sub_pt,
                                           const std::string & sub_division) {
    if (other.get_nb_dof_per_sub_pt() != nb_dof_per_sub_pt) {
      std::stringstream err_msg{};
      err_msg << "Cannot cast field'" << other.get_name()
              << "', because it has " << other.get_nb_dof_per_sub_pt()
              << " degrees of freedom per sub-point, rather than the "
              << nb_dof_per_sub_pt << " components which are requested.";
      throw FieldError(err_msg.str());
    }
    if (other.get_sub_division_tag() != sub_division) {
      std::stringstream err_msg{};
      err_msg << "Cannot cast field'" << other.get_name()
              << "', because it's subdivision is '"
              << other.get_sub_division_tag() << "', rather than "
              << sub_division << ", which are requested.";
      throw FieldError(err_msg.str());
    }
    return TypedField::safe_cast(other);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  const TypedField<T> &
  TypedField<T>::safe_cast(const Field & other,
                           const Index_t & nb_dof_per_sub_pt,
                           const std::string & sub_division) {
    if (other.get_nb_dof_per_sub_pt() != nb_dof_per_sub_pt) {
      std::stringstream err_msg{};
      err_msg << "Cannot cast field'" << other.get_name()
              << "', because it has " << other.get_nb_dof_per_sub_pt()
              << " degrees of freedom per sub-point, rather than the "
              << nb_dof_per_sub_pt << " components which are requested.";
      throw FieldError(err_msg.str());
    }
    if (other.get_sub_division_tag() != sub_division) {
      std::stringstream err_msg{};
      err_msg << "Cannot cast field'" << other.get_name()
              << "', because it's subdivision is '"
              << other.get_sub_division_tag() << "', rather than "
              << sub_division << ", which are requested.";
      throw FieldError(err_msg.str());
    }
    return TypedField::safe_cast(other);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedField<T>::resize() {
    if (not this->has_nb_sub_pts()) {
      std::stringstream error_message{};
      error_message << "Can't compute the size of field '" << this->get_name()
                    << "' because the number of points per pixel for "
                       "subdivisions tagged '"
                    << this->get_sub_division_tag() << "' is not yet known.";
      throw FieldError(error_message.str());
    }

    auto && size{this->get_nb_entries()};
    const auto expected_size{size * this->get_nb_dof_per_sub_pt() +
                             this->pad_size};
    if (this->values.size() != expected_size or
        static_cast<Index_t>(this->current_size) != size) {
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
      throw FieldError("push_back() makes no sense on global fields (you can't "
                       "add individual pixels");
    }
    if (not this->has_nb_sub_pts()) {
      throw FieldError("Cannot push_back into a field before the number of "
                       "sub-division points has been set for it");
    }
    if (this->nb_dof_per_sub_pt != 1) {
      throw FieldError("This is not a scalar field. push_back an array.");
    }
    const auto & nb_sub{this->get_nb_sub_pts()};
    this->current_size += nb_sub;
    for (Index_t sub_pt_id{0}; sub_pt_id < nb_sub; ++sub_pt_id) {
      this->values.push_back(value);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void TypedField<T>::push_back(
      const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> &
          value) {
    if (this->is_global()) {
      throw FieldError("push_back() makes no sense on global fields (you can't "
                       "add individual pixels");
    }
    if (not this->has_nb_sub_pts()) {
      throw FieldError("Cannot push_back into a field before the number of "
                       "sub-division points has bee set for.");
    }
    if (this->nb_dof_per_sub_pt != value.size()) {
      std::stringstream error{};
      error << "You are trying to push an array with " << value.size()
            << "components into a field with " << this->nb_dof_per_sub_pt
            << " components.";
      throw FieldError(error.str());
    }
    const auto & nb_sub{this->get_nb_sub_pts()};
    this->current_size += nb_sub;
    for (Index_t sub_pt_id{0}; sub_pt_id < nb_sub; ++sub_pt_id) {
      for (Index_t i{0}; i < this->nb_dof_per_sub_pt; ++i) {
        this->values.push_back(value.data()[i]);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_map(const Index_t & nb_rows,
                                    const Index_t & nb_cols) -> Eigen_map {
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
  auto TypedFieldBase<T>::eigen_map(const Index_t & nb_rows,
                                    const Index_t & nb_cols) const
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
  TypedFieldBase<T> &
  TypedFieldBase<T>::operator=(const TypedFieldBase & other) {
    switch (this->collection.get_domain()) {
    case FieldCollection::ValidityDomain::Local: {
      this->eigen_vec() = other.eigen_vec();
      break;
    }
    case FieldCollection::ValidityDomain::Global: {
      auto && my_strides{this->get_strides(IterUnit::Pixel)};
      auto && other_strides{other.get_strides(IterUnit::Pixel)};
      if (my_strides == other_strides) {
        this->eigen_vec() = other.eigen_vec();
      } else {
        raw_mem_ops::strided_copy(this->get_shape(IterUnit::Pixel),
                                  my_strides, other_strides, other.data(),
                                  this->data_ptr);
      }
      break;
    }
    default:
      throw FieldError("Unknown ValidityDomain type");
      break;
    }
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
  TypedFieldBase<T> &
  TypedFieldBase<T>::operator+=(const TypedFieldBase & other) {
    this->eigen_vec() += other.eigen_vec();
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedFieldBase<T> &
  TypedFieldBase<T>::operator-=(const TypedFieldBase & other) {
    this->eigen_vec() -= other.eigen_vec();
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_vec() -> Eigen_map {
    return this->eigen_map(this->size() * this->nb_dof_per_sub_pt, 1);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_vec() const -> Eigen_cmap {
    return this->eigen_map(this->size() * this->nb_dof_per_sub_pt, 1);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_sub_pt() -> Eigen_map {
    return this->eigen_map(this->nb_dof_per_sub_pt, this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_sub_pt() const -> Eigen_cmap {
    return this->eigen_map(this->nb_dof_per_sub_pt, this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_pixel() -> Eigen_map {
    const auto & nb_sub{this->get_nb_sub_pts()};
    return this->eigen_map(this->nb_dof_per_sub_pt * nb_sub,
                           this->size() / nb_sub);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_pixel() const -> Eigen_cmap {
    const auto & nb_sub{this->get_nb_sub_pts()};
    return this->eigen_map(this->nb_dof_per_sub_pt * nb_sub,
                           this->size() / nb_sub);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::get_pixel_map(const Index_t & nb_rows)
      -> FieldMap<T, Mapping::Mut> {
    return (nb_rows == -1)
               ? FieldMap<T, Mapping::Mut>{*this, IterUnit::Pixel}
               : FieldMap<T, Mapping::Mut>{*this, nb_rows, IterUnit::Pixel};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::get_pixel_map(const Index_t & nb_rows) const
      -> FieldMap<T, Mapping::Const> {
    return (nb_rows == -1)
               ? FieldMap<T, Mapping::Const>{*this, IterUnit::Pixel}
               : FieldMap<T, Mapping::Const>{*this, nb_rows,
                                             IterUnit::Pixel};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::get_sub_pt_map(const Index_t & nb_rows)
      -> FieldMap<T, Mapping::Mut> {
    return (nb_rows == Unknown)
               ? FieldMap<T, Mapping::Mut>{*this, IterUnit::SubPt}
               : FieldMap<T, Mapping::Mut>{*this, nb_rows, IterUnit::SubPt};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::get_sub_pt_map(const Index_t & nb_rows) const
      -> FieldMap<T, Mapping::Const> {
    return (nb_rows == -1)
               ? FieldMap<T, Mapping::Const>{*this, IterUnit::SubPt}
               : FieldMap<T, Mapping::Const>{*this, nb_rows,
                                             IterUnit::SubPt};
  }

  template <typename T>
  WrappedField<T>::WrappedField(const std::string & unique_name,
                                FieldCollection & collection,
                                const Index_t & nb_dof_per_sub_pt,
                                const size_t & size, T * ptr,
                                const std::string & sub_division_tag,
                                const Unit & unit)
      : Parent{unique_name, collection, nb_dof_per_sub_pt, sub_division_tag,
               unit},
        size{static_cast<size_t>(size)} {
    this->current_size = size / this->nb_dof_per_sub_pt;

    if (size != this->nb_dof_per_sub_pt * this->current_size) {
      std::stringstream error{};
      error << "Size mismatch: the provided array has a size of " << size
            << " which is not a multiple of the specified number of components "
               "(nb_dof_per_sub_pt = "
            << this->nb_dof_per_sub_pt << ").";
      throw FieldError(error.str());
    }
    if (this->get_nb_entries() != Index_t(this->current_size)) {
      std::stringstream error{};
      error << "Size mismatch: This field should store "
            << this->nb_dof_per_sub_pt << " component(s) on "
            << this->collection.get_nb_pixels() << " pixels/voxels with "
            << this->get_nb_sub_pts()
            << " sub point(s) each, i.e. with a total of "
            << this->collection.get_nb_pixels() * this->nb_dof_per_sub_pt
            << " scalar values, but you supplied an array of size " << size
            << '.';
      throw FieldError(error.str());
    }
    this->set_data_ptr(ptr);
  }

  template <typename T>
  WrappedField<T>::WrappedField(const std::string & unique_name,
                                FieldCollection & collection,
                                const Index_t & nb_dof_per_sub_pt,
                                Eigen::Ref<EigenRep_t> values,
                                const std::string & sub_division_tag,
                                const Unit & unit)
      : WrappedField{unique_name,
                     collection,
                     nb_dof_per_sub_pt,
                     static_cast<size_t>(values.size()),
                     values.data(),
                     sub_division_tag,
                     unit} {}

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto WrappedField<T>::make_const(const std::string & unique_name,
                                   FieldCollection & collection,
                                   Index_t nb_dof_per_sub_pt,
                                   Eigen::Ref<const EigenRep_t> values,
                                   const std::string & sub_division,
                                   const Unit & unit)
      -> std::unique_ptr<const WrappedField> {
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> map{
        const_cast<T *>(values.data()), values.rows(), values.cols()};
    return std::make_unique<WrappedField>(
        unique_name, collection, nb_dof_per_sub_pt, map, sub_division, unit);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void WrappedField<T>::set_pad_size(const size_t & pad_size) {
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
  void WrappedField<T>::resize() {
    auto && size{this->get_nb_entries()};
    const auto expected_size{size * this->get_nb_dof_per_sub_pt() +
                             this->pad_size};
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
