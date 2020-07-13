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

#include "ccoord_operations.hh"
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
  TypedField<T> & TypedField<T>::operator=(const TypedField & other) {
    Parent::operator=(other);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedField<T> & TypedField<T>::operator=(const Parent & other) {
    Parent::operator=(other);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
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
                                           const Index_t & nb_components,
                                           const std::string & sub_division) {
    if (other.get_nb_components() != nb_components) {
      std::stringstream err_msg{};
      err_msg << "Cannot cast field'" << other.get_name()
              << "', because it has " << other.get_nb_components()
              << " degrees of freedom per sub-point, rather than the "
              << nb_components << " components which are requested.";
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
                           const Index_t & nb_components,
                           const std::string & sub_division) {
    if (other.get_nb_components() != nb_components) {
      std::stringstream err_msg{};
      err_msg << "Cannot cast field'" << other.get_name()
              << "', because it has " << other.get_nb_components()
              << " degrees of freedom per sub-point, rather than the "
              << nb_components << " components which are requested.";
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

    auto && size{this->nb_sub_pts * this->get_nb_pixels()};
    const auto expected_size{size * this->get_nb_components() +
                             this->pad_size};
    if (this->values.size() != expected_size or
        static_cast<Index_t>(this->current_nb_entries) != size) {
      this->current_nb_entries = size;
      this->values.resize(expected_size);
    }
    this->set_data_ptr(this->values.data());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  size_t TypedField<T>::get_buffer_size() const {
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
    if (this->nb_components != 1) {
      throw FieldError("This is not a scalar field. push_back an array.");
    }
    const auto & nb_sub{this->get_nb_sub_pts()};
    this->current_nb_entries += nb_sub;
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
    if (this->nb_components != value.size()) {
      std::stringstream error{};
      error << "You are trying to push an array with " << value.size()
            << "components into a field with " << this->nb_components
            << " components.";
      throw FieldError(error.str());
    }
    const auto & nb_sub{this->get_nb_sub_pts()};
    this->current_nb_entries += nb_sub;
    for (Index_t sub_pt_id{0}; sub_pt_id < nb_sub; ++sub_pt_id) {
      for (Index_t i{0}; i < this->nb_components; ++i) {
        this->values.push_back(value.data()[i]);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedField<T> & TypedField<T>::clone(const std::string & new_name,
                                       const bool & allow_overwrite) const {
    const bool field_exists{this->get_collection().field_exists(new_name)};

    if (field_exists and not allow_overwrite) {
      std::stringstream err_msg{};
      err_msg << "The field '" << new_name
              << "' already exists, and you did not set 'allow_overwrite' "
                 "to true";
      throw FieldError{err_msg.str()};
    }

    TypedField<T> & other{
        field_exists
            ? this->safe_cast(this->get_collection().get_field(new_name),
                              this->nb_components, this->sub_division_tag)
            : this->get_collection().template register_field<T>(
                  new_name, this->nb_components, this->sub_division_tag,
                  this->unit)};

    other = *this;
    return other;
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
    if (not CcoordOps::is_buffer_contiguous(this->get_pixels_shape(),
                                            this->get_pixels_strides())) {
      throw FieldError("Eigen representation is only available for fields with "
                       "contiguous storage.");
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
    if (not CcoordOps::is_buffer_contiguous(this->get_pixels_shape(),
                                            this->get_pixels_strides())) {
      throw FieldError("Eigen representation is only available for fields with "
                       "contiguous storage.");
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
      auto && my_shape{this->get_shape(IterUnit::SubPt)};
      auto && other_shape{other.get_shape(IterUnit::SubPt)};
      if (my_shape != other_shape) {
        std::stringstream s;
        s << "Shape mismatch: Copying a field with shape " << other_shape
          << " onto a field with shape " << my_shape << " is not supported.";
        throw FieldError(s.str());
      }
      auto && my_strides{this->get_strides(IterUnit::SubPt)};
      auto && other_strides{other.get_strides(IterUnit::SubPt)};
      raw_mem_ops::strided_copy(my_shape, other_strides, my_strides,
                                other.data(), this->data_ptr);
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
    if (this->get_nb_entries() == Unknown) {
      throw FieldError("Field has unknown number of entries");
    }
    return this->eigen_map(this->get_nb_entries() * this->nb_components, 1);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_vec() const -> Eigen_cmap {
    if (this->get_nb_entries() == Unknown) {
      throw FieldError("Field has unknown number of entries");
    }
    return this->eigen_map(this->get_nb_entries() * this->nb_components, 1);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_sub_pt() -> Eigen_map {
    if (this->get_nb_entries() == Unknown) {
      throw FieldError("Field has unknown number of entries");
    }
    return this->eigen_map(this->nb_components, this->get_nb_entries());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_sub_pt() const -> Eigen_cmap {
    if (this->get_nb_entries() == Unknown) {
      throw FieldError("Field has unknown number of entries");
    }
    return this->eigen_map(this->nb_components, this->get_nb_entries());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_pixel() -> Eigen_map {
    if (this->get_nb_entries() == Unknown) {
      throw FieldError("Field has unknown number of entries");
    }
    const auto & nb_sub{this->get_nb_sub_pts()};
    return this->eigen_map(this->nb_components * nb_sub,
                           this->get_nb_entries() / nb_sub);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedFieldBase<T>::eigen_pixel() const -> Eigen_cmap {
    if (this->get_nb_entries() == Unknown) {
      throw FieldError("Field has unknown number of entries");
    }
    const auto & nb_sub{this->get_nb_sub_pts()};
    return this->eigen_map(this->nb_components * nb_sub,
                           this->get_nb_entries() / nb_sub);
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
               : FieldMap<T, Mapping::Const>{*this, nb_rows, IterUnit::Pixel};
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
               : FieldMap<T, Mapping::Const>{*this, nb_rows, IterUnit::SubPt};
  }

  template <typename T>
  WrappedField<T>::WrappedField(const std::string & unique_name,
                                FieldCollection & collection,
                                const Index_t & nb_components,
                                const size_t & size, T * ptr,
                                const std::string & sub_division_tag,
                                const Unit & unit,
                                const Shape_t & strides)
      : Parent{unique_name, collection, nb_components, sub_division_tag,
               unit},
        size{size},
        strides{strides} {
    this->current_nb_entries = size / this->nb_components;

    if (static_cast<Index_t>(size) !=
        this->nb_components * this->current_nb_entries) {
      std::stringstream error{};
      error << "Size mismatch: the provided array has a size of " << size
            << " which is not a multiple of the specified number of components "
               "(nb_components = "
            << this->nb_components << ").";
      throw FieldError(error.str());
    }
    if (this->get_nb_entries() !=
        static_cast<Index_t>(this->current_nb_entries)) {
      std::stringstream error{};
      error << "Size mismatch: This field should store "
            << this->nb_components << " component(s) on "
            << this->collection.get_nb_pixels() << " pixels ("
            << this->get_pixels_shape() << " grid) with "
            << this->get_nb_sub_pts()
            << " sub-point(s) each (sub-point tag '" << sub_division_tag
            << "'), i.e. with a total of "
            << this->get_nb_entries() * this->nb_components
            << " scalar values, but you supplied an array of size " << size
            << ".";
      throw FieldError(error.str());
    }
    this->set_data_ptr(ptr);
  }

  template <typename T>
  WrappedField<T>::WrappedField(const std::string & unique_name,
                                FieldCollection & collection,
                                const Shape_t & components_shape,
                                const size_t & size, T * ptr,
                                const std::string & sub_division_tag,
                                const Unit & unit,
                                const Shape_t & strides)
      : Parent{unique_name, collection, components_shape, sub_division_tag,
               unit},
        size{size},
        strides{strides} {
    this->current_nb_entries = size / this->nb_components;

    if (static_cast<Index_t>(size) !=
        this->nb_components * this->current_nb_entries) {
      std::stringstream error{};
      error << "Size mismatch: the provided array has a size of " << size
            << " which is not a multiple of the specified number of components "
               "(nb_components = "
            << this->nb_components << ").";
      throw FieldError(error.str());
    }
    if (this->get_nb_entries() != Index_t(this->current_nb_entries)) {
      std::stringstream error{};
      error << "Size mismatch: This field should store "
            << this->nb_components << " component(s) (shape "
            << this->components_shape << ") on "
            << this->collection.get_nb_pixels() << " pixels ("
            << this->get_pixels_shape() << " grid) with "
            << this->get_nb_sub_pts()
            << " sub-point(s) each (sub-point tag '" << sub_division_tag
            << "'), i.e. with a total of "
            << this->get_nb_entries() * this->nb_components
            << " scalar values, but you supplied an array of size " << size
            << ".";
      throw FieldError(error.str());
    }
    this->set_data_ptr(ptr);
  }

  template <typename T>
  WrappedField<T>::WrappedField(const std::string & unique_name,
                                FieldCollection & collection,
                                const Index_t & nb_components,
                                Eigen::Ref<EigenRep_t> values,
                                const std::string & sub_division_tag,
                                const Unit & unit,
                                const Shape_t & strides)
      : WrappedField{unique_name,
                     collection,
                     nb_components,
                     static_cast<size_t>(values.size()),
                     values.data(),
                     sub_division_tag,
                     unit,
                     strides} {}

  template <typename T>
  WrappedField<T>::WrappedField(const std::string & unique_name,
                                FieldCollection & collection,
                                const Shape_t & components_shape,
                                Eigen::Ref<EigenRep_t> values,
                                const std::string & sub_division_tag,
                                const Unit & unit,
                                const Shape_t & strides)
      : WrappedField{unique_name,
                     collection,
                     components_shape,
                     static_cast<size_t>(values.size()),
                     values.data(),
                     sub_division_tag,
                     unit,
                     strides} {}

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto WrappedField<T>::make_const(const std::string & unique_name,
                                   FieldCollection & collection,
                                   const Index_t & nb_components,
                                   Eigen::Ref<const EigenRep_t> values,
                                   const std::string & sub_division,
                                   const Unit & unit,
                                   const Shape_t & strides)
      -> std::unique_ptr<const WrappedField> {
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> map{
        const_cast<T *>(values.data()), values.rows(), values.cols()};
    return std::make_unique<WrappedField>(
        unique_name, collection, nb_components, map, sub_division, unit,
        strides);
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
    const auto expected_size{size * this->get_nb_components() +
                             this->pad_size};
    if (expected_size != this->get_buffer_size()) {
      std::stringstream error{};
      error << "Wrapped fields cannot be resized. The current wrapped size is "
            << this->get_buffer_size() << ". Resize to " << expected_size
            << " was attempted.";
      throw FieldError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  Shape_t WrappedField<T>::get_strides(const IterUnit & iter_type,
                                       Index_t element_size) const {
    if (this->strides.size() > 0) {
      if (iter_type == IterUnit::SubPt) {
        return this->strides;
      } else {
        throw FieldError("Pixel iteration is not supported for wrapped fields "
                         "with arbitrary strides.");
      }
    } else {
      return Parent::get_strides(iter_type, element_size);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  StorageOrder WrappedField<T>::get_storage_order() const {
    if (this->strides.size() > 0) {
      return StorageOrder::Unknown;
    } else {
      return Parent::get_storage_order();
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  size_t WrappedField<T>::get_buffer_size() const {
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
