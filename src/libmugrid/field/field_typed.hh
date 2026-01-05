/**
 * @file   field_typed.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   10 Aug 2019
 *
 * @brief  Field classes for which the scalar type has been defined
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

#ifndef SRC_LIBMUGRID_FIELD_TYPED_HH_
#define SRC_LIBMUGRID_FIELD_TYPED_HH_

#include "field/field.hh"
#include "collection/field_collection.hh"
#include "core/enums.hh"
#include "memory/array.hh"

#include "Eigen/Dense"

#include <type_traits>
#include <memory>

namespace muGrid {

  //! forward declaration
  template <typename T, Mapping Mutability>
  class FieldMap;
  //! forward declaration
  template <typename T, typename MemorySpace = HostSpace>
  class TypedFieldBase;

  template <typename T, typename MemorySpace>
  class TypedFieldBase : public Field {
    static_assert(std::is_scalar<T>::value or std::is_same<T, Complex>::value,
                  "You can only register fields templated with one of the "
                  "numeric types Real, Complex, Int, or UInt");

   protected:
    /**
     * `Field`s are supposed to only exist in the form of `std::unique_ptr`s
     * held by a `FieldCollection. The `Field` constructor is protected to
     * ensure this. Fields are instantiated through the `register_field`
     * methods FieldCollection.
     * @param unique_name unique field name (unique within a collection)
     * @param collection reference to the holding field collection.
     * @param nb_components number of components to store per quadrature
     *        point
     */
    TypedFieldBase(const std::string & unique_name,
                   FieldCollection & collection, Index_t nb_components,
                   const std::string & sub_division, const Unit & unit)
        : Parent{unique_name, collection, nb_components, sub_division, unit} {}

    /**
     * `Field`s are supposed to only exist in the form of `std::unique_ptr`s
     * held by a `FieldCollection. The `Field` constructor is protected to
     * ensure this. Fields are instantiated through the `register_field`
     * methods FieldCollection.
     * @param unique_name unique field name (unique within a collection)
     * @param collection reference to the holding field collection.
     * @param components_shape number of components to store per quadrature
     * point
     */
    TypedFieldBase(const std::string & unique_name,
                   FieldCollection & collection,
                   const Shape_t & components_shape,
                   const std::string & sub_division, const Unit & unit)
        : Parent{unique_name, collection, components_shape, sub_division,
                 unit} {}

   public:
    //! Memory space type
    using Memory_Space = MemorySpace;

    //! Array type for storage
    using View_t = Array<T, MemorySpace>;

    /**
     * Simple structure used to allow for lazy evaluation of the unary '-' sign.
     * When assiging the the negative of a field to another, as in field_a =
     * -field_b, this structure allows to implement this operation without
     * needing a temporary object holding the negative value of field_b.
     */
    struct Negative {
      //! field on which the unary '-' was applied
      const TypedFieldBase & field;
    };
    //! stored scalar type
    using Scalar = T;

    //! Eigen type used to represent the field's data
    using EigenRep_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    //! Eigen type used to represent the field's data
    using EigenVecRep_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    //! eigen map (handle for EigenRep_t)
    using Eigen_map = Eigen::Map<EigenRep_t>;

    //! eigen const map (handle for EigenRep_t)
    using Eigen_cmap = Eigen::Map<const EigenRep_t>;

    //! eigen vector map (handle for EigenVecRep_t)
    using EigenVec_map = Eigen::Map<EigenVecRep_t>;

    //! eigen vector const map (handle for EigenVecRep_t)
    using EigenVec_cmap = Eigen::Map<const EigenVecRep_t>;

    //! base class
    using Parent = Field;

    //! Default constructor
    TypedFieldBase() = delete;

    //! Copy constructor
    TypedFieldBase(const TypedFieldBase & other) = delete;

    //! Move constructor
    TypedFieldBase(TypedFieldBase && other) = default;

    //! Destructor
    virtual ~TypedFieldBase() = default;

    //! Move assignment operator
    TypedFieldBase & operator=(TypedFieldBase && other) = delete;

    //! Copy assignment operator
    TypedFieldBase & operator=(const TypedFieldBase & other);

    //! Copy assignment operator
    TypedFieldBase & operator=(const Negative & other);

    //! Copy assignment operators
    TypedFieldBase & operator=(const EigenRep_t & other);

    //! Unary negative
    Negative operator-() const;

    //! addition assignment
    TypedFieldBase & operator+=(const TypedFieldBase & other);

    //! subtraction assignment
    TypedFieldBase & operator-=(const TypedFieldBase & other);

    //! return the unified type descriptor for this field's element type
    TypeDescriptor get_type_descriptor() const final {
      return type_to_descriptor<T>();
    }

    //! return the size of the elementary field entry in bytes
    std::size_t get_element_size_in_bytes() const final {
      return sizeof(T);
    }

    //! return a vector map onto the underlying data (host-space only)
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, Eigen_map> eigen_mat();
    //! return a const vector map onto the underlying data (host-space only)
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, Eigen_cmap> eigen_mat() const;

    //! return a vector map onto the underlying data (host-space only)
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, EigenVec_map> eigen_vec();
    //! return a const vector map onto the underlying data (host-space only)
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, EigenVec_cmap> eigen_vec() const;

    /**
     * return a matrix map onto the underlying data with one column per
     * quadrature point (host-space only)
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, Eigen_map> eigen_sub_pt();
    /**
     * return a const matrix map onto the underlying data with one column per
     * quadrature point (host-space only)
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, Eigen_cmap> eigen_sub_pt() const;

    /**
     * return a matrix map onto the underlying data with one column per
     * pixel (host-space only)
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, Eigen_map> eigen_pixel();
    /**
     * return a const matrix map onto the underlying data with one column per
     * pixel (host-space only)
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, Eigen_cmap> eigen_pixel() const;

    template <typename T_int, Mapping Mutability>
    friend class FieldMap;

    /**
     * convenience function returns a map of this field, iterable per pixel.
     * (host-space only)
     *
     * @param nb_rows optional specification of the number of rows for the
     * iterate. If left to default value, a matrix of shape `nb_components`
     * × `nb_quad_pts` is used
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, FieldMap<T, Mapping::Mut>>
    get_pixel_map(const Index_t & nb_rows = Unknown);

    /**
     * convenience function returns a const map of this field, iterable per
     * pixel. (host-space only)
     *
     * @param nb_rows optional specification of the number of rows for the
     * iterate. If left to default value, a matrix of shape `nb_components`
     * × `nb_quad_pts` is used
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, FieldMap<T, Mapping::Const>>
    get_pixel_map(const Index_t & nb_rows = Unknown) const;

    /**
     * convenience function returns a map of this field, iterable per quadrature
     * point. (host-space only)
     *
     * @param nb_rows optional specification of the number of rows for the
     * iterate. If left to default value, a column vector is used
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, FieldMap<T, Mapping::Mut>>
    get_sub_pt_map(const Index_t & nb_rows = Unknown);

    /**
     * convenience function returns a const  map of this field, iterable per
     * quadrature point. (host-space only)
     *
     * @param nb_rows optional specification of the number of rows for the
     * iterate. If left to default value, a column vector is used
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, FieldMap<T, Mapping::Const>>
    get_sub_pt_map(const Index_t & nb_rows = Unknown) const;

    /**
     * Get the raw data pointer. Only available for host-space fields.
     * Don't use unless interfacing with external libs.
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, T *> data() {
      return this->values.data();
    }

    /**
     * Get the raw data pointer (const). Only available for host-space fields.
     * Don't use unless interfacing with external libs.
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, const T *> data() const {
      return this->values.data();
    }

    /**
     * return a pointer to the raw data. Don't use unless interfacing with
     * external libs.
     *
     * @param assert_host_memory If true (default), throws an error if the
     *        field is on device memory. Set to false only when passing the
     *        pointer to CUDA-aware libraries (e.g., CUDA-aware MPI).
     **/
    void * get_void_data_ptr(bool assert_host_memory = true) const final;

    //! non-const eigen_map with arbitrary sizes (host-space only)
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, Eigen_map>
    eigen_map(const Index_t & nb_rows, const Index_t & nb_cols);
    //! const eigen_map with arbitrary sizes (host-space only)
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, Eigen_cmap>
    eigen_map(const Index_t & nb_rows, const Index_t & nb_cols) const;

    //! Get the underlying Array for use in kernels
    View_t & view() { return this->values; }
    //! Get the underlying Array (const) for use in kernels
    const View_t & view() const { return this->values; }

    /**
     * Deep copy from another field (potentially in a different memory space).
     * This performs a host-device or device-host transfer as needed.
     */
    template <typename OtherSpace>
    void deep_copy_from(const TypedFieldBase<T, OtherSpace> & src);

    /**
     * Check if field resides on device (GPU) memory.
     * Implementation uses compile-time type trait.
     */
    bool is_on_device() const final {
      return is_device_space_v<MemorySpace>;
    }

    /**
     * Get DLPack device type for this field's memory space.
     */
    int get_dlpack_device_type() const final {
      return dlpack_device_type_v<MemorySpace>;
    }

    /**
     * Get device ID for multi-GPU systems.
     * Returns the device ID from the field collection's Device.
     */
    int get_device_id() const final {
      return this->get_collection().get_device().get_device_id();
    }

    /**
     * Get device string for Python interoperability.
     * Returns "cpu", "cuda:N", or "rocm:N" where N is the device ID.
     */
    std::string get_device_string() const final {
      std::string base{device_name<MemorySpace>()};
      if constexpr (is_device_space_v<MemorySpace>) {
        return base + ":" + std::to_string(this->get_device_id());
      } else {
        return base;
      }
    }

   protected:
    //! Array storage for the raw field data
    View_t values{};
  };

  //! Forward declaration of TypedField (default arg already in field_collection.hh)
  template <typename T, typename MemorySpace>
  class TypedField;

  /**
   * A `muGrid::TypedField` holds a certain number of components (scalars of
   * type `T` per quadrature point of a `muGrid::FieldCollection`'s domain.
   *
   * @tparam T type of scalar to hold. Must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`.
   * @tparam MemorySpace Memory space (HostSpace by default, or CUDASpace/ROCmSpace for GPU)
   */
  template <typename T, typename MemorySpace>
  class TypedField : public TypedFieldBase<T, MemorySpace> {
   protected:
    /**
     * `Field`s are supposed to only exist in the form of `std::unique_ptr`s
     * held by a FieldCollection. The `Field` constructor is protected to
     * ensure this.
     * @param unique_name unique field name (unique within a collection)
     * @param nb_components number of components to store per quadrature
     * point
     * @param collection reference to the holding field collection.
     */
    TypedField(const std::string & unique_name, FieldCollection & collection,
               const Index_t & nb_components, const std::string & sub_division,
               const Unit & unit)
        : Parent{unique_name, collection, nb_components, sub_division, unit} {}

    /**
     * `Field`s are supposed to only exist in the form of `std::unique_ptr`s
     * held by a FieldCollection. The `Field` constructor is protected to
     * ensure this.
     * @param unique_name unique field name (unique within a collection)
     * @param collection reference to the holding field collection.
     * @param components_shape number of components to store per quadrature
     * point
     * @param storage_oder in-memory storage order of the components
     */
    TypedField(const std::string & unique_name, FieldCollection & collection,
               const Shape_t & components_shape,
               const std::string & sub_division, const Unit & unit)
        : Parent{unique_name, collection, components_shape, sub_division,
                 unit} {}

   public:
    //! base class
    using Parent = TypedFieldBase<T, MemorySpace>;

    //! Eigen type to represent the field's data
    using EigenRep_t = typename Parent::EigenRep_t;

    //! convenience alias
    using Negative = typename Parent::Negative;

    //! Default constructor
    TypedField() = delete;

    //! Copy constructor
    TypedField(const TypedField & other) = delete;

    //! Move constructor
    TypedField(TypedField && other) = delete;

    //! Destructor
    virtual ~TypedField() = default;

    //! Move assignment operator
    TypedField & operator=(TypedField && other) = delete;

    //! Copy assignment operator
    TypedField & operator=(const TypedField & other);

    //! Copy assignment operator
    TypedField & operator=(const Parent & other);

    //! Copy assignment operator
    TypedField & operator=(const Negative & other);

    //! Copy assignment operator
    TypedField & operator=(const EigenRep_t & other);

    void set_zero() final;

    //! cast a reference to a base type to this type, with full checks
    static TypedField & safe_cast(Field & other);

    //! cast a const reference to a base type to this type, with full checks
    static const TypedField & safe_cast(const Field & other);

    /**
     * cast a reference to a base type to this type safely, plus check whether
     * it has the right number of components
     */
    static TypedField & safe_cast(Field & other, const Index_t & nb_components,
                                  const std::string & sub_division);

    /**
     * cast a const reference to a base type to this type safely, plus check
     * whether it has the right number of components
     */
    static const TypedField & safe_cast(const Field & other,
                                        const Index_t & nb_components,
                                        const std::string & sub_division);

    size_t get_buffer_size() const final;

    /**
     * add a new scalar value at the end of the field (incurs runtime cost, do
     * not use this in any hot loop). If your field has more than one quadrature
     * point per pixel the same scalar value is pushed back on all quadrature
     * points of the pixel. (host-space only)
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, void>
    push_back(const T & value);

    /**
     * add a new scalar value at the end of the field (incurs runtime cost, do
     * not use this in any hot loop). Even if you have several quadrature points
     * per pixel you push back only a single value on a single quadrature point.
     * Thus you can push back different values on quadrature points belongign to
     * the same pixel. (host-space only)
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, void>
    push_back_single(const T & value);

    /**
     * add a new non-scalar value at the end of the field (incurs runtime cost,
     * do not use this in any hot loop) If your field has more than one
     * quadrature point per pixel the same non-scalar value is pushed back on
     * all quadrature points of the pixel. (host-space only)
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, void>
    push_back(const Eigen::Ref<
              const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> & value);

    /**
     * add a new non-scalar value at the end of the field (incurs runtime cost,
     * do not use this in any hot loop) Even if you have several quadrature
     * points per pixel you push back only a single non-scalar value on a single
     * quadrature point. Thus you can push back different values on quadrature
     * points belongign to the same pixel. (host-space only)
     */
    template <typename M = MemorySpace>
    std::enable_if_t<is_host_space_v<M>, void>
    push_back_single(
        const Eigen::Ref<
            const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> & value);

    /**
     * perform a full copy of a field
     * @param new_name the name under which the new field will be stored
     * @param allow_overwrite by default, this function throws an error if the
     * destination field already exists to avoid accidental clobbering of
     * fields. If set to true, the copy will be made into the existing field.
     */
    TypedField & clone(const std::string & new_name,
                       const bool & allow_overwrite = false) const;

    //! give access to collections
    friend FieldCollection;

   protected:
    void resize() final;
  };

  //! Alias for real-valued host fields
  using RealField = TypedField<Real, HostSpace>;
  //! Alias for complex-valued host fields
  using ComplexField = TypedField<Complex, HostSpace>;
  //! Alias for integer-valued host fields
  using IntField = TypedField<Int, HostSpace>;
  //! Alias for unsigned integer-valued host fields
  using UintField = TypedField<Uint, HostSpace>;
  //! Alias for index-valued host fields
  using IndexField = TypedField<Index_t, HostSpace>;

  //! Alias for real-valued device fields (CUDA/HIP)
  using RealFieldDevice = TypedField<Real, DefaultDeviceSpace>;
  //! Alias for complex-valued device fields
  using ComplexFieldDevice = TypedField<Complex, DefaultDeviceSpace>;
  //! Alias for integer-valued device fields
  using IntFieldDevice = TypedField<Int, DefaultDeviceSpace>;
  //! Alias for unsigned integer-valued device fields
  using UintFieldDevice = TypedField<Uint, DefaultDeviceSpace>;
  //! Alias for index-valued device fields
  using IndexFieldDevice = TypedField<Index_t, DefaultDeviceSpace>;

  //! Alias for real-valued fields with configurable memory space
  template <typename MemorySpace = HostSpace>
  using RealFieldT = TypedField<Real, MemorySpace>;
  //! Alias for complex-valued fields with configurable memory space
  template <typename MemorySpace = HostSpace>
  using ComplexFieldT = TypedField<Complex, MemorySpace>;
  //! Alias for integer-valued fields with configurable memory space
  template <typename MemorySpace = HostSpace>
  using IntFieldT = TypedField<Int, MemorySpace>;
  //! Alias for unsigned integer-valued fields with configurable memory space
  template <typename MemorySpace = HostSpace>
  using UintFieldT = TypedField<Uint, MemorySpace>;

  /**
   * Free function for deep copying between fields in different memory spaces.
   *
   * This function handles layout conversion between AoS (Array of Structures,
   * used by host/CPU) and SoA (Structure of Arrays, used by device/GPU).
   *
   * For same-space copies (host-host or device-device), a fast byte copy is
   * used. For cross-space copies (host-device or device-host), layout
   * conversion is performed so that the logical field values are preserved.
   */
  template <typename T, typename DstSpace, typename SrcSpace>
  void deep_copy(TypedFieldBase<T, DstSpace> & dst,
                 const TypedFieldBase<T, SrcSpace> & src) {
    if (dst.view().size() != src.view().size()) {
      throw FieldError("Size mismatch in deep_copy");
    }

    const auto src_order = src.get_storage_order();
    const auto dst_order = dst.get_storage_order();

    // Same storage order: use fast byte copy
    if (src_order == dst_order) {
      muGrid::deep_copy(dst.view(), src.view());
      return;
    }

    // Different storage orders: need layout conversion
    // This happens when copying between host (AoS) and device (SoA)

    const Index_t nb_buffer_pixels =
        src.get_collection().get_nb_buffer_pixels();
    const Index_t nb_components = src.get_nb_components();
    const Index_t nb_sub_pts = src.get_nb_sub_pts();
    const Index_t nb_dof_per_pixel = nb_components * nb_sub_pts;
    const Index_t total_size = nb_buffer_pixels * nb_dof_per_pixel;

    if (total_size == 0) {
      return;
    }

    // Allocate temporary host buffer for conversion
    Array<T, HostSpace> tmp(total_size);

    if constexpr (is_host_space_v<SrcSpace> && is_device_space_v<DstSpace>) {
      // Host (AoS) -> Device (SoA): Convert on host, then copy to device
      // AoS: data[pixel * nb_dof_per_pixel + dof]
      // SoA: data[dof * nb_buffer_pixels + pixel]
      const T * src_data = src.view().data();
      T * tmp_data = tmp.data();
      for (Index_t pixel = 0; pixel < nb_buffer_pixels; ++pixel) {
        for (Index_t dof = 0; dof < nb_dof_per_pixel; ++dof) {
          tmp_data[dof * nb_buffer_pixels + pixel] =
              src_data[pixel * nb_dof_per_pixel + dof];
        }
      }
      // Copy converted SoA data to device
      muGrid::deep_copy(dst.view(), tmp);
    } else if constexpr (is_device_space_v<SrcSpace> &&
                         is_host_space_v<DstSpace>) {
      // Device (SoA) -> Host (AoS): Copy to host, then convert
      // Copy SoA data from device to temp buffer
      muGrid::deep_copy(tmp, src.view());
      // Convert SoA -> AoS
      const T * tmp_data = tmp.data();
      T * dst_data = dst.view().data();
      for (Index_t pixel = 0; pixel < nb_buffer_pixels; ++pixel) {
        for (Index_t dof = 0; dof < nb_dof_per_pixel; ++dof) {
          dst_data[pixel * nb_dof_per_pixel + dof] =
              tmp_data[dof * nb_buffer_pixels + pixel];
        }
      }
    } else {
      // Should not reach here - both host or both device with different orders
      throw FieldError(
          "Unexpected storage order mismatch in same-space deep_copy");
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  Field & FieldCollection::register_field_helper(
      const std::string & unique_name, const Index_t & nb_components,
      const std::string & sub_division_tag, const Unit & unit,
      bool allow_existing) {
    static_assert(std::is_scalar<T>::value or std::is_same<T, Complex>::value,
                  "You can only register fields templated with one of the "
                  "numeric types Real, Complex, Int, or UInt");
    if (this->field_exists(unique_name)) {
      if (allow_existing) {
        auto & field{*this->fields[unique_name]};
        field.assert_type_descriptor(type_to_descriptor<T>());
        if (field.get_nb_components() != nb_components) {
          std::stringstream error{};
          error << "You can't change the number of components of a field "
                << "by re-registering it. Field '" << unique_name << "' has "
                << field.get_nb_components()
                << " components and you are trying to register it with "
                << nb_components << " components.";
          throw FieldCollectionError(error.str());
        }
        if (field.get_sub_division_tag() != sub_division_tag) {
          throw FieldCollectionError(
              "You can't change the sub-division tag of a field "
              "by re-registering it.");
        }
        if (field.get_physical_unit() != unit) {
          throw FieldCollectionError(
              "You can't change the physical unit of a field "
              "by re-registering it.");
        }
        return field;
      } else {
        std::stringstream error{};
        error << "A Field of name '" << unique_name
              << "' is already registered in this field collection. "
              << "Currently registered fields: ";
        std::string prelude{""};
        for (const auto & name_field_pair : this->fields) {
          error << prelude << '\'' << name_field_pair.first << '\'';
          prelude = ", ";
        }
        throw FieldCollectionError(error.str());
      }
    }

    //! If you get a compiler warning about narrowing conversion on the
    //! following line, please check whether you are creating a TypedField with
    //! the number of components specified in 'int' rather than 'size_t'.
    Field * raw_ptr{nullptr};
    if (this->device.is_device()) {
      raw_ptr = new TypedField<T, DefaultDeviceSpace>{
          unique_name, *this, nb_components, sub_division_tag, unit};
    } else {
      raw_ptr = new TypedField<T, HostSpace>{
          unique_name, *this, nb_components, sub_division_tag, unit};
    }
    Field & retref{*raw_ptr};
    Field_ptr field{raw_ptr};
    if (this->initialised) {
      retref.resize();
    }
    this->fields[unique_name] = std::move(field);
    return retref;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  Field & FieldCollection::register_field_helper(
      const std::string & unique_name, const Shape_t & components_shape,
      const std::string & sub_division_tag, const Unit & unit,
      bool allow_existing) {
    static_assert(std::is_scalar<T>::value or std::is_same<T, Complex>::value,
                  "You can only register fields templated with one of the "
                  "numeric types Real, Complex, Int, or UInt");
    if (this->field_exists(unique_name)) {
      if (allow_existing) {
        auto & field{*this->fields[unique_name]};
        field.assert_type_descriptor(type_to_descriptor<T>());
        if (field.get_components_shape() != components_shape) {
          throw FieldCollectionError(
              "You can't change the shape of a field by re-registering it.");
        }
        if (field.get_sub_division_tag() != sub_division_tag) {
          throw FieldCollectionError(
              "You can't change the sub-division tag of a field "
              "by re-registering it.");
        }
        if (field.get_physical_unit() != unit) {
          throw FieldCollectionError(
              "You can't change the physical unit of a field "
              "by re-registering it.");
        }
        return field;
      } else {
        std::stringstream error{};
        error << "A Field of name '" << unique_name
              << "' is already registered in this field collection. "
              << "Currently registered fields: ";
        std::string prelude{""};
        for (const auto & name_field_pair : this->fields) {
          error << prelude << '\'' << name_field_pair.first << '\'';
          prelude = ", ";
        }
        throw FieldCollectionError(error.str());
      }
    }

    //! If you get a compiler warning about narrowing conversion on the
    //! following line, please check whether you are creating a TypedField with
    //! the number of components specified in 'int' rather than 'size_t'.
    Field * raw_ptr{nullptr};
    if (this->device.is_device()) {
      raw_ptr = new TypedField<T, DefaultDeviceSpace>{
          unique_name, *this, components_shape, sub_division_tag, unit};
    } else {
      raw_ptr = new TypedField<T, HostSpace>{
          unique_name, *this, components_shape, sub_division_tag, unit};
    }
    Field & retref{*raw_ptr};
    Field_ptr field{raw_ptr};
    if (this->initialised) {
      retref.resize();
    }
    this->fields[unique_name] = std::move(field);
    return retref;
  }

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FIELD_TYPED_HH_
