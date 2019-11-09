/**
 * @file   mapped_nfield.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   04 Sep 2019
 *
 * @brief  convenience class to deal with data structures common to most
 *         internal variable fields in materials
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

#ifndef SRC_LIBMUGRID_MAPPED_NFIELD_HH_
#define SRC_LIBMUGRID_MAPPED_NFIELD_HH_

#include "nfield_map_static.hh"
#include "nfield_collection.hh"
#include "nfield_typed.hh"

#include <string>

namespace muGrid {

  /**
   * MappedNFields are a combination of a field and an associated map, and as
   * such it does not introduce any new functionality that Fields and FieldMaps
   * do not already possess. They provide a convenience structure for the
   * default use case of internal variables, which are typically used only by a
   * single material and always the same way.
   */
  template <class FieldMapType>
  class MappedField {
   public:
    //! stored scalar type
    using Scalar = typename FieldMapType::Scalar;
    //! return type for iterators over this- map
    using Return_t = typename FieldMapType::template Return_t<
        FieldMapType::FieldMutability()>;
    //! iterator over this map
    using iterator = typename FieldMapType::iterator;
    //! constant iterator over this map
    using const_iterator = typename FieldMapType::const_iterator;
    //! detemine at compile time whether the field map is statically sized
    constexpr static bool IsStatic() { return FieldMapType::IsStatic(); }
    //! Default constructor
    MappedField() = delete;

    /**
     * Constructor with name and collection for statically sized mapped fields
     */
    template <bool StaticConstructor = IsStatic(),
              std::enable_if_t<StaticConstructor, int> = 0>
    MappedField(const std::string & unique_name, NFieldCollection & collection)
        : nb_components{compute_nb_components_static(unique_name, collection)},
          field(collection.register_field<Scalar>(unique_name,
                                                  this->nb_components)),
          map{this->field} {
      static_assert(
          StaticConstructor == IsStatic(),
          "StaticConstructor is a SFINAE parameter, do not touch it.");
    }

    /**
     * Constructor for dynamically sized mapped field
     *
     * @param unique_name unique identifier for this field
     * @param nb_rows number of rows for the iterates
     * @param nb_cols number of columns for the iterates
     * @param iter_type whether to iterate over pixels or quadrature points
     * @param collection collection where the field is to be registered
     */
    template <bool StaticConstructor = IsStatic(),
              std::enable_if_t<not StaticConstructor, int> = 0>
    MappedField(const std::string & unique_name, const Dim_t & nb_rows,
                const Dim_t & nb_cols, const Iteration & iter_type,
                NFieldCollection & collection)
        : nb_components{compute_nb_components_dynamic(
              nb_rows, nb_cols, iter_type, unique_name, collection)},
          field(collection.register_field<Scalar>(unique_name,
                                                  this->nb_components)),
          map{this->field, nb_rows, iter_type} {
      static_assert(
          StaticConstructor == IsStatic(),
          "StaticConstructor is a SFINAE parameter, do not touch it.");
    }

    //! Copy constructor
    MappedField(const MappedField & other) = delete;

    //! Move constructor
    MappedField(MappedField && other) = default;

    //! Destructor
    virtual ~MappedField() = default;

    //! Copy assignment operator
    MappedField & operator=(const MappedField & other) = delete;

    //! Move assignment operator
    MappedField & operator=(MappedField && other) = default;

    //! random access operator
    Return_t operator[](size_t index) { return this->map[index]; }

    //! stl
    iterator begin() { return this->map.begin(); }

    //! stl
    iterator end() { return this->map.end(); }

    //! stl
    const_iterator begin() const { return this->map.begin(); }

    //! stl
    const_iterator end() const { return this->map.end(); }

    //! return a reference to the mapped field
    TypedNField<Scalar> & get_field() { return this->field; }

    //! return a reference to the map
    FieldMapType & get_map() { return this->map; }

   protected:
    /**
     * evaluate and return the number of components the dynamically mapped field
     * needs to store per quadrature point
     */
    template <bool StaticConstructor = IsStatic(),
              std::enable_if_t<not StaticConstructor, int> = 0>
    static Dim_t compute_nb_components_dynamic(const Dim_t & nb_rows,
                                               const Dim_t & nb_cols,
                                               const Iteration & iter_type,
                                               const std::string & unique_name,
                                               NFieldCollection & collection) {
      static_assert(
          StaticConstructor == IsStatic(),
          "StaticConstructor is a SFINAE parameter, do not touch it.");

      const Dim_t dof_per_quad_pt{nb_rows * nb_cols};
      switch (iter_type) {
      case Iteration::QuadPt: {
        return dof_per_quad_pt;
        break;
      }
      case Iteration::Pixel: {
        if (not collection.has_nb_quad()) {
          throw NFieldMapError("Can't create a pixel map for field '" +
                               unique_name +
                               "' before the number of quadrature points has "
                               "been set for the field collection.");
        }
        const auto & nb_quad{collection.get_nb_quad()};
        if ((dof_per_quad_pt / nb_quad) * nb_quad != dof_per_quad_pt) {
          std::stringstream error{};
          error << "Inconsistent input to create a mapped field: the number of "
                   "components per iterate ("
                << dof_per_quad_pt << " = " << nb_rows << " × " << nb_cols
                << ") is not a multiple of the number of quad points ("
                << nb_quad << ").";
          throw NFieldMapError(error.str());
        }
        return dof_per_quad_pt / collection.get_nb_quad();
        break;
      }
      default:
        throw NFieldMapError("unknown iteration type");
        break;
      }
    }

    /**
     * evaluate and return the number of components the statically mapped field
     * needs to store per quadrature point
     */
    template <bool StaticConstructor = IsStatic(),
              std::enable_if_t<StaticConstructor, int> = 0>
    static Dim_t compute_nb_components_static(const std::string & unique_name,
                                              NFieldCollection & collection) {
      static_assert(
          StaticConstructor == IsStatic(),
          "StaticConstructor is a SFINAE parameter, do not touch it.");
      switch (FieldMapType::GetIterationType()) {
      case Iteration::QuadPt: {
        return FieldMapType::Stride();
        break;
      }
      case Iteration::Pixel: {
        if (not collection.has_nb_quad()) {
          throw NFieldMapError("Can't create a pixel map for field '" +
                               unique_name +
                               "' before the number of quadrature points has "
                               "been set for the field collection.");
        }
        return FieldMapType::Stride() / collection.get_nb_quad();
        break;
      }
      default:
        throw NFieldMapError("unknown iteration type");
        break;
      }
    }

    Dim_t nb_components;  //!< number of components stored per quadrature point
    TypedNField<Scalar> & field;  //!< reference to mapped field
    FieldMapType map;             //!< associated field map
  };

  /**
   * Alias of `muGrid::MappedField` for a map with corresponding
   * `muSpectre::NField` you wish to iterate over pixel by pixel or quadrature
   * point by quadrature point with statically sized `Eigen::Matrix` iterates
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam NbRow number of rows of the iterate
   * @tparam NbCol number of columns of the iterate
   * @tparam IterationType whether to iterate over pixels or quadrature points
   */
  template <typename T, Mapping Mutability, Dim_t NbRow, Dim_t NbCol,
            Iteration IterationType = Iteration::QuadPt>
  using MappedMatrixNField =
      MappedField<MatrixNFieldMap<T, Mutability, NbRow, NbCol, IterationType>>;

  /**
   * Alias of `muGrid::MappedField` for a map with corresponding
   * `muSpectre::NField` you wish to iterate over pixel by pixel or quadrature
   * point by quadrature point with statically sized `Eigen::Array` iterates
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam NbRow number of rows of the iterate
   * @tparam NbCol number of columns of the iterate
   * @tparam IterationType whether to iterate over pixels or quadrature points
   */
  template <typename T, Mapping Mutability, Dim_t NbRow, Dim_t NbCol,
            Iteration IterationType = Iteration::QuadPt>
  using MappedArrayNField =
      MappedField<ArrayNFieldMap<T, Mutability, NbRow, NbCol, IterationType>>;

  /**
   * Alias of `muGrid::MappedField` for a map of scalars with corresponding
   * `muSpectre::NField` you wish to iterate over quadrature point by quadrature
   * point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   */
  template <typename T, Mapping Mutability>
  using MappedScalarNField = MappedField<ScalarNFieldMap<T, Mutability>>;

  /**
   * Alias of `muGrid::MappedField` for a map of second-rank with corresponding
   * `muSpectre::NField` you wish to iterate over quadrature point by quadrature
   * point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam Dim spatial dimension of the tensors
   */
  template <typename T, Mapping Mutability, Dim_t Dim>
  using MappedT2NField = MappedField<T2NFieldMap<T, Mutability, Dim>>;

  /**
   * Alias of `muGrid::MappedField` for a map of fourth-rank with corresponding
   * `muSpectre::NField` you wish to iterate over quadrature point by quadrature
   * point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam Dim spatial dimension of the tensors
   */
  template <typename T, Mapping Mutability, Dim_t Dim>
  using MappedT4NField = MappedField<T4NFieldMap<T, Mutability, Dim>>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MAPPED_NFIELD_HH_
