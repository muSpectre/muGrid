/**
 * @file   mapped_state_nfield.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   09 Sep 2019
 *
 * @brief  Convenience class extending the mapped field concept to state fields
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

#ifndef SRC_LIBMUGRID_MAPPED_STATE_NFIELD_HH_
#define SRC_LIBMUGRID_MAPPED_STATE_NFIELD_HH_

#include "state_nfield_map_static.hh"
#include "state_nfield.hh"
#include "nfield_collection.hh"

namespace muGrid {

  /**
   * MappedStateNFields are a combination of a state field and an associated
   * map, and as such it does not introduce any new functionality that
   * StateFields and StateFieldMaps do not already possess. They provide a
   * convenience structure for the default use case of internal variables, which
   * are typically used only by a single material and always the same way.
   */
  template <class StateFieldMapType>
  class MappedStateField {
   public:
    //! stored scalar type
    using Scalar = typename StateFieldMapType::Scalar;
    //! return type for iterators over this- map
    using Return_t = typename StateFieldMapType::template StaticStateWrapper<
        StateFieldMapType::FieldMutability()>;
    //! iterator over this map
    using iterator = typename StateFieldMapType::iterator;
    //! constant iterator over this map
    using const_iterator = typename StateFieldMapType::const_iterator;

    //! Deleted default constructor
    MappedStateField() = delete;

    //! Constructor with name and collection
    MappedStateField(const std::string & unique_name,
                     NFieldCollection & collection)
        : nb_components{compute_nb_components(unique_name, collection)},
          state_field(collection.register_state_field<Scalar>(
              unique_name, StateFieldMapType::GetNbMemory(),
              this->nb_components)),
          map{this->state_field} {}

    //! Copy constructor
    MappedStateField(const MappedStateField & other) = delete;

    //! Move constructor
    MappedStateField(MappedStateField && other) = default;

    //! Destructor
    virtual ~MappedStateField() = default;

    //! Copy assignment operator
    MappedStateField & operator=(const MappedStateField & other) = delete;

    //! Move assignment operator
    MappedStateField & operator=(MappedStateField && other) = default;

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

    //! return a reference to the mapped state field
    TypedStateNField<Scalar> & get_state_field() { return this->state_field; }

    //! return a reference to the map
    StateFieldMapType & get_map() { return this->map; }

   protected:
    /**
     * evaluate and return the number of components the statically mapped state
     * field needs to store per quadrature point
     */
    static Dim_t compute_nb_components(const std::string & unique_prefix,
                                       NFieldCollection & collection) {
      switch (StateFieldMapType::GetIterationType()) {
      case Iteration::QuadPt: {
        return StateFieldMapType::StaticNFieldMap_t::Stride();
        break;
      }
      case Iteration::Pixel: {
        if (not collection.has_nb_quad()) {
          throw NFieldMapError("Can't create a pixel map for state field '" +
                               unique_prefix +
                               "' before the number of quadrature points has "
                               "been set for the field collection.");
        }
        return StateFieldMapType::StaticNFieldMap_t::Stride() /
               collection.get_nb_quad();
        break;
      }
      default:
        throw NFieldMapError("unknown iteration type");
        break;
      }
    }

    Dim_t nb_components;  //!< number of components stored per quadrature point
    TypedStateNField<Scalar> & state_field;  //!< ref to mapped state field
    StateFieldMapType map;                   //!< associated field map
  };

  /**
   * Alias of `muGrid::MappedStateField` for a map with corresponding
   * `muSpectre::StateNField` you wish to iterate over pixel by pixel or
   * quadrature point by quadrature point with statically sized `Eigen::Matrix`
   * iterates
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam NbRow number of rows of the iterate
   * @tparam NbCol number of columns of the iterate
   * @tparam NbMemory number of previous values to store
   * @tparam IterationType whether to iterate over pixels or quadrature points
   */
  template <typename T, Mapping Mutability, Dim_t NbRow, Dim_t NbCol,
            size_t NbMemory = 1, Iteration IterationType = Iteration::QuadPt>
  using MappedMatrixStateNField =
      MappedStateField<MatrixStateNFieldMap<T, Mutability, NbRow, NbCol,
                                            NbMemory, IterationType>>;

  /**
   * Alias of `muGrid::MappedStateField` for a map with corresponding
   * `muSpectre::StateNField` you wish to iterate over pixel by pixel or
   * quadrature point by quadrature point with statically sized `Eigen::Array`
   * iterates
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam NbRow number of rows of the iterate
   * @tparam NbCol number of columns of the iterate
   * @tparam NbMemory number of previous values to store
   * @tparam IterationType whether to iterate over pixels or quadrature points
   */
  template <typename T, Mapping Mutability, Dim_t NbRow, Dim_t NbCol,
            size_t NbMemory = 1, Iteration IterationType = Iteration::QuadPt>
  using MappedArrayStateNField =
      MappedStateField<ArrayStateNFieldMap<T, Mutability, NbRow, NbCol,
                                           NbMemory, IterationType>>;

  /**
   * Alias of `muGrid::MappedStateField` for a map of scalars with corresponding
   * `muSpectre::StateNField` you wish to iterate over quadrature point by
   * quadrature point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam NbMemory number of previous values to store
   */
  template <typename T, Mapping Mutability, size_t NbMemory = 1>
  using MappedScalarStateNField =
      MappedStateField<ScalarStateNFieldMap<T, Mutability, NbMemory>>;

  /**
   * Alias of `muGrid::MappedStateField` for a map of second-rank with
   * corresponding `muSpectre::StateNField` you wish to iterate over quadrature
   * point by quadrature point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam Dim spatial dimension of the tensors
   * @tparam NbMemory number of previous values to store
   */
  template <typename T, Mapping Mutability, Dim_t Dim, size_t NbMemory = 1>
  using MappedT2StateNField =
      MappedStateField<T2StateNFieldMap<T, Mutability, Dim, NbMemory>>;

  /**
   * Alias of `muGrid::MappedStateField` for a map of fourth-rank with
   * corresponding `muSpectre::StateNField` you wish to iterate over quadrature
   * point by quadrature point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam Dim spatial dimension of the tensors
   * @tparam NbMemory number of previous values to store
   */
  template <typename T, Mapping Mutability, Dim_t Dim, size_t NbMemory = 1>
  using MappedT4StateNField =
      MappedStateField<T4StateNFieldMap<T, Mutability, Dim, NbMemory>>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MAPPED_STATE_NFIELD_HH_
