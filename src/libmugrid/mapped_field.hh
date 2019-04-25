/**
 * @file   mapped_field.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   27 Mar 2019
 *
 * @brief convenience functions to deal with data structures common to most
 *        internal variable fields in materials
 *
 * Copyright © 2019 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
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

#include "libmugrid/grid_common.hh"
#include "libmugrid/field.hh"
#include "libmugrid/statefield.hh"
#include "libmugrid/field_collection_local.hh"

#include <string>
#include <type_traits>

#ifndef SRC_LIBMUGRID_MAPPED_FIELD_HH_
#define SRC_LIBMUGRID_MAPPED_FIELD_HH_

namespace muGrid {

  namespace internal {

    /**
     * simple structure calling make_field for regular (non-state) fields
     */
    template <class Field, bool IsStateField>
    struct FieldMaker {
      using Collection_t = typename Field::collection_t;
      static Field & make(const std::string & unique_name,
                          Collection_t & collection) {
        return make_field<Field>(unique_name, collection);
      }
    };

    /**
     * specialisation for statefields
     */

    template <class StateField>
    struct FieldMaker<StateField, true> {
      using Collection_t = typename StateField::collection_t;
      static StateField & make(const std::string & unique_prefix,
                               Collection_t & collection) {
        return make_statefield<StateField>(unique_prefix, collection);
      }
    };

    /**
     * simple structure constructing regular (non-state) maps
     */
    template <class Map, class Field, bool IsStateField>
    struct MapMaker {
      static Map make(Field & field) { return Map{field}; }
    };

    /**
     * specialisation for statefield maps
     */

    template <class Map, class Field>
    struct MapMaker<Map, Field, true> {
      using Collection_t = typename Field::collection_t;
      static StateFieldMap<Map, Field::nb_memory()> make(Field & field) {
        constexpr static size_t NbMemory{Field::nb_memory()};
        return StateFieldMap<Map, NbMemory>{field};
      }
    };

    template <class FieldMap, class Field, bool IsStateField>
    struct FieldMapProvider {
      using type = StateFieldMap<FieldMap, Field::nb_memory()>;
    };

    template <class FieldMap, class Field>
    struct FieldMapProvider<FieldMap, Field, false> {
      using type = FieldMap;
    };

  }  // namespace internal

  /**
   * MappedFields are a combination of a field and an associated map, and as
   * such it does not introduce any new functionality that Fields and FieldMaps
   * do not already possess. They provide a convenience structure for the
   * default use case of internal variables, which are typically used only by a
   * single material and always the same way.
   */
  template <class Collection, class FieldMap, class Field>
  class MappedField {
   public:
    using Collection_t = Collection;
    constexpr static bool IsStateField{
        std::is_base_of<StateFieldBase<Collection>, Field>::value};
    using FieldMap_t = typename internal::FieldMapProvider<FieldMap, Field,
                                                           IsStateField>::type;
    using Field_t = Field;  //! Just needed for compile-time inspection
    using iterator = typename FieldMap_t::iterator;
    // const iterators do not make sense for statefields, so they are defined
    // only for regular field_maps
    using const_iterator = typename FieldMap::const_iterator;

    using reference =
        std::conditional_t<IsStateField, typename FieldMap_t::value_type,
                           typename FieldMap::reference>;
    using const_reference = typename FieldMap::const_reference;
    using size_type = typename FieldMap::size_type;

    //! Default constructor
    MappedField() = delete;

    //! Copy constructor
    MappedField(const MappedField & other) = delete;

    //! Constructor
    MappedField(Collection & collection, std::string unique_name)
        : field{internal::FieldMaker<Field, IsStateField>::make(unique_name,
                                                                collection)},
          map{internal::MapMaker<FieldMap, Field, IsStateField>::make(field)} {}

    //! Move constructor
    MappedField(MappedField && other) = default;

    //! Destructor
    virtual ~MappedField() = default;

    //! Copy assignment operator
    MappedField & operator=(const MappedField & other) = delete;

    //! Move assignment operator
    MappedField & operator=(MappedField && other) = default;

    inline reference operator[](size_type index) { return this->map[index]; }

    template <bool IsStandard = not IsStateField>
    inline std::enable_if_t<IsStandard, const_reference>
    operator[](size_type index) const {
      static_assert(IsStandard == not IsStateField,
                    "IsStandard is a SFINAE parameter, do not set manually");
      return this->map[index];
    }

    inline iterator begin() { return this->map.begin(); }
    template <bool HasConst = not IsStateField>
    inline std::enable_if_t<HasConst, const_iterator> cbegin() {
      static_assert(HasConst == not IsStateField,
                    "HasConst is a SFINAE parameter, do not set it manually");
      return this->map.cbegin();
    }
    template <bool HasConst = not IsStateField>
    inline std::enable_if_t<HasConst, const_iterator> cbegin() const {
      static_assert(HasConst == not IsStateField,
                    "HasConst is a SFINAE parameter, do not set it manually");
      return this->map.cbegin();
    }
    template <bool HasConst = not IsStateField>
    inline std::enable_if_t<HasConst, const_iterator> begin() const {
      static_assert(HasConst == not IsStateField,
                    "HasConst is a SFINAE parameter, do not set it manually");
      return this->map.begin();
    }

    inline iterator end() { return this->map.end(); }
    template <bool HasConst = not IsStateField>
    inline std::enable_if_t<HasConst, const_iterator> cend() {
      static_assert(HasConst == not IsStateField,
                    "HasConst is a SFINAE parameter, do not set it manually");
      return this->map.cend();
    }
    template <bool HasConst = not IsStateField>
    inline std::enable_if_t<HasConst, const_iterator> cend() const {
      static_assert(HasConst == not IsStateField,
                    "HasConst is a SFINAE parameter, do not set it manually");
      return this->map.cend();
    }
    template <bool HasConst = not IsStateField>
    inline std::enable_if_t<HasConst, const_iterator> end() const {
      static_assert(HasConst == not IsStateField,
                    "HasConst is a SFINAE parameter, do not set it manually");
      return this->map.end();
    }

    inline Field & get_field() { return this->field; }
    inline const Field & get_field() const { return this->field; }

    inline FieldMap_t & get_map() { return this->map; }
    inline const FieldMap_t & get_map() const { return this->map; }

    constexpr static Dim_t spatial_dimension() {
      return Field::spatial_dimension();
    }

   protected:
    Field & field;
    FieldMap_t map;
  };

  /**
   * Alias to simply create a matrix field and its associated matrix
   * iterator
   */
  template <typename T, Dim_t Dim, Dim_t NbRows, Dim_t NbCols,
            bool ConstField = false>
  using MappedMatrixField = MappedField<
      LocalFieldCollection<Dim>,
      MatrixFieldMap<LocalFieldCollection<Dim>, T, NbRows, NbCols, ConstField>,
      MatrixField<LocalFieldCollection<Dim>, T, NbRows, NbCols>>;

  /**
   * Alias to simply create a array field and its associated array
   * iterator
   */
  template <typename T, Dim_t Dim, Dim_t NbRows, Dim_t NbCols = 1,
            bool ConstField = false>
  using MappedArrayField = MappedField<
      LocalFieldCollection<Dim>,
      ArrayFieldMap<LocalFieldCollection<Dim>, T, NbRows, NbCols, ConstField>,
      MatrixField<LocalFieldCollection<Dim>, T, NbRows, NbCols>>;
  /**
   * Alias to simply create second-rank tensor field and its associated matrix
   * iterator
   */
  template <typename T, Dim_t DimS, Dim_t DimM = DimS, bool ConstField = false>
  using MappedT2Field = MappedField<
      LocalFieldCollection<DimS>,
      MatrixFieldMap<LocalFieldCollection<DimS>, T, DimM, DimM, ConstField>,
      TensorField<LocalFieldCollection<DimS>, T, secondOrder, DimM>>;

  /**
   * Alias to simply create fourth-rank tensor field and its associated T4Mat
   * iterator
   */
  template <typename T, Dim_t DimS, Dim_t DimM = DimS, bool ConstField = false>
  using MappedT4Field = MappedField<
      LocalFieldCollection<DimS>,
      MatrixFieldMap<LocalFieldCollection<DimS>, T, DimM * DimM, DimM * DimM,
                     ConstField>,
      TensorField<LocalFieldCollection<DimS>, T, fourthOrder, DimM>>;

  /**
   * Alias to simply create a scalar field and its associaced iterator
   * iterator
   */
  template <typename T, Dim_t Dim, bool ConstField = false>
  using MappedScalarField =
      MappedField<LocalFieldCollection<Dim>,
                  ScalarFieldMap<LocalFieldCollection<Dim>, T, ConstField>,
                  ScalarField<LocalFieldCollection<Dim>, T>>;

  /* ---------------------------------------------------------------------- */

  /**
   * Alias to simply create a matrix state field and its associated matrix
   * iterator
   */
  template <typename T, Dim_t Dim, Dim_t NbRows, Dim_t NbCols,
            size_t NbMemory = 1, bool ConstField = false>
  using MappedMatrixStateField = MappedField<
      LocalFieldCollection<Dim>,
      MatrixFieldMap<LocalFieldCollection<Dim>, T, NbRows, NbCols, ConstField>,
      StateField<MatrixField<LocalFieldCollection<Dim>, T, NbRows, NbCols>,
                 NbMemory>>;

  /**
   * Alias to simply create a array state field and its associated array
   * iterator
   */
  template <typename T, Dim_t Dim, Dim_t NbRows, Dim_t NbCols = 1,
            size_t NbMemory = 1, bool ConstField = false>
  using MappedArrayStateField = MappedField<
      LocalFieldCollection<Dim>,
      ArrayFieldMap<LocalFieldCollection<Dim>, T, NbRows, NbCols, ConstField>,
      StateField<MatrixField<LocalFieldCollection<Dim>, T, NbRows, NbCols>,
                 NbMemory>>;
  /**
   * Alias to simply create second-rank tensor state field and its associated
   * matrix iterator
   */
  template <typename T, Dim_t DimS, Dim_t DimM = DimS, size_t NbMemory = 1,
            bool ConstField = false>
  using MappedT2StateField = MappedField<
      LocalFieldCollection<DimS>,
      MatrixFieldMap<LocalFieldCollection<DimS>, T, DimM, DimM, ConstField>,
      StateField<TensorField<LocalFieldCollection<DimS>, T, secondOrder, DimM>,
                 NbMemory>>;

  /**
   * Alias to simply create fourth-rank tensor state field and its associated
   * T4Mat iterator
   */
  template <typename T, Dim_t DimS, Dim_t DimM = DimS, size_t NbMemory = 1,
            bool ConstField = false>
  using MappedT4StateField = MappedField<
      LocalFieldCollection<DimS>,
      MatrixFieldMap<LocalFieldCollection<DimS>, T, DimM * DimM, DimM * DimM,
                     ConstField>,
      StateField<TensorField<LocalFieldCollection<DimS>, T, fourthOrder, DimM>,
                 NbMemory>>;

  /**
   * Alias to simply create a scalar state field and its associaced iterator
   * iterator
   */
  template <typename T, Dim_t Dim, size_t NbMemory = 1, bool ConstField = false>
  using MappedScalarStateField = MappedField<
      LocalFieldCollection<Dim>,
      ScalarFieldMap<LocalFieldCollection<Dim>, T, ConstField>,
      StateField<ScalarField<LocalFieldCollection<Dim>, T>, NbMemory>>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MAPPED_FIELD_HH_
