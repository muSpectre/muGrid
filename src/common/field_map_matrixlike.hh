/**
 * file   field_map_matrixlike.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   26 Sep 2017
 *
 * @brief  Eigen-Matrix and -Array maps over strongly typed fields
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

#ifndef FIELD_MAP_MATRIXLIKE_H
#define FIELD_MAP_MATRIXLIKE_H

#include <memory>
#include <Eigen/Dense>

#include "common/field_map_base.hh"
#include "common/T4_map_proxy.hh"

namespace muSpectre {

  namespace internal {

    /* ---------------------------------------------------------------------- */
    enum class Map_t{Matrix, Array, T4Matrix};
    template<Map_t map_type>
    struct NameWrapper {
      const static std::string field_info_root;
    };
    template<>
    const std::string NameWrapper<Map_t::Array>::field_info_root{"Array"};
    template<>
    const std::string NameWrapper<Map_t::Matrix>::field_info_root{"Matrix"};
    template<>
    const std::string NameWrapper<Map_t::T4Matrix>::field_info_root{"T4Matrix"};

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, class EigenArray, Map_t map_type,
              bool ConstField>
    class MatrixLikeFieldMap: public FieldMap<FieldCollection,
                                              typename EigenArray::Scalar,
                                              EigenArray::SizeAtCompileTime,
                                              ConstField>
    {
    public:
      using Parent = FieldMap<FieldCollection,
                              typename EigenArray::Scalar,
                              EigenArray::SizeAtCompileTime, ConstField>;
      using Ccoord = Ccoord_t<FieldCollection::spatial_dim()>;
      using value_type = EigenArray;
      using const_reference = const value_type;
      using reference = std::conditional_t<ConstField,
                                           const_reference,
                                           value_type>; // since it's a resource handle
      using size_type = typename Parent::size_type;
      using pointer = std::unique_ptr<EigenArray>;
      using TypedField = typename Parent::TypedField;
      using Field = typename TypedField::Parent;
      using const_iterator= typename Parent::template iterator<MatrixLikeFieldMap, true>;
      using iterator = std::conditional_t<
        ConstField,
        const_iterator,
        typename Parent::template iterator<MatrixLikeFieldMap>>;
      using reverse_iterator = std::reverse_iterator<iterator>;
      using const_reverse_iterator = std::reverse_iterator<const_iterator>;

      friend iterator;

      //! Default constructor
      MatrixLikeFieldMap() = delete;

      /**
       * Constructor using a (non-typed) field. Compatibility is enforced at
       * runtime.  This should not be a performance concern, as this constructor
       * will not be called in anny inner loops (if used as intended).
       */
      template <bool isntConst=!ConstField>
      MatrixLikeFieldMap(std::enable_if_t<isntConst, Field &> field);
      template <bool isConst=ConstField>
      MatrixLikeFieldMap(std::enable_if_t<isConst, const Field &> field);

      /**
       * Constructor using a typed field. Compatibility is enforced
       * statically. It is not always possible to call this constructor, as the
       * type of the field might not be known at compile time.
       */
      template<class FC, typename T2, Dim_t NbC>
      MatrixLikeFieldMap(TypedFieldBase<FC, T2, NbC> & field);

      //! Copy constructor
      MatrixLikeFieldMap(const MatrixLikeFieldMap &other) = delete;

      //! Move constructor
      MatrixLikeFieldMap(MatrixLikeFieldMap &&other) noexcept = delete;

      //! Destructor
      virtual ~MatrixLikeFieldMap() noexcept = default;

      //! Copy assignment operator
      MatrixLikeFieldMap& operator=(const MatrixLikeFieldMap &other) = delete;

      //! Move assignment operator
      MatrixLikeFieldMap& operator=(MatrixLikeFieldMap &&other) noexcept = delete;

      //! give human-readable field map type
      inline std::string info_string() const override final;

      //! member access
      template <class ref = reference>
      inline ref operator[](size_type index);
      inline reference operator[](const Ccoord& ccoord);

      //! return an iterator to head of field for ranges
      inline iterator begin(){return iterator(*this);}
      inline const_iterator cbegin(){return const_iterator(*this);}
      //! return an iterator to tail of field for ranges
      inline iterator end(){return iterator(*this, false);};
      inline const_iterator cend(){return const_iterator(*this, false);}

    protected:
      //! for sad, legacy iterator use
      inline pointer ptr_to_val_t(size_type index);
      const static std::string field_info_root;
    private:
    };

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, class EigenArray, Map_t map_type,
              bool ConstField>
    const std::string MatrixLikeFieldMap<FieldCollection, EigenArray, map_type, ConstField>::
      field_info_root{NameWrapper<map_type>::field_info_root};

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, class EigenArray, Map_t map_type,
              bool ConstField>
    template <bool isntConst>
    MatrixLikeFieldMap<FieldCollection, EigenArray, map_type, ConstField>::
    MatrixLikeFieldMap(std::enable_if_t<isntConst, Field &> field)
      :Parent(field) {
      static_assert((isntConst != ConstField),
                    "let the compiler deduce isntConst, this is a SFINAE "
                    "parameter");
      this->check_compatibility();
    }

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, class EigenArray, Map_t map_type,
              bool ConstField>
    template <bool isConst>
    MatrixLikeFieldMap<FieldCollection, EigenArray, map_type, ConstField>::
    MatrixLikeFieldMap(std::enable_if_t<isConst, const Field &> field)
      :Parent(field) {
      static_assert((isConst == ConstField),
                    "let the compiler deduce isntConst, this is a SFINAE "
                    "parameter");
      this->check_compatibility();
    }

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, class EigenArray, Map_t map_type,
              bool ConstField>
    template<class FC, typename T2, Dim_t NbC>
    MatrixLikeFieldMap<FieldCollection, EigenArray, map_type, ConstField>::
    MatrixLikeFieldMap(TypedFieldBase<FC, T2, NbC> & field)
      :Parent(field) {
    }

    /* ---------------------------------------------------------------------- */
    //! human-readable field map type
    template<class FieldCollection, class EigenArray, Map_t map_type,
              bool ConstField>
    std::string
    MatrixLikeFieldMap<FieldCollection, EigenArray, map_type, ConstField>::
    info_string() const {
      std::stringstream info;
      info << field_info_root << "("
           << typeid(typename EigenArray::value_type).name() << ", "
           << EigenArray::RowsAtCompileTime << "x"
           << EigenArray::ColsAtCompileTime << ")";
      return info.str();
    }

    /* ---------------------------------------------------------------------- */
    //! member access
    template<class FieldCollection, class EigenArray, Map_t map_type,
              bool ConstField>
    template <class ref>
    ref
    MatrixLikeFieldMap<FieldCollection, EigenArray, map_type, ConstField>::
    operator[](size_type index) {
      return ref(this->get_ptr_to_entry(index));
    }

    template<class FieldCollection, class EigenArray, Map_t map_type,
              bool ConstField>
    typename MatrixLikeFieldMap<FieldCollection, EigenArray, map_type, ConstField>::reference
    MatrixLikeFieldMap<FieldCollection, EigenArray, map_type, ConstField>::
    operator[](const Ccoord & ccoord) {
      auto && index = this->collection.get_index(ccoord);
      return reference(this->get_ptr_to_entry(std::move(index)));
    }

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, class EigenArray, Map_t map_type,
              bool ConstField>
    typename MatrixLikeFieldMap<FieldCollection, EigenArray, map_type, ConstField>::pointer
    MatrixLikeFieldMap<FieldCollection, EigenArray, map_type, ConstField>::
    ptr_to_val_t(size_type index) {
      return std::make_unique<value_type>
        (this->get_ptr_to_entry(std::move(index)));
    }

  }  // internal

  /* ---------------------------------------------------------------------- */
  //! short-hand for an Eigen matrix map as iterate
  template <class FieldCollection, typename T, Dim_t NbRows, Dim_t NbCols,
            bool ConstField=false>
  using MatrixFieldMap = internal::MatrixLikeFieldMap
    <FieldCollection,
     std::conditional_t<ConstField,
                        Eigen::Map<const Eigen::Matrix<T, NbRows, NbCols>>,
                        Eigen::Map<Eigen::Matrix<T, NbRows, NbCols>>>,
     internal::Map_t::Matrix, ConstField>;

  /* ---------------------------------------------------------------------- */
  //! short-hand for an Eigen matrix map as iterate
  template <class FieldCollection, typename T, Dim_t Dim,
            bool MapConst=false, bool Symmetric=false>
  using T4MatrixFieldMap = internal::MatrixLikeFieldMap
    <FieldCollection,
     T4Map<T, Dim, MapConst, Symmetric>,
     internal::Map_t::T4Matrix, MapConst>;

  /* ---------------------------------------------------------------------- */
  //! short-hand for an Eigen array map as iterate
  template <class FieldCollection, typename T, Dim_t NbRows, Dim_t NbCols,
            bool ConstField=false>
  using ArrayFieldMap = internal::MatrixLikeFieldMap
    <FieldCollection,
     std::conditional_t<ConstField,
                        Eigen::Map<const Eigen::Array<T, NbRows, NbCols>>,
                        Eigen::Map<Eigen::Array<T, NbRows, NbCols>>>,
     internal::Map_t::Array, ConstField>;


}  // muSpectre

#endif /* FIELD_MAP_MATRIXLIKE_H */
