/**
 * @file   T4_map_proxy.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   19 Nov 2017
 *
 * @brief  Map type to allow fourth-order tensor-like maps on 2D matrices
 *
 * Copyright © 2017 Till Junge
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
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef T4_MAP_PROXY_H
#define T4_MAP_PROXY_H

#include "common/eigen_tools.hh"

#include <Eigen/Dense>
#include <Eigen/src/Core/util/Constants.h>

#include <type_traits>

namespace muSpectre {

  /**
   * simple adapter function to create a matrix that can be mapped as a tensor
   */
  template <typename T, Dim_t Dim>
  using T4Mat = Eigen::Matrix<T, Dim*Dim, Dim*Dim>;

  /**
   * Map onto `muSpectre::T4Mat`
   */
  template <typename T, Dim_t Dim, bool ConstMap=false>
  using T4MatMap = std::conditional_t<ConstMap,
                                      Eigen::Map<const T4Mat<T, Dim>>,
                                      Eigen::Map<T4Mat<T, Dim>>>;


  template<class Derived>
  struct DimCounter{};

  template <class Derived>
  struct DimCounter<Eigen::MatrixBase<Derived>> {
  private:
    using Type = Eigen::MatrixBase<Derived>;
    constexpr static Dim_t Rows{Type::RowsAtCompileTime};
  public:
    static_assert(Rows != Eigen::Dynamic,
                  "matrix type not statically sized");
    static_assert(Rows == Type::ColsAtCompileTime,
                  "matrix type not square");
    constexpr static Dim_t value{ct_sqrt(Rows)};
    static_assert(value*value == Rows,
                  "Only integer numbers of dimensions allowed");
  };
  /**
   * provides index-based access to fourth-order Tensors represented
   * by square matrices
   */
  template <typename T4>
  inline auto get(const Eigen::MatrixBase<T4>& t4, Dim_t i, Dim_t j, Dim_t k, Dim_t l)
    -> decltype(auto) {
    constexpr Dim_t Dim{DimCounter<Eigen::MatrixBase<T4>>::value};
    const auto myColStride{
      (t4.colStride() == 1) ? t4.colStride(): t4.colStride()/Dim};
    const auto myRowStride{
      (t4.rowStride() == 1) ? t4.rowStride(): t4.rowStride()/Dim};
    return t4(i * myRowStride + j * myColStride,
              k * myRowStride + l * myColStride);
  }

  template <typename T4>
  inline auto get(Eigen::MatrixBase<T4>& t4, Dim_t i, Dim_t j, Dim_t k, Dim_t l)
    -> decltype(t4.coeffRef(i,j)) {
    constexpr Dim_t Dim{DimCounter<Eigen::MatrixBase<T4>>::value};
    const auto myColStride{
      (t4.colStride() == 1) ? t4.colStride(): t4.colStride()/Dim};
    const auto myRowStride{
      (t4.rowStride() == 1) ? t4.rowStride(): t4.rowStride()/Dim};
    return t4.coeffRef(i * myRowStride + j * myColStride,
                       k * myRowStride + l * myColStride);
  }

  // /* ---------------------------------------------------------------------- */
  // /** Proxy class mapping a fourth-order tensor onto a 2D matrix (in
  //     order to avoid the use of Eigen::Tensor. This class is, however
  //     byte-compatible with Tensors (i.e., you can map this onto a
  //     tensor instead of a matrix)
  // **/
  // template <typename T, Dim_t Dim, bool MapConst=false, bool Symmetric=false,
  //           int MapOptions=Eigen::Unaligned,
  //           typename StrideType=Eigen::Stride<0, 0>>
  // class T4Map:
  //   public Eigen::MapBase<T4Map<T, Dim, MapConst, Symmetric,
  //                               MapOptions, StrideType>>
  // {
  //   public:
  //   typedef Eigen::MapBase<T4Map> Base;
  //   EIGEN_DENSE_PUBLIC_INTERFACE(T4Map);

  //   using matrix_type = Eigen::Matrix<T, Dim*Dim, Dim*Dim>;
  //   using PlainObjectType =
  //     std::conditional_t<MapConst,
  //                        const matrix_type, matrix_type>;
  //   using ConstType = T4Map<T, Dim, true, Symmetric, MapOptions, StrideType>;
  //   using Base::colStride;
  //   using Base::rowStride;
  //   using Base::IsRowMajor;
  //   typedef typename Base::PointerType PointerType;
  //   typedef PointerType PointerArgType;
  //   using trueScalar = std::conditional_t<MapConst, const Scalar, Scalar>;
  //   EIGEN_DEVICE_FUNC
  //   inline PointerType cast_to_pointer_type(PointerArgType ptr) { return ptr; }

  //   EIGEN_DEVICE_FUNC
  //   inline Eigen::Index innerStride() const
  //   {
  //     return StrideType::InnerStrideAtCompileTime != 0 ? m_stride.inner() : 1;
  //   }

  //   template <class Derived>
  //   inline T4Map & operator=(const Eigen::MatrixBase<Derived> & other) {
  //     this->map = other;
  //     return *this;
  //   }

  //   EIGEN_DEVICE_FUNC
  //   inline Eigen::Index outerStride() const
  //   {
  //     return StrideType::OuterStrideAtCompileTime != 0 ? m_stride.outer()
  //       : IsVectorAtCompileTime ? this->size()
  //       : int(Flags)&Eigen::RowMajorBit ? this->cols()
  //       : this->rows();
  //   }

  //   /** Constructor in the fixed-size case.
  //    *
  //    * \param dataPtr pointer to the array to map
  //    * \param stride optional Stride object, passing the strides.
  //    */
  //   EIGEN_DEVICE_FUNC
  //   explicit inline T4Map(PointerArgType dataPtr, const StrideType& stride = StrideType())
  //     : Base(cast_to_pointer_type(dataPtr)), m_stride(stride),
  //       map(cast_to_pointer_type(dataPtr))
  //   {
  //     PlainObjectType::Base::_check_template_params();
  //   }

  //   EIGEN_INHERIT_ASSIGNMENT_OPERATORS(T4Map);

  //   /** My accessor to mimick tensorial access
  //    **/
  //   inline const Scalar& operator()(Dim_t i, Dim_t j, Dim_t k, Dim_t l ) const {
  //     const auto myColStride{
  //       (colStride() == 1) ? colStride(): colStride()/Dim};
  //     const auto myRowStride{
  //       (rowStride() == 1) ? rowStride(): rowStride()/Dim};
  //     return this->map.coeff(i * myRowStride + j * myColStride,
  //                            k * myRowStride + l * myColStride);
  //   }


  //   inline trueScalar& operator()(Dim_t i, Dim_t j, Dim_t k, Dim_t l ) {
  //     const auto myColStride{
  //       (colStride() == 1) ? colStride(): colStride()/Dim};
  //     const auto myRowStride{
  //       (rowStride() == 1) ? rowStride(): rowStride()/Dim};
  //     return this->map.coeffRef(i * myRowStride + j * myColStride,
  //                               k * myRowStride + l * myColStride);
  //   }

  // protected:
  //   StrideType m_stride;
  //   Eigen::Map<PlainObjectType> map;
  // };


}  // muSpectre

// namespace Eigen {
//   //! forward declarations
//   template<typename T> struct traits;

//   /* ---------------------------------------------------------------------- */
//   namespace internal {
//     template<typename T, muSpectre::Dim_t Dim, bool MapConst, bool Symmetric,
//              int MapOptions, typename StrideType>
//     struct traits<muSpectre::T4Map<T, Dim, MapConst, Symmetric,
//                                    MapOptions, StrideType> >
//       : public traits<Matrix<T, Dim*Dim, Dim*Dim>>
//     {
//       using PlainObjectType = Matrix<T, Dim*Dim, Dim*Dim>;
//       typedef traits<PlainObjectType> TraitsBase;
//       enum {
//         InnerStrideAtCompileTime = StrideType::InnerStrideAtCompileTime == 0
//         ? int(PlainObjectType::InnerStrideAtCompileTime)
//         : int(StrideType::InnerStrideAtCompileTime),
//         OuterStrideAtCompileTime = StrideType::OuterStrideAtCompileTime == 0
//         ? int(PlainObjectType::OuterStrideAtCompileTime)
//         : int(StrideType::OuterStrideAtCompileTime),
//         Alignment = int(MapOptions)&int(AlignedMask),
//         Flags0 = TraitsBase::Flags & (~NestByRefBit),
//         Flags = is_lvalue<PlainObjectType>::value ? int(Flags0) : (int(Flags0) & ~LvalueBit)
//       };
//     private:
//       enum { Options }; // Expressions don't have Options
//     };
//   } // namespace internal
// } // namespace Eigen

#endif /* T4_MAP_PROXY_H */
