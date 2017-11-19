/**
 * file   T4_map_proxy.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   19 Nov 2017
 *
 * @brief  Map type to allow fourth-order tensor-like maps on 2D matrices
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

#include <Eigen/Dense>


#ifndef T4_MAP_PROXY_H
#define T4_MAP_PROXY_H

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  /** Proxy class mapping a fourth-order tensor onto a 2D matrix (in
      order to avoid the use of Eigen::Tensor
  **/
  template <typename T, Dim_t Dim, bool Symmetric=false,
            int MapOptions=Eigen::Unaligned>
  class T4Map:
    public Eigen::MapBase<Eigen::Matrix<T, Dim*Dim, Dim*Dim>, MapOptions> {
    {
    public:
      typedef MapBase<T4Map> Base;
      EIGEN_DENSE_PUBLIC_INTERFACE(T4Map);

      typedef typename Base::PointerType PointerType;
      typedef PointerType PointerArgType;
      EIGEN_DEVICE_FUNC
        inline PointerType cast_to_pointer_type(PointerArgType ptr) { return ptr; }

      EIGEN_DEVICE_FUNC
        inline Index innerStride() const
      {
        return StrideType::InnerStrideAtCompileTime != 0 ? m_stride.inner() : 1;
      }

      EIGEN_DEVICE_FUNC
        inline Index outerStride() const
      {
        return StrideType::OuterStrideAtCompileTime != 0 ? m_stride.outer()
          : IsVectorAtCompileTime ? this->size()
          : int(Flags)&RowMajorBit ? this->cols()
          : this->rows();
      }

      /** Constructor in the fixed-size case.
       *
       * \param dataPtr pointer to the array to map
       * \param stride optional Stride object, passing the strides.
       */
      EIGEN_DEVICE_FUNC
        explicit inline T4Map(PointerArgType dataPtr, const StrideType& stride = StrideType())
        : Base(cast_to_pointer_type(dataPtr)), m_stride(stride)
      {
        PlainObjectType::Base::_check_template_params();
      }

      EIGEN_INHERIT_ASSIGNMENT_OPERATORS(T4Map);

      /** My accessor to mimick tensorial access
       **/
      inline const Scalar& operator()(Dim_t i, Dim_t j, Dim_t k, Dim_t l ) const {
        constexpr auto myColStride{
          (colStride() == 1) ? colStride(): colStride()/Dim};
        constexpr auto myRowStride{
          (rowStride() == 1) ? rowStride(): rowStride()/Dim};
        return this->operator()(i * myRowStride + j * myColStride,
                                k * myRowStride + l * myColStride);
      }

      inline Scalar& operator()(Dim_t i, Dim_t j, Dim_t k, Dim_t l ) {
        constexpr auto myColStride{
          (colStride() == 1) ? colStride(): colStride()/Dim};
        constexpr auto myRowStride{
          (rowStride() == 1) ? rowStride(): rowStride()/Dim};
        return this->operator()(i * myRowStride + j * myColStride,
                                k * myRowStride + l * myColStride);
      }

    protected:
      StrideType m_stride;
    };


}  // muSpectre

#endif /* T4_MAP_PROXY_H */
