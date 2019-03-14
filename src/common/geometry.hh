/**
 * @file   geometry.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   18 Apr 2018
 *
 * @brief  Geometric calculation helpers
 *
 * Copyright © 2018 Till Junge
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

#include "common/muSpectre_common.hh"
#include <libmugrid/tensor_algebra.hh>
#include <libmugrid/eigen_tools.hh>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <array>

#ifndef SRC_COMMON_GEOMETRY_HH_
#define SRC_COMMON_GEOMETRY_HH_

namespace muSpectre {

  /**
   * The rotation matrices depend on the order in which we rotate
   * around different axes. See [[
   * https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix ]] to
   * find the matrices
   */
  enum class RotationOrder {
    Z,
    XZXEuler,
    XYXEuler,
    YXYEuler,
    YZYEuler,
    ZYZEuler,
    ZXZEuler,
    XZYTaitBryan,
    XYZTaitBryan,
    YXZTaitBryan,
    YZXTaitBryan,
    ZYXTaitBryan,
    ZXYTaitBryan
  };

  namespace internal {

    template <Dim_t Dim>
    struct DefaultOrder {
      constexpr static RotationOrder value{RotationOrder::ZXYTaitBryan};
    };

    template <>
    struct DefaultOrder<twoD> {
      constexpr static RotationOrder value{RotationOrder::Z};
    };

  }  // namespace internal

  template <Dim_t Dim, RotationOrder Order = internal::DefaultOrder<Dim>::value>
  class Rotator {
   public:
    static_assert(((Dim == twoD) and (Order == RotationOrder::Z)) or
                      ((Dim == threeD) and (Order != RotationOrder::Z)),
                  "In 2d, only order 'Z' makes sense. In 3d, it doesn't");
    using Angles_t = Eigen::Matrix<Real, (Dim == twoD) ? 1 : 3, 1>;
    using RotMat_t = Eigen::Matrix<Real, Dim, Dim>;

    //! Default constructor
    Rotator() = delete;

    explicit Rotator(const Eigen::Ref<const Angles_t> & angles)
        : angles{angles}, rot_mat{this->compute_rotation_matrix()} {}

    //! Copy constructor
    Rotator(const Rotator & other) = default;

    //! Move constructor
    Rotator(Rotator && other) = default;

    //! Destructor
    virtual ~Rotator() = default;

    //! Copy assignment operator
    Rotator & operator=(const Rotator & other) = default;

    //! Move assignment operator
    Rotator & operator=(Rotator && other) = default;

    /**
     * Applies the rotation into the frame defined by the rotation
     * matrix
     *
     * @param input is a first-, second-, or fourth-rank tensor
     * (column vector, square matrix, or T4Matrix, or a Eigen::Map of
     * either of these, or an expression that evaluates into any of
     * these)
     */
    template <class In_t>
    inline decltype(auto) rotate(In_t && input);

    /**
     * Applies the rotation back out from the frame defined by the
     * rotation matrix
     *
     * @param input is a first-, second-, or fourth-rank tensor
     * (column vector, square matrix, or T4Matrix, or a Eigen::Map of
     * either of these, or an expression that evaluates into any of
     * these)
     */
    template <class In_t>
    inline decltype(auto) rotate_back(In_t && input);

    const RotMat_t & get_rot_mat() const { return rot_mat; }

   protected:
    inline RotMat_t compute_rotation_matrix();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Angles_t angles;
    RotMat_t rot_mat;

   private:
  };

  namespace internal {

    template <RotationOrder Order, Dim_t Dim>
    struct RotationMatrixComputer {};

    template <RotationOrder Order>
    struct RotationMatrixComputer<Order, twoD> {
      constexpr static Dim_t Dim{twoD};
      using RotMat_t = typename Rotator<Dim, Order>::RotMat_t;
      using Angles_t = typename Rotator<Dim, Order>::Angles_t;

      inline static decltype(auto)
      compute(const Eigen::Ref<Angles_t> & angles) {
        static_assert(Order == RotationOrder::Z,
                      "Two-d rotations can only be around the z axis");
        return RotMat_t(Eigen::Rotation2Dd(angles(0)));
      }
    };

    template <RotationOrder Order>
    struct RotationMatrixComputer<Order, threeD> {
      constexpr static Dim_t Dim{threeD};
      using RotMat_t = typename Rotator<Dim, Order>::RotMat_t;
      using Angles_t = typename Rotator<Dim, Order>::Angles_t;

      inline static decltype(auto)
      compute(const Eigen::Ref<Angles_t> & angles) {
        static_assert(Order != RotationOrder::Z,
                      "three-d rotations cannot only be around the z axis");

        switch (Order) {
        case RotationOrder::ZXZEuler: {
          return RotMat_t(
              (Eigen::AngleAxisd(angles(0), Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(angles(1), Eigen::Vector3d::UnitX()) *
               Eigen::AngleAxisd(angles(2), Eigen::Vector3d::UnitZ())));
          break;
        }
        case RotationOrder::ZXYTaitBryan: {
          return RotMat_t(
              (Eigen::AngleAxisd(angles(0), Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(angles(1), Eigen::Vector3d::UnitX()) *
               Eigen::AngleAxisd(angles(2), Eigen::Vector3d::UnitY())));
        }
        default: { throw std::runtime_error("not yet implemented."); }
        }
      }
    };

  }  // namespace internal

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim, RotationOrder Order>
  auto Rotator<Dim, Order>::compute_rotation_matrix() -> RotMat_t {
    return internal::RotationMatrixComputer<Order, Dim>::compute(this->angles);
  }

  namespace internal {

    template <Dim_t Rank>
    struct RotationHelper {};

    /* ---------------------------------------------------------------------- */
    template <>
    struct RotationHelper<firstOrder> {
      template <class In_t, class Rot_t>
      inline static decltype(auto) rotate(In_t && input, Rot_t && R) {
        return R * input;
      }
    };

    /* ---------------------------------------------------------------------- */
    template <>
    struct RotationHelper<secondOrder> {
      template <class In_t, class Rot_t>
      inline static decltype(auto) rotate(In_t && input, Rot_t && R) {
        return R * input * R.transpose();
      }
    };

    /* ---------------------------------------------------------------------- */
    template <>
    struct RotationHelper<fourthOrder> {
      template <class In_t, class Rot_t>
      inline static decltype(auto) rotate(In_t && input, Rot_t && R) {
        constexpr Dim_t Dim{muGrid::EigenCheck::tensor_dim<Rot_t>::value};
        auto && rotator_forward{
            Matrices::outer_under(R.transpose(), R.transpose())};
        auto && rotator_back = Matrices::outer_under(R, R);

        // Clarification. When I return this value as an
        // expression, clang segfaults or returns an uninitialised
        // tensor, hence the explicit cast into a T4Mat.
        return T4Mat<Real, Dim>(rotator_back * input * rotator_forward);
      }
    };
  }  // namespace internal

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim, RotationOrder Order>
  template <class In_t>
  auto Rotator<Dim, Order>::rotate(In_t && input) -> decltype(auto) {
    constexpr Dim_t tensor_rank{
        muGrid::EigenCheck::tensor_rank<In_t, Dim>::value};

    return internal::RotationHelper<tensor_rank>::rotate(
        std::forward<In_t>(input), this->rot_mat);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim, RotationOrder Order>
  template <class In_t>
  auto Rotator<Dim, Order>::rotate_back(In_t && input) -> decltype(auto) {
    constexpr Dim_t tensor_rank{
        muGrid::EigenCheck::tensor_rank<In_t, Dim>::value};

    return internal::RotationHelper<tensor_rank>::rotate(
        std::forward<In_t>(input), this->rot_mat.transpose());
  }

}  // namespace muSpectre

#endif  // SRC_COMMON_GEOMETRY_HH_
