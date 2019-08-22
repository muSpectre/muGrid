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

  template <Dim_t Dim>
  class RotatorBase {
   public:
    using RotMat_t = Eigen::Matrix<Real, Dim, Dim>;
    //! Default constructor
    RotatorBase() = delete;

    explicit RotatorBase(const RotMat_t rotation_matrix_input)
        : rot_mat{rotation_matrix_input} {}
    //! Copy constructor
    RotatorBase(const RotatorBase & other) = default;

    //! Move constructor
    RotatorBase(RotatorBase && other) = default;

    //! Destructor
    virtual ~RotatorBase() = default;

    //! Copy assignment operator
    RotatorBase & operator=(const RotatorBase & other) = default;

    //! Move assignment operator

    RotatorBase & operator=(RotatorBase && other) = default;

    /**
     * Applies the rotation into the frame define my the rotation

     * matrix
     *
     * @param input is a first-, second-, or fourth-rank tensor
     * (column vector, square matrix, or T4Matrix, or a Eigen::Map of
     * either of these, or an expression that evaluates into any of
     * these)
     */
    template <class Derived>
    inline decltype(auto)
    rotate(const Eigen::MatrixBase<Derived> & input) const;

    /**

     * Applies the rotation back out from the frame define my the
     * rotation matrix
     *
     * @param input is a first-, second-, or fourth-rank tensor
     * (column vector, square matrix, or T4Matrix, or a Eigen::Map of
     * either of these, or an expression that evaluates into any of
     * these)
     */
    template <class Derived>
    inline decltype(auto)
    rotate_back(const Eigen::MatrixBase<Derived> & input) const;

    const RotMat_t & get_rot_mat() const { return this->rot_mat; }

    void set_rot_mat(const Eigen::Ref<RotMat_t> & mat_inp) {
      this->rot_mat = mat_inp;
    }

   protected:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    RotMat_t rot_mat;
  };

  template <Dim_t Dim, RotationOrder Order = internal::DefaultOrder<Dim>::value>
  class RotatorAngle : public RotatorBase<Dim> {
    static_assert(((Dim == twoD) and (Order == RotationOrder::Z)) or
                      ((Dim == threeD) and (Order != RotationOrder::Z)),
                  "In 2d, only order 'Z' makes sense. In 3d, it doesn't");

   public:
    using Parent = RotatorBase<Dim>;
    using Angles_t = Eigen::Matrix<Real, (Dim == twoD) ? 1 : 3, 1>;
    using RotMat_t = Eigen::Matrix<Real, Dim, Dim>;

    //! Default constructor
    RotatorAngle() = delete;

    explicit RotatorAngle(const Eigen::Ref<const Angles_t> & angles_inp)
        : Parent(this->compute_rotation_matrix_angle(angles_inp)),
          angles{angles_inp} {}

    //! Copy constructor
    RotatorAngle(const RotatorAngle & other) = default;

    //! Move constructor
    RotatorAngle(RotatorAngle && other) = default;

    //! Destructor
    virtual ~RotatorAngle() = default;

    //! Copy assignment operator
    RotatorAngle & operator=(const RotatorAngle & other) = default;

    //! Move assignment operator
    RotatorAngle & operator=(RotatorAngle && other) = default;

   protected:
    inline RotMat_t compute_rotation_matrix_angle(Angles_t angles);
    inline RotMat_t compute_this_rotation_matrix_angle() {
      return compute_rotation_matrix(this->angles);
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Angles_t angles;

   private:
  };

  /* ---------------------------------------------------------------------- */
  namespace internal {

    template <RotationOrder Order, Dim_t Dim>
    struct RotationMatrixComputerAngle {};

    template <RotationOrder Order>
    struct RotationMatrixComputerAngle<Order, twoD> {
      constexpr static Dim_t Dim{twoD};
      using RotMat_t = typename RotatorAngle<Dim, Order>::RotMat_t;
      using Angles_t = typename RotatorAngle<Dim, Order>::Angles_t;

      inline static decltype(auto)
      compute(const Eigen::Ref<Angles_t> & angles) {
        static_assert(Order == RotationOrder::Z,
                      "Two-d rotations can only be around the z axis");
        return RotMat_t(Eigen::Rotation2Dd(angles(0)));
      }
    };

    template <RotationOrder Order>
    struct RotationMatrixComputerAngle<Order, threeD> {
      constexpr static Dim_t Dim{threeD};
      using RotMat_t = typename RotatorAngle<Dim, Order>::RotMat_t;
      using Angles_t = typename RotatorAngle<Dim, Order>::Angles_t;

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
  auto RotatorAngle<Dim, Order>::compute_rotation_matrix_angle(Angles_t angles)
      -> RotMat_t {
    return internal::RotationMatrixComputerAngle<Order, Dim>::compute(angles);
  }

  /* ---------------------------------------------------------------------- */
  /**
   * this class is used to make the vector a aligned to the vec b by means of a
   rotation system, the input for the constructor is the vector itself and the
   functions
   rotate and rotate back would be available as they exist in the parent class
   (RotatorBase) nad can be used in order to do the functionality of the class
  */
  template <Dim_t Dim>
  class RotatorTwoVec : public RotatorBase<Dim> {
   public:
    using Parent = RotatorBase<Dim>;
    using Vec_t = Eigen::Matrix<Real, (Dim == twoD) ? 2 : 3, 1>;
    using RotMat_t = Eigen::Matrix<Real, Dim, Dim>;

    //! Default constructor
    RotatorTwoVec() = delete;

    RotatorTwoVec(Vec_t vec_a_inp, Vec_t vec_b_inp)
        : Parent(this->compute_rotation_matrix_TwoVec(vec_a_inp, vec_b_inp)),
          vec_ref{vec_a_inp}, vec_des{vec_b_inp} {}

    //! Copy constructor
    RotatorTwoVec(const RotatorTwoVec & other) = default;

    //! Move constructor
    RotatorTwoVec(RotatorTwoVec && other) = default;

    //! Destructor
    virtual ~RotatorTwoVec() = default;

    //! Copy assignment operator
    RotatorTwoVec & operator=(const RotatorTwoVec & other) = default;

    //! Move assignment operator
    RotatorTwoVec & operator=(RotatorTwoVec && other) = default;

   protected:
    inline RotMat_t compute_rotation_matrix_TwoVec(Vec_t vec_ref,
                                                   Vec_t vec_des);
    inline RotMat_t compute_this_rotation_matrix_TwoVec() {
      return compute_rotation_matrix(this->vec_ref, this->vec_des);
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Vec_t vec_ref, vec_des;

   private:
  };
  /* ---------------------------------------------------------------------- */
  namespace internal {
    template <Dim_t Dim>
    struct RotationMatrixComputerTwoVec {};
    template <>
    struct RotationMatrixComputerTwoVec<twoD> {
      constexpr static Dim_t Dim{twoD};
      using RotMat_t = typename RotatorTwoVec<Dim>::RotMat_t;
      using Vec_t = typename RotatorTwoVec<Dim>::Vec_t;

      inline static RotMat_t compute(Vec_t vec_ref, Vec_t vec_des) {
        Real v_ref_norm =
            sqrt(vec_ref(0) * vec_ref(0) + vec_ref(1) * vec_ref(1));
        Real v_des_norm =
            sqrt(vec_des(0) * vec_des(0) + vec_des(1) * vec_des(1));
        RotMat_t ret_mat;
        ret_mat(0, 0) = ret_mat(1, 1) =
            (((vec_ref(0) / v_ref_norm) * (vec_des(0) / v_des_norm)) +
             ((vec_des(1) / v_des_norm) * (vec_ref(1) / v_ref_norm)));

        ret_mat(1, 0) =
            (((vec_ref(0) / v_ref_norm) * (vec_des(1) / v_des_norm)) -
             ((vec_des(0) / v_des_norm) * (vec_ref(1) / v_ref_norm)));
        ret_mat(0, 1) = -ret_mat(1, 0);
        return ret_mat;
      }
    };

    template <>
    struct RotationMatrixComputerTwoVec<threeD> {
      constexpr static Dim_t Dim{threeD};
      using RotMat_t = typename RotatorTwoVec<Dim>::RotMat_t;
      using Vec_t = typename RotatorTwoVec<Dim>::Vec_t;
      inline static RotMat_t compute(Vec_t vec_ref, Vec_t vec_des) {
        return Eigen::Quaternion<double>::FromTwoVectors(vec_ref, vec_des)
            .normalized()
            .toRotationMatrix();
      }
    };

  }  // namespace internal

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  auto RotatorTwoVec<Dim>::compute_rotation_matrix_TwoVec(Vec_t vec_ref,
                                                          Vec_t vec_des)
      -> RotMat_t {
    return internal::RotationMatrixComputerTwoVec<Dim>::compute(vec_ref,
                                                                vec_des);
  }

  /* ---------------------------------------------------------------------- */
  /**
   * this class is used to make a vector aligned to x-axis of the coordinate
   system, the input for the constructor is the vector itself and the functions
   rotate and rotate back would be available as they exist in the parent class
   (RotatorBase) nad can be used in order to do the functionality of the class
   */
  template <Dim_t Dim>
  class RotatorNormal : public RotatorBase<Dim> {
   public:
    using Parent = RotatorBase<Dim>;
    using Vec_t = Eigen::Matrix<Real, Dim, 1>;
    using RotMat_t = Eigen::Matrix<Real, Dim, Dim>;

    //! Default constructor
    RotatorNormal() = delete;

    explicit RotatorNormal(Vec_t vec)
        : Parent(this->compute_rotation_matrix_normal(vec)), vec{vec} {}

    //! Copy constructor
    RotatorNormal(const RotatorNormal & other) = default;

    //! Move constructor
    RotatorNormal(RotatorNormal && other) = default;

    //! Destructor
    virtual ~RotatorNormal() = default;

    //! Copy assignment operator
    RotatorNormal & operator=(const RotatorNormal & other) = default;

    //! Move assignment operator
    RotatorNormal & operator=(RotatorNormal && other) = default;

   protected:
    inline RotMat_t compute_rotation_matrix_normal(Vec_t vec);
    inline RotMat_t compute_this_rotation_matrix_normal() {
      return compute_rotation_matrix_normal(this->vec);
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Vec_t vec;

   private:
  };

  /* ---------------------------------------------------------------------- */
  namespace internal {
    template <Dim_t Dim>
    struct RotationMatrixComputerNormal {};
    template <>
    struct RotationMatrixComputerNormal<twoD> {
      constexpr static Dim_t Dim{twoD};
      using RotMat_t = typename RotatorTwoVec<Dim>::RotMat_t;
      using Vec_t = typename RotatorTwoVec<Dim>::Vec_t;

      inline static RotMat_t compute(Vec_t vec) {
        Real vec_norm = sqrt(vec(0) * vec(0) + vec(1) * vec(1));
        Vec_t x;
        x << 1.0, 0.0;

        RotMat_t ret_mat;
        ret_mat(0, 0) = ret_mat(1, 1) = ((vec(0) / vec_norm) * x(0));
        ret_mat(1, 0) = -(-(vec(1) / vec_norm) * x(0));
        ret_mat(0, 1) = -ret_mat(1, 0);
        return ret_mat;
      }
    };

    template <>
    struct RotationMatrixComputerNormal<threeD> {
      constexpr static Dim_t Dim{threeD};
      using RotMat_t = typename RotatorTwoVec<Dim>::RotMat_t;
      using Vec_t = typename RotatorTwoVec<Dim>::Vec_t;
      inline static RotMat_t compute(Vec_t vec) {
        Real eps = 0.1;
        Vec_t vec1 = vec / vec.norm();
        Vec_t x(Vec_t::UnitX());
        Vec_t y(Vec_t::UnitY());
        Vec_t n_x = vec1.cross(x);
        Vec_t vec2 = ((n_x.norm() > eps) * n_x +
                      (1 - (n_x.norm() > eps)) * (vec1.cross(y)));
        Vec_t vec3 = vec1.cross(vec2);
        RotMat_t ret_mat;
        ret_mat << vec1(0), vec2(0) / vec2.norm(), vec3(0) / vec3.norm(),
            vec1(1), vec2(1) / vec2.norm(), vec3(1) / vec3.norm(), vec1(2),
            vec2(2) / vec2.norm(), vec3(2) / vec3.norm();
        return ret_mat;
      }
    };
  }  // namespace internal

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  auto RotatorNormal<Dim>::compute_rotation_matrix_normal(Vec_t vec)
      -> RotMat_t {
    return internal::RotationMatrixComputerNormal<Dim>::compute(vec);
  }

  /* ---------------------------------------------------------------------- */

  namespace internal {

    template <Dim_t Rank>
    struct RotationHelper {};

    /* ---------------------------------------------------------------------- */
    template <>
    struct RotationHelper<firstOrder> {
      template <class Derived1, class Derived2>
      inline static decltype(auto)
      rotate(const Eigen::MatrixBase<Derived1> & input,
             const Eigen::MatrixBase<Derived2> & R) {
        return R * input;
      }
    };

    /* ---------------------------------------------------------------------- */
    template <>
    struct RotationHelper<secondOrder> {
      template <class Derived1, class Derived2>
      inline static decltype(auto)
      rotate(const Eigen::MatrixBase<Derived1> & input,
             const Eigen::MatrixBase<Derived2> & R) {
        return R * input * R.transpose();
      }
    };

    /* ---------------------------------------------------------------------- */
    template <>
    struct RotationHelper<fourthOrder> {
      template <class Derived1, class Derived2>
      inline static decltype(auto)
      rotate(const Eigen::MatrixBase<Derived1> & input,
             const Eigen::MatrixBase<Derived2> & R) {
        constexpr Dim_t Dim{muGrid::EigenCheck::tensor_dim<Derived2>::value};
        auto && rotator_forward{
            Matrices::outer_under(R.transpose(), R.transpose())};
        auto && rotator_back = Matrices::outer_under(R, R);

        // unclear behaviour. When I return this value as an
        // expression, clange segfaults or returns an uninitialised
        // tensor
        return muGrid::T4Mat<Real, Dim>(rotator_back * input * rotator_forward);
      }
    };
  }  // namespace internal

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  template <class Derived1>
  auto RotatorBase<Dim>::rotate(const Eigen::MatrixBase<Derived1> & input) const
      -> decltype(auto) {
    constexpr Dim_t tensor_rank{
        muGrid::EigenCheck::tensor_rank<Derived1, Dim>::value};

    return internal::RotationHelper<tensor_rank>::rotate(input, this->rot_mat);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  template <class Derived1>
  auto
  RotatorBase<Dim>::rotate_back(const Eigen::MatrixBase<Derived1> & input) const
      -> decltype(auto) {
    constexpr Dim_t tensor_rank{
        muGrid::EigenCheck::tensor_rank<Derived1, Dim>::value};

    return internal::RotationHelper<tensor_rank>::rotate(
        input, this->rot_mat.transpose());
  }

}  // namespace muSpectre

#endif  // SRC_COMMON_GEOMETRY_HH_
