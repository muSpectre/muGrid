/**
 * @file   test_geometry.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   19 Apr 2018
 *
 * @brief  Tests for tensor rotations
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
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
 *
 */

#include "common/geometry.hh"
#include "tests.hh"
#include "libmugrid/test_goodies.hh"

#include <libmugrid/T4_map_proxy.hh>
#include <common/geometry.hh>

#include <Eigen/Dense>
#include <boost/mpl/list.hpp>

#include <cmath>
#include <iostream>

namespace muSpectre {

  enum class IsCollinear { yes, no };

  BOOST_AUTO_TEST_SUITE(geometry);

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim_>
  struct RotationFixture {
    static constexpr Dim_t Dim{Dim_};
    using Vec_t = Eigen::Matrix<Real, Dim, 1>;
    using Mat_t = Eigen::Matrix<Real, Dim, Dim>;
    using Ten_t = muGrid::T4Mat<Real, Dim>;
    static constexpr Dim_t get_Dim() { return Dim_; }

    using Rot_t = RotatorBase<Dim>;

    explicit RotationFixture(Rot_t rot_mat_inp)
        : rot_mat_holder{std::make_unique<Mat_t>(rot_mat_inp)},
          rot_mat{*this->rot_mat_holder}, rotator(rot_mat) {}

    std::unique_ptr<Vec_t> v_holder{std::make_unique<Vec_t>(Vec_t::Random())};
    const Vec_t & v{*this->v_holder};
    std::unique_ptr<Mat_t> m_holder{std::make_unique<Mat_t>(Mat_t::Random())};
    const Mat_t & m{*this->m_holder};
    std::unique_ptr<Ten_t> t_holder{std::make_unique<Ten_t>(Ten_t::Random())};
    const Ten_t & t{*this->t_holder};
    std::unique_ptr<Mat_t> rot_mat_holder;
    const Mat_t & rot_mat;

    Rot_t rotator;

    muGrid::testGoodies::RandRange<Real> rr{};
  };

  template <Dim_t Dim_, RotationOrder Rot>
  struct RotationAngleFixture {
    static constexpr Dim_t Dim{Dim_};
    using Parent = RotationFixture<Dim>;
    using Vec_t = Eigen::Matrix<Real, Dim, 1>;
    using Mat_t = Eigen::Matrix<Real, Dim, Dim>;
    using Ten_t = muGrid::T4Mat<Real, Dim>;
    using Angles_t = Eigen::Matrix<Real, (Dim == threeD ? 3 : 1), 1>;
    using RotAng_t = RotatorAngle<Dim, Rot>;

    static constexpr RotationOrder EulerOrder{Rot};
    static constexpr Dim_t get_Dim() { return Dim_; }

    RotationAngleFixture() : rotator{euler} {}

    std::unique_ptr<Vec_t> v_holder{std::make_unique<Vec_t>(Vec_t::Random())};
    const Vec_t & v{*this->v_holder};
    std::unique_ptr<Mat_t> m_holder{std::make_unique<Mat_t>(Mat_t::Random())};
    const Mat_t & m{*this->m_holder};
    std::unique_ptr<Ten_t> t_holder{std::make_unique<Ten_t>(Ten_t::Random())};
    const Ten_t & t{*this->t_holder};
    std::unique_ptr<Angles_t> euler_holder{
        std::make_unique<Angles_t>(2 * muGrid::pi * Angles_t::Random())};
    const Angles_t & euler{*this->euler_holder};
    RotatorAngle<Dim, Rot> rotator;
  };

  template <Dim_t Dim_, IsCollinear is_aligned = IsCollinear::no>
  struct RotationTwoVecFixture {
    static constexpr Dim_t Dim{Dim_};
    using Parent = RotationFixture<Dim>;
    using Vec_t = Eigen::Matrix<Real, Dim, 1>;
    using Mat_t = Eigen::Matrix<Real, Dim, Dim>;
    using Ten_t = muGrid::T4Mat<Real, Dim>;

    static constexpr Dim_t get_Dim() { return Dim_; }

    RotationTwoVecFixture()
        : vec_ref_holder{std::make_unique<Vec_t>(this->ref_vec_maker())},
          vec_ref{*this->vec_ref_holder},
          vec_des_holder{std::make_unique<Vec_t>(this->des_vec_maker())},
          vec_des{*this->vec_des_holder},
          rotator(*this->vec_ref_holder, *this->vec_des_holder) {}

    Vec_t ref_vec_maker() {
      Vec_t ret_vec{Vec_t::Random()};
      return ret_vec / ret_vec.norm();
    }
    Vec_t des_vec_maker() {
      if (is_aligned == IsCollinear::yes) {
        return -this->vec_ref;
      } else {
        Vec_t ret_vec{Vec_t::Random()};
        return ret_vec / ret_vec.norm();
      }
    }

    std::unique_ptr<Vec_t> v_holder{std::make_unique<Vec_t>(Vec_t::Random())};
    const Vec_t & v{*this->v_holder};
    std::unique_ptr<Mat_t> m_holder{std::make_unique<Mat_t>(Mat_t::Random())};
    const Mat_t & m{*this->m_holder};
    std::unique_ptr<Ten_t> t_holder{std::make_unique<Ten_t>(Ten_t::Random())};
    const Ten_t & t{*this->t_holder};
    std::unique_ptr<Vec_t> vec_ref_holder{
        std::make_unique<Vec_t>(Vec_t::Random())};
    const Vec_t & vec_ref{*this->vec_ref_holder};
    std::unique_ptr<Vec_t> vec_des_holder{
        std::make_unique<Vec_t>(Vec_t::Random())};
    const Vec_t & vec_des{*this->vec_des_holder};

    RotatorTwoVec<Dim> rotator;
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim_, IsCollinear is_aligned = IsCollinear::no>
  struct RotationNormalFixture {
    static constexpr Dim_t Dim{Dim_};
    using Parent = RotationFixture<Dim>;
    using Vec_t = Eigen::Matrix<Real, Dim, 1>;
    using Mat_t = Eigen::Matrix<Real, Dim, Dim>;
    using Ten_t = muGrid::T4Mat<Real, Dim>;
    static constexpr Dim_t get_Dim() { return Dim_; }

    // Constructor :
    RotationNormalFixture()
        : vec_norm_holder{std::make_unique<Vec_t>(this->vec_norm_maker())},
          vec_norm{*this->vec_norm_holder}, rotator(vec_norm) {}

    Vec_t vec_norm_maker() {
      if (is_aligned == IsCollinear::yes) {
        return -Vec_t::UnitX();
      } else {
        Vec_t ret_vec{Vec_t::Random()};
        return ret_vec / ret_vec.norm();
      }
    }

    std::unique_ptr<Vec_t> v_holder{std::make_unique<Vec_t>(Vec_t::Random())};
    const Vec_t & v{*this->v_holder};
    std::unique_ptr<Mat_t> m_holder{std::make_unique<Mat_t>(Mat_t::Random())};
    const Mat_t & m{*this->m_holder};
    std::unique_ptr<Ten_t> t_holder{std::make_unique<Ten_t>(Ten_t::Random())};
    const Ten_t & t{*this->t_holder};

    std::unique_ptr<Vec_t> vec_norm_holder;
    Vec_t & vec_norm{*this->v_holder};

    RotatorNormal<Dim> rotator;
  };
  /* ---------------------------------------------------------------------- */
  using fix_list = boost::mpl::list<
      RotationAngleFixture<twoD, RotationOrder::Z>,
      RotationAngleFixture<threeD, RotationOrder::ZXYTaitBryan>,
      RotationNormalFixture<twoD>, RotationNormalFixture<threeD>,
      RotationTwoVecFixture<twoD>, RotationTwoVecFixture<threeD>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(rotation_test, Fix, fix_list, Fix) {
    using Vec_t = typename Fix::Vec_t;
    using Mat_t = typename Fix::Mat_t;
    using Ten_t = typename Fix::Ten_t;

    constexpr const Dim_t Dim{Fix::get_Dim()};

    const Vec_t & v{Fix::v};
    const Mat_t & m{Fix::m};
    const Ten_t & t{Fix::t};
    const Mat_t & R{Fix::rotator.get_rot_mat()};

    Vec_t v_ref{R * v};
    Mat_t m_ref{R * m * R.transpose()};
    Ten_t t_ref{Ten_t::Zero()};
    for (int i = 0; i < Dim; ++i) {
      for (int a = 0; a < Dim; ++a) {
        for (int l = 0; l < Dim; ++l) {
          for (int b = 0; b < Dim; ++b) {
            for (int m = 0; m < Dim; ++m) {
              for (int n = 0; n < Dim; ++n) {
                for (int o = 0; o < Dim; ++o) {
                  for (int p = 0; p < Dim; ++p) {
                    muGrid::get(t_ref, a, b, o, p) +=
                        R(a, i) * R(b, l) * muGrid::get(t, i, l, m, n) *
                        R(o, m) * R(p, n);
                  }
                }
              }
            }
          }
        }
      }
    }

    Vec_t v_rotator(Fix::rotator.rotate(v));
    Mat_t m_rotator(Fix::rotator.rotate(m));
    Ten_t t_rotator(Fix::rotator.rotate(t));

    auto v_error{(v_rotator - v_ref).norm() / v_ref.norm()};
    BOOST_CHECK_LT(v_error, tol);

    auto m_error{(m_rotator - m_ref).norm() / m_ref.norm()};
    BOOST_CHECK_LT(m_error, tol);

    auto t_error{(t_rotator - t_ref).norm() / t_ref.norm()};
    BOOST_CHECK_LT(t_error, tol);
    if (t_error >= tol) {
      std::cout << "t4_reference:" << std::endl << t_ref << std::endl;
      std::cout << "t4_rotator:" << std::endl << t_rotator << std::endl;
    }

    Vec_t v_back{Fix::rotator.rotate_back(v_rotator)};
    Mat_t m_back{Fix::rotator.rotate_back(m_rotator)};
    Ten_t t_back{Fix::rotator.rotate_back(t_rotator)};

    v_error = (v_back - v).norm() / v.norm();
    BOOST_CHECK_LT(v_error, tol);

    m_error = (m_back - m).norm() / m.norm();
    BOOST_CHECK_LT(m_error, tol);

    t_error = (t_back - t).norm() / t.norm();
    BOOST_CHECK_LT(t_error, tol);
  }

  /* ---------------------------------------------------------------------- */
  using threeD_list = boost::mpl::list<
      RotationAngleFixture<threeD, RotationOrder::ZXYTaitBryan>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(rotation_matrix_test, Fix, threeD_list,
                                   Fix) {
    using Mat_t = typename Fix::Mat_t;
    auto c{Eigen::cos(Fix::euler.array())};
    Real c_1{c[0]}, c_2{c[1]}, c_3{c[2]};
    auto s{Eigen::sin(Fix::euler.array())};
    Real s_1{s[0]}, s_2{s[1]}, s_3{s[2]};
    Mat_t rot_ref;

    switch (Fix::EulerOrder) {
    case RotationOrder::ZXYTaitBryan: {
      rot_ref << c_1 * c_3 - s_1 * s_2 * s_3, -c_2 * s_1,
          c_1 * s_3 + c_3 * s_1 * s_2, c_3 * s_1 + c_1 * s_2 * s_3, c_1 * c_2,
          s_1 * s_3 - c_1 * c_3 * s_2, -c_2 * s_3, s_2, c_2 * c_3;

      break;
    }
    default: {
      BOOST_CHECK(false);
      break;
    }
    }
    auto err{(rot_ref - Fix::rotator.get_rot_mat()).norm()};
    BOOST_CHECK_LT(err, tol);
    if (not(err < tol)) {
      std::cout << "Reference:" << std::endl << rot_ref << std::endl;
      std::cout << "Rotator:" << std::endl
                << Fix::rotator.get_rot_mat() << std::endl;
    }
  }

  /* ---------------------------------------------------------------------- */
  using twovec_list =
      boost::mpl::list<RotationTwoVecFixture<threeD>,
                       RotationTwoVecFixture<twoD>,
                       RotationTwoVecFixture<threeD, IsCollinear::yes>,
                       RotationTwoVecFixture<twoD, IsCollinear::yes>>;

  /* ----------------------------------------------------------------------*/
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(rotation_twovec_test, Fix, twovec_list,
                                   Fix) {
    using Vec_t = typename Fix::Vec_t;
    Vec_t vec_ref{Fix::vec_ref};
    Vec_t vec_des{Fix::vec_des};
    Vec_t vec_res{Fix::rotator.rotate(vec_ref)};
    Vec_t vec_back{Fix::rotator.rotate_back(vec_res)};

    auto err_f{(vec_res - vec_des).norm()};
    BOOST_CHECK_LT(err_f, tol);
    if (err_f >= tol) {
      std::cout << "Destination:" << std::endl << vec_des << std::endl;
      std::cout << "Rotated:" << std::endl << vec_res << std::endl;
    }
    auto err_b{(vec_back - vec_ref).norm()};
    BOOST_CHECK_LT(err_b, tol);
    if (err_b >= tol) {
      std::cout << "Refrence:" << std::endl << vec_ref << std::endl;
      std::cout << "Rotated Back:" << std::endl << vec_back << std::endl;
    }
  }

  /* ---------------------------------------------------------------------- */
  using normal_list =
      boost::mpl::list<RotationNormalFixture<threeD>,
                       RotationNormalFixture<twoD>,
                       RotationNormalFixture<threeD, IsCollinear::yes>,
                       RotationNormalFixture<twoD, IsCollinear::yes>>;

  /* ----------------------------------------------------------------------*/
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(rotation_normal_test, Fix, normal_list,
                                   Fix) {
    using Vec_t = typename Fix::Vec_t;
    Vec_t vec_ref{Fix::vec_norm};
    Vec_t vec_des{Vec_t::UnitX()};
    Vec_t vec_res{Fix::rotator.rotate_back(vec_ref)};
    Vec_t vec_back{Fix::rotator.rotate(vec_res)};

    // cehcking whether the reuslt of the rotation of the vector rotated
    // is aligned with the destination which in face x-axi
    auto err_f{(vec_res - vec_des).norm()};
    BOOST_CHECK_LT(err_f, tol);
    if (err_f >= tol) {
      std::cout << "Destination:" << std::endl << vec_des << std::endl;
      std::cout << "Rotated:" << std::endl << vec_res << std::endl;
    }

    // checking if the result of rotating back the result of rotaion (x-axis)
    // is aligned with the original vector before rotation

    auto err_b{(vec_back - vec_ref).norm()};
    BOOST_CHECK_LT(err_b, tol);
    if (err_b >= tol) {
      std::cout << "Refrence:" << std::endl << vec_ref << std::endl;
      std::cout << "Rotated Back:" << std::endl << vec_back << std::endl;
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
