/**
 * @file   test_material_phase_field_fracture.cc
 *
 * @author  W. Beck Andrews <william.beck.andrews@imtek.uni-freiburg.de>
 *
 * @date   24 Mar 2021
 *
 * @brief  Testing MaterialPhaseFieldFracture.  Basic functionality tests
 * plus two trivial cases (when the phase field is zero or when all
 * eigenvalues have the same sign, the material response is linear elastic),
 * and two test cases (with zero and non-zero Poisson's ratio) that compare
 * to stress tangents precomputed and checked against finite differences in
 * a separate Python notebook.
 *
 * Copyright © 2021 W. Beck Andrews
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

#include "tests.hh"
#include "libmugrid/test_goodies.hh"

#include "materials/material_phase_field_fracture.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/iterable_proxy.hh"

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(material_phase_field_fracture)
  /* ---------------------------------------------------------------------- */
  template <Index_t Dim>
  struct MaterialFixture {
    using Mat_t = MaterialPhaseFieldFracture<Dim>;
    const Real ksmall{1e-4};
    const Real phi{0.5};
    const Real Youngs_modulus{1.0e3};
    const Real Poisson_ratio{0.3};
    MaterialFixture() : mat("phasefield", mdim(), NbQuadPts(), ksmall) {}
    constexpr static Index_t mdim() { return Dim; }
    constexpr static Index_t sdim() { return mdim(); }
    constexpr static Index_t NbQuadPts() { return 1; }
    Mat_t mat;
  };

  using mats = boost::mpl::list<MaterialFixture<twoD>, MaterialFixture<threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mats, Fix) {
    BOOST_CHECK_EQUAL("phasefield", Fix::mat.get_name());
    auto & mat{Fix::mat};
    auto mdim{Fix::mdim()};
    BOOST_CHECK_EQUAL(mdim, mat.MaterialDimension());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_add_pixel, Fix, mats, Fix) {
    auto & mat{Fix::mat};
    muGrid::testGoodies::RandRange<Index_t> rng;
    const Dim_t nb_pixel{7}, box_size{17};
    for (Dim_t i = 0; i < nb_pixel; ++i) {
      const Index_t c{rng.randval(0, box_size)};
      BOOST_CHECK_NO_THROW(
          mat.add_pixel(c, Fix::Youngs_modulus, Fix::Poisson_ratio, Fix::phi));
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_set_get, Fix, mats, Fix) {
    auto & mat{Fix::mat};
    muGrid::testGoodies::RandRange<Real> rng;
    const Dim_t nb_pixel{7};
    for (Dim_t i = 0; i < nb_pixel; ++i) {
      mat.add_pixel(i, Fix::Youngs_modulus, Fix::Poisson_ratio, Fix::phi);
    }
    mat.initialise();
    for (Dim_t i = 0; i < nb_pixel; ++i) {
      const Real c{rng.randval(0, 10000)};
      mat.set_phase_field(i, c);
      BOOST_CHECK_EQUAL(mat.get_phase_field(i), c);
    }
  }
  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, bool is_compression_arg = false,
            bool is_shear_arg = false>
  struct MaterialZeroPhi {
    using Mat_t = MaterialPhaseFieldFracture<Dim>;
    const Real phi{0.0};
    const Real Youngs_modulus{1.0e3};
    const Real Poisson_ratio{0.3};
    const Real ksmall{1e-4};
    const bool is_shear{is_shear_arg};
    const bool is_compression{is_compression_arg};
    MaterialZeroPhi() : mat("phasefield", mdim(), NbQuadPts(), ksmall) {}
    constexpr static Index_t mdim() { return Dim; }
    constexpr static Index_t sdim() { return mdim(); }
    constexpr static Index_t NbQuadPts() { return 1; }
    const Real small_tol{1.e-12};
    Mat_t mat;
  };

  using mats_zero_phi = boost::mpl::list<
      MaterialZeroPhi<twoD, false, false>, MaterialZeroPhi<twoD, false, true>,
      MaterialZeroPhi<twoD, true, false>, MaterialZeroPhi<twoD, true, true>,
      MaterialZeroPhi<threeD, false, false>,
      MaterialZeroPhi<threeD, false, true>,
      MaterialZeroPhi<threeD, true, false>,
      MaterialZeroPhi<threeD, true, true>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_zero_phi, Fix, mats_zero_phi, Fix) {
    constexpr Dim_t Dim{Fix::mdim()};
    using Strain_t = Eigen::Matrix<Real, Dim, Dim>;
    using Stiffness_t = muGrid::T4Mat<Real, Dim>;
    auto & mat{Fix::mat};
    mat.add_pixel({0}, Fix::Youngs_modulus, Fix::Poisson_ratio, Fix::phi);
    mat.initialise();
    Strain_t E{};
    E.setZero();  // turns out E{} does a bad job of initializing to 0.0
    if (Fix::is_shear) {
      E(0, Dim - 1) = 0.001;
      E(Dim - 1, 0) = 0.001;
    } else {
      E(0, 0) = 0.001;
      E(Dim - 1, Dim - 1) = 0.001;
    }
    if (Fix::is_compression) {
      E = -E;
    }
    using Hooke = MatTB::Hooke<Dim, Strain_t, Stiffness_t>;
    Real lambda =
        Hooke::compute_lambda(Fix::Youngs_modulus, Fix::Poisson_ratio);
    Real mu = Hooke::compute_mu(Fix::Youngs_modulus, Fix::Poisson_ratio);
    Stiffness_t C_ref{Hooke::compute_C_T4(lambda, mu)};
    Strain_t stress_ref{Matrices::tensmult(C_ref, E)};
    Stiffness_t stress_tangent{};
    Strain_t stress{};
    std::tie(stress, stress_tangent) =
        mat.evaluate_stress_tangent(E, lambda, mu, Fix::phi, Fix::ksmall);
    Strain_t stress_2nd_fn{
        mat.evaluate_stress(E, lambda, mu, Fix::phi, Fix::ksmall)};
    BOOST_CHECK_LT((stress - stress_2nd_fn).norm(), Fix::small_tol);
    BOOST_CHECK_LT((stress - stress_ref).norm(), Fix::small_tol);
    BOOST_CHECK_LT((stress_tangent - C_ref).norm(), Fix::small_tol);
  }
  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, bool is_compression_arg = false>
  struct MaterialBiaxial {
    using Mat_t = MaterialPhaseFieldFracture<Dim>;
    const Real phi{0.5};
    const Real Youngs_modulus{1.0e3};
    const Real Poisson_ratio{0.3};
    const Real ksmall{1e-4};
    const bool is_compression{is_compression_arg};
    MaterialBiaxial() : mat("phasefield", mdim(), NbQuadPts(), ksmall) {}
    constexpr static Index_t mdim() { return Dim; }
    constexpr static Index_t sdim() { return mdim(); }
    constexpr static Index_t NbQuadPts() { return 1; }
    const Real small_tol{1.e-12};
    Mat_t mat;
  };

  using mats_biaxial = boost::mpl::list<
      MaterialBiaxial<twoD, false>, MaterialBiaxial<twoD, true>,
      MaterialBiaxial<threeD, false>, MaterialBiaxial<threeD, true>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_biaxial, Fix, mats_biaxial, Fix) {
    constexpr Dim_t Dim{Fix::mdim()};
    using Strain_t = Eigen::Matrix<Real, Dim, Dim>;
    using Stiffness_t = muGrid::T4Mat<Real, Dim>;
    auto & mat{Fix::mat};
    mat.add_pixel({0}, Fix::Youngs_modulus, Fix::Poisson_ratio, Fix::phi);
    mat.initialise();
    Strain_t E{};
    E.setZero();
    E(0, 0) = 0.001;
    E(Dim - 1, Dim - 1) = 0.001;
    Real interp;
    if (Fix::is_compression) {
      E = -E;
      interp = 1.0;
    } else {
      interp = (1.0 - Fix::phi) * (1.0 - Fix::phi) * (1.0 - Fix::ksmall) +
               Fix::ksmall;
    }
    using Hooke = MatTB::Hooke<Dim, Strain_t, Stiffness_t>;
    Real lambda =
        Hooke::compute_lambda(Fix::Youngs_modulus, Fix::Poisson_ratio);
    Real mu = Hooke::compute_mu(Fix::Youngs_modulus, Fix::Poisson_ratio);
    Stiffness_t C_ref{Hooke::compute_C_T4(lambda, mu) * interp};
    Strain_t stress_ref{Matrices::tensmult(C_ref, E)};
    Stiffness_t stress_tangent{};
    Strain_t stress{};
    std::tie(stress, stress_tangent) =
        mat.evaluate_stress_tangent(E, lambda, mu, Fix::phi, Fix::ksmall);
    Strain_t stress_2nd_fn{
        mat.evaluate_stress(E, lambda, mu, Fix::phi, Fix::ksmall)};
    BOOST_CHECK_LT((stress - stress_2nd_fn).norm(), Fix::small_tol);
    BOOST_CHECK_LT((stress - stress_ref).norm(), Fix::small_tol);
    BOOST_CHECK_LT((stress_tangent - C_ref).norm(), Fix::small_tol);
  }
  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, bool is_compression_arg = true>
  struct MaterialReferenceNoPoisson {
    using Mat_t = MaterialPhaseFieldFracture<Dim>;
    const Real phi{0.5};
    const Real Youngs_modulus{1e3};
    const Real Poisson_ratio{0.0};
    const Real ksmall{1e-4};
    const bool is_compression{is_compression_arg};
    MaterialReferenceNoPoisson()
        : mat("phasefield", mdim(), NbQuadPts(), ksmall) {}
    constexpr static Index_t mdim() { return Dim; }
    constexpr static Index_t sdim() { return mdim(); }
    constexpr static Index_t NbQuadPts() { return 1; }
    const Real large_tol{1.e-6};
    const Real small_tol{1.e-12};
    Mat_t mat;
  };

  using mats_reference_no_poisson =
      boost::mpl::list<MaterialReferenceNoPoisson<twoD>,
                       MaterialReferenceNoPoisson<threeD>,
                       MaterialReferenceNoPoisson<threeD, false>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_reference_no_poisson, Fix,
                                   mats_reference_no_poisson, Fix) {
    constexpr Dim_t Dim{Fix::mdim()};
    using Strain_t = Eigen::Matrix<Real, Dim, Dim>;
    using Stiffness_t = muGrid::T4Mat<Real, Dim>;
    auto & mat{Fix::mat};
    mat.add_pixel({0}, Fix::Youngs_modulus, Fix::Poisson_ratio, Fix::phi);
    mat.initialise();
    Strain_t E{};
    Stiffness_t temp{};
    if (Dim == 2) {
      E << 0.001, 0.001, 0.001, -0.0006;
      temp << 372.94683683, -132.11550336, -132.11550336, 17.85344640,
          -132.11550336, 301.09254430, 301.09254430, -160.68101760,
          -132.11550336, 301.09254430, 301.09254430, -160.68101760, 17.85344640,
          -160.68101760, -160.68101760, 841.42127037;
    } else {
      E << 0.001, 0.001, 0.0, 0.001, -0.0005, -0.0008, 0.0, -0.0008, -0.0012;
      if (Fix::is_compression) {
        temp << 371.37922461, -129.54246894, -0.65953777, -129.54246894,
            47.75065469, -34.98597465, -0.65953777, -34.98597465, 16.31350467,
            -129.54246894, 306.98560646, 35.04949180, 306.98560646,
            -173.17967126, 38.76862047, 35.04949180, 38.76862047, -7.90093056,
            -0.65953777, 35.04949180, 336.82948958, 35.04949180, 38.76862047,
            -95.44526361, 336.82948958, -95.44526361, 52.24718707,
            -129.54246894, 306.98560646, 35.04949180, 306.98560646,
            -173.17967126, 38.76862047, 35.04949180, 38.76862047, -7.90093056,
            47.75065469, -173.17967126, 38.76862047, -173.17967126,
            794.97516875, 53.25277198, 38.76862047, 53.25277198, -13.63284915,
            -34.98597465, 38.76862047, -95.44526361, 38.76862047, 53.25277198,
            438.19963534, -95.44526361, 438.19963534, 31.44786578, -0.65953777,
            35.04949180, 336.82948958, 35.04949180, 38.76862047, -95.44526361,
            336.82948958, -95.44526361, 52.24718707, -34.98597465, 38.76862047,
            -95.44526361, 38.76862047, 53.25277198, 438.19963534, -95.44526361,
            438.19963534, 31.44786578, 16.31350467, -7.90093056, 52.24718707,
            -7.90093056, -13.63284915, 31.44786578, 52.24718707, 31.44786578,
            982.85798622;
      } else {
        E = -E;
        temp << 878.69577539, 129.54246894, 0.65953777, 129.54246894,
            -47.75065469, 34.98597465, 0.65953777, 34.98597465, -16.31350467,
            129.54246894, 318.05189354, -35.04949180, 318.05189354,
            173.17967126, -38.76862047, -35.04949180, -38.76862047, 7.90093056,
            0.65953777, -35.04949180, 288.20801042, -35.04949180, -38.76862047,
            95.44526361, 288.20801042, 95.44526361, -52.24718707, 129.54246894,
            318.05189354, -35.04949180, 318.05189354, 173.17967126,
            -38.76862047, -35.04949180, -38.76862047, 7.90093056, -47.75065469,
            173.17967126, -38.76862047, 173.17967126, 455.09983125,
            -53.25277198, -38.76862047, -53.25277198, 13.63284915, 34.98597465,
            -38.76862047, 95.44526361, -38.76862047, -53.25277198, 186.83786466,
            95.44526361, 186.83786466, -31.44786578, 0.65953777, -35.04949180,
            288.20801042, -35.04949180, -38.76862047, 95.44526361, 288.20801042,
            95.44526361, -52.24718707, 34.98597465, -38.76862047, 95.44526361,
            -38.76862047, -53.25277198, 186.83786466, 95.44526361, 186.83786466,
            -31.44786578, -16.31350467, 7.90093056, -52.24718707, 7.90093056,
            13.63284915, -31.44786578, -52.24718707, -31.44786578, 267.21701378;
      }
    }
    Stiffness_t C_ref{muGrid::testGoodies::from_numpy(temp)};
    using Hooke = MatTB::Hooke<Dim, Strain_t, Stiffness_t>;
    Real lambda =
        Hooke::compute_lambda(Fix::Youngs_modulus, Fix::Poisson_ratio);
    Real mu = Hooke::compute_mu(Fix::Youngs_modulus, Fix::Poisson_ratio);
    Stiffness_t stress_tangent{};
    Strain_t stress{};
    std::tie(stress, stress_tangent) =
        mat.evaluate_stress_tangent(E, lambda, mu, Fix::phi, Fix::ksmall);
    Strain_t stress_2nd_fn{
        mat.evaluate_stress(E, lambda, mu, Fix::phi, Fix::ksmall)};
    Real error = (stress_tangent - C_ref).norm() / C_ref.norm();
    if (error > Fix::large_tol) {
      std::cout << "stiffness reference:\n" << C_ref << std::endl;
      std::cout << "stiffness computed:\n" << stress_tangent << std::endl;
    }

    BOOST_CHECK_LT(error, Fix::large_tol);
    BOOST_CHECK_LT((stress - stress_2nd_fn).norm(), Fix::small_tol);
  }
  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, bool is_compression_arg = true>
  struct MaterialReference {
    using Mat_t = MaterialPhaseFieldFracture<Dim>;
    const Real phi{0.5};
    const Real Youngs_modulus{1e3};
    const Real Poisson_ratio{0.3};
    const Real ksmall{1e-4};
    const bool is_compression{is_compression_arg};
    MaterialReference() : mat("phasefield", mdim(), NbQuadPts(), ksmall) {}
    constexpr static Index_t mdim() { return Dim; }
    constexpr static Index_t sdim() { return mdim(); }
    constexpr static Index_t NbQuadPts() { return 1; }
    const Real large_tol{1.e-6};
    const Real small_tol{1.e-12};
    Mat_t mat;
  };

  using mats_reference =
      boost::mpl::list<MaterialReference<twoD>, MaterialReference<threeD>,
                       MaterialReference<threeD, false>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_reference, Fix, mats_reference, Fix) {
    constexpr Dim_t Dim{Fix::mdim()};
    using Strain_t = Eigen::Matrix<Real, Dim, Dim>;
    using Stiffness_t = muGrid::T4Mat<Real, Dim>;
    auto & mat{Fix::mat};
    mat.add_pixel({0}, Fix::Youngs_modulus, Fix::Poisson_ratio, Fix::phi);
    mat.initialise();
    Strain_t E{};
    Stiffness_t temp{};
    if (Dim == 2) {
      E << 0.001, 0.001, 0.001, -0.0006;
      temp << 431.15622064, -101.62731028, -101.62731028, 158.00745877,
          -101.62731028, 231.60964946, 231.60964946, -123.60078277,
          -101.62731028, 231.60964946, 231.60964946, -123.60078277,
          158.00745877, -123.60078277, -123.60078277, 791.52116951;
    } else {
      E << 0.001, 0.001, 0.0, 0.001, -0.0005, -0.0008, 0.0, -0.0008, -0.0012;
      if (Fix::is_compression) {
        temp << 862.59940354, -99.64805303, -0.50733674, -99.64805303,
            613.65434976, -26.91228819, -0.50733674, -26.91228819, 589.47192667,
            -99.64805303, 236.14277420, 26.96114754, 236.14277420,
            -133.21513174, 29.82201574, 26.96114754, 29.82201574, -6.07763889,
            -0.50733674, 26.96114754, 259.09960737, 26.96114754, 29.82201574,
            -73.41943355, 259.09960737, -73.41943355, 40.19014390, -99.64805303,
            236.14277420, 26.96114754, 236.14277420, -133.21513174, 29.82201574,
            26.96114754, 29.82201574, -6.07763889, 613.65434976, -133.21513174,
            29.82201574, -133.21513174, 1188.44243750, 40.96367076, 29.82201574,
            40.96367076, 566.43626988, -26.91228819, 29.82201574, -73.41943355,
            29.82201574, 40.96367076, 337.07664257, -73.41943355, 337.07664257,
            24.19066599, -0.50733674, 26.96114754, 259.09960737, 26.96114754,
            29.82201574, -73.41943355, 259.09960737, -73.41943355, 40.19014390,
            -26.91228819, 29.82201574, -73.41943355, 29.82201574, 40.96367076,
            337.07664257, -73.41943355, 337.07664257, 24.19066599, 589.47192667,
            -6.07763889, 40.19014390, -6.07763889, 566.43626988, 24.19066599,
            40.19014390, 24.19066599, 1332.96768170;
      } else {
        E = -E;
        temp << 820.19386569, 99.64805303, 0.50733674, 99.64805303,
            107.54276562, 26.91228819, 0.50733674, 26.91228819, 131.72518872,
            99.64805303, 244.65530272, -26.96114754, 244.65530272, 133.21513174,
            -29.82201574, -26.96114754, -29.82201574, 6.07763889, 0.50733674,
            -26.96114754, 221.69846956, -26.96114754, -29.82201574, 73.41943355,
            221.69846956, 73.41943355, -40.19014390, 99.64805303, 244.65530272,
            -26.96114754, 244.65530272, 133.21513174, -29.82201574,
            -26.96114754, -29.82201574, 6.07763889, 107.54276562, 133.21513174,
            -29.82201574, 133.21513174, 494.35083173, -40.96367076,
            -29.82201574, -40.96367076, 154.76084550, 26.91228819, -29.82201574,
            73.41943355, -29.82201574, -40.96367076, 143.72143436, 73.41943355,
            143.72143436, -24.19066599, 0.50733674, -26.96114754, 221.69846956,
            -26.96114754, -29.82201574, 73.41943355, 221.69846956, 73.41943355,
            -40.19014390, 26.91228819, -29.82201574, 73.41943355, -29.82201574,
            -40.96367076, 143.72143436, 73.41943355, 143.72143436, -24.19066599,
            131.72518872, 6.07763889, -40.19014390, 6.07763889, 154.76084550,
            -24.19066599, -40.19014390, -24.19066599, 349.82558753;
      }
    }
    Stiffness_t C_ref{muGrid::testGoodies::from_numpy(temp)};
    using Hooke = MatTB::Hooke<Dim, Strain_t, Stiffness_t>;
    Real lambda =
        Hooke::compute_lambda(Fix::Youngs_modulus, Fix::Poisson_ratio);
    Real mu = Hooke::compute_mu(Fix::Youngs_modulus, Fix::Poisson_ratio);
    Stiffness_t stress_tangent{};
    Strain_t stress{};
    std::tie(stress, stress_tangent) =
        mat.evaluate_stress_tangent(E, lambda, mu, Fix::phi, Fix::ksmall);
    Strain_t stress_2nd_fn{
        mat.evaluate_stress(E, lambda, mu, Fix::phi, Fix::ksmall)};
    Real error = (stress_tangent - C_ref).norm() / C_ref.norm();
    if (not(error < Fix::large_tol)) {
      std::cout << "stiffness reference:\n" << C_ref << std::endl;
      std::cout << "stiffness computed:\n" << stress_tangent << std::endl;
    }

    BOOST_CHECK_LT(error, Fix::large_tol);
    BOOST_CHECK_LT((stress - stress_2nd_fn).norm(), Fix::small_tol);
  }
  BOOST_AUTO_TEST_SUITE_END()
}  // namespace muSpectre
