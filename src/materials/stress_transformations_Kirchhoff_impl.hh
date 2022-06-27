/**
 * @file   stress_transformations_Kirchhoff_impl.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   29 Oct 2018
 *
 * @brief  Implementation of stress conversions for Kirchhoff stress
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

#ifndef SRC_MATERIALS_STRESS_TRANSFORMATIONS_KIRCHHOFF_IMPL_HH_
#define SRC_MATERIALS_STRESS_TRANSFORMATIONS_KIRCHHOFF_IMPL_HH_
namespace muSpectre {
  namespace MatTB {
    namespace internal {
      /**
       * Specialisation for the case where we get Kirchhoff stress (τ)
       */
      template <Dim_t Dim, StrainMeasure StrainM>
      struct PK1_stress<Dim, StressMeasure::Kirchhoff, StrainM>
          : public PK1_stress<Dim, StressMeasure::no_stress_,
                              StrainMeasure::no_strain_> {
        //! returns the converted stress
        template <class Strain_t, class Stress_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && tau) {
          return tau * F.inverse().transpose();
        }
      };

      /**
       * Specialisation for the case where we get Kirchhoff stress (τ) derived
       * with respect to Gradient
       */
      template <Dim_t Dim>
      struct PK1_stress<Dim, StressMeasure::Kirchhoff,
                        StrainMeasure::PlacementGradient>
          : public PK1_stress<Dim, StressMeasure::Kirchhoff,
                              StrainMeasure::no_strain_> {
        //! short-hand
        using Parent = PK1_stress<Dim, StressMeasure::Kirchhoff,
                                  StrainMeasure::no_strain_>;
        using Parent::compute;
        //! returns the converted stress and stiffness
        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && tau,
                                             Tangent_t && C) {
          using T4_t = muGrid::T4Mat<Real, Dim>;
          using T2_t = Eigen::Matrix<Real, Dim, Dim>;
          T2_t F_inv{F.inverse()};
          T4_t K{T4_t::Zero()};

          // K = [I _⊗  F⁻¹] c - [τF⁻ᵀ ⁻⊗ F⁻¹]
          for (Dim_t i{0}; i < Dim; ++i) {
            for (Dim_t j{0}; j < Dim; ++j) {
              for (Dim_t k{0}; k < Dim; ++k) {
                for (Dim_t l{0}; l < Dim; ++l) {
                  for (Dim_t n{0}; n < Dim; ++n) {
                    get(K, i, j, k, l) += F_inv(j, n) * get(C, i, n, k, l);
                  }
                  for (Dim_t a{0}; a < Dim; ++a) {
                    get(K, i, j, k, l) -=
                        (tau(i, a) * F_inv(l, a) * F_inv(j, k));
                  }
                }
              }
            }
          }
          T2_t P{tau * F_inv.transpose()};
          return std::make_tuple(std::move(P), std::move(K));
        }
      };

      /**
       * Specialisation for the case where we get Kirchhoff stress (τ) derived
       * with respect to GreenLagrange
       */
      template <Dim_t Dim>
      struct PK1_stress<Dim, StressMeasure::Kirchhoff,
                        StrainMeasure::GreenLagrange>
          : public PK1_stress<Dim, StressMeasure::Kirchhoff,
                              StrainMeasure::no_strain_> {
        //! short-hand
        using Parent = PK1_stress<Dim, StressMeasure::Kirchhoff,
                                  StrainMeasure::no_strain_>;
        using Parent::compute;
        //! returns the converted stress and stiffness
        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && tau,
                                             Tangent_t && C) {
          using T4_t = muGrid::T4Mat<Real, Dim>;
          using T2_t = Eigen::Matrix<Real, Dim, Dim>;
          T2_t F_inv{F.inverse()};
          T4_t K{T4_t::Zero()};

          // K = [I _⊗  F⁻¹] C [Fᵀ _⊗  I] - [τF⁻ᵀ ⁻⊗ F⁻¹]
          for (Dim_t i{0}; i < Dim; ++i) {
            for (Dim_t j{0}; j < Dim; ++j) {
              for (Dim_t k{0}; k < Dim; ++k) {
                for (Dim_t l{0}; l < Dim; ++l) {
                  for (Dim_t n{0}; n < Dim; ++n) {
                    for (Dim_t s{0}; s < Dim; ++s) {
                      get(K, i, j, k, l) +=
                        F_inv(j, n) * get(C, i, n, s, l) * F(k, s);
                    }
                  }
                  for (Dim_t a{0}; a < Dim; ++a) {
                    get(K, i, j, k, l) -=
                        (tau(i, a) * F_inv(l, a) * F_inv(j, k));
                  }
                }
              }
            }
          }
          T2_t P{tau * F_inv.transpose()};

          return std::make_tuple(std::move(P), std::move(K));
        }
      };

      /**
       * Specialisation for the case where we get Kirchhoff stress (τ) derived
       * with respect to the logarithmic strain (log of right stretch)
       */
      template <Dim_t Dim>
      struct PK1_stress<Dim, StressMeasure::Kirchhoff,
                        StrainMeasure::Log>
          : public PK1_stress<Dim, StressMeasure::Kirchhoff,
                              StrainMeasure::no_strain_> {
        //! short-hand
        using Parent = PK1_stress<Dim, StressMeasure::Kirchhoff,
                                  StrainMeasure::no_strain_>;
        using Parent::compute;
        //! returns the converted stress and stiffness
        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && tau,
                                             Tangent_t && C) {
          using T4_t = muGrid::T4Mat<Real, Dim>;
          using T2_t = Eigen::Matrix<Real, Dim, Dim>;
          using Vec_t = Eigen::Matrix<Real, Dim, 1>;
          T2_t F_inv{F.inverse()};
          T4_t K{T4_t::Zero()};
          T2_t A{F.transpose() * F};

          // compute derivative ∂ln(A)/∂A, see (77) through (80) in Geers 2003
          // (https://doi.org/10.1016/j.cma.2003.07.014)
          T4_t dlnA_dA{T4_t::Zero()};
          const muGrid::SelfAdjointDecomp_t<Dim> spectral_decomp{
                                        muGrid::spectral_decomposition(A)};
          {
            const Vec_t & eig_vals{spectral_decomp.eigenvalues()};
            const Vec_t log_eig_vals{eig_vals.array().log().matrix()};
            const T2_t & eig_vecs{spectral_decomp.eigenvectors()};

            T2_t g_vals{};
            for (int i{0}; i < Dim; ++i) {
              g_vals(i, i) = 1 / eig_vals(i);
              for (int j{i + 1}; j < Dim; ++j) {
                if (std::abs((eig_vals(i) - eig_vals(j)) / eig_vals(i))
                    < 1e-12) {
                  g_vals(i, j) = g_vals(j, i) = g_vals(i, i);
                } else {
                  g_vals(i, j) = ((log_eig_vals(j) - log_eig_vals(i)) /
                                                  (eig_vals(j) - eig_vals(i)));
                  g_vals(j, i) = g_vals(i, j);
                }
              }
            }

            for (int i{0}; i < Dim; ++i) {
              for (int j{0}; j < Dim; ++j) {
                T2_t dyad = eig_vecs.col(i) * eig_vecs.col(j).transpose();
                T4_t outerDyad = Matrices::outer(dyad, dyad.transpose());
                dlnA_dA += g_vals(i, j) * outerDyad;
              }
            }
            }

          double helper;
          // K = 0.5 [I _⊗  F⁻¹] C [∂ln(A)/∂A + [∂ln(A)/∂A]^RT] [Fᵀ _⊗  I]
          //                                                     - [τF⁻ᵀ ⁻⊗ F⁻¹]
          for (Dim_t i{0}; i < Dim; ++i) {
            for (Dim_t j{0}; j < Dim; ++j) {
              for (Dim_t k{0}; k < Dim; ++k) {
                for (Dim_t l{0}; l < Dim; ++l) {
                  for (Dim_t n{0}; n < Dim; ++n) {
                    get(K, i, j, k, l) -=
                        (tau(i, n) * F_inv(l, n) * F_inv(j, k));
                    for (Dim_t m{0}; m < Dim; ++m) {
                      for (Dim_t o{0}; o < Dim; ++o) {
                        helper = 0;
                        for (Dim_t p{0}; p < Dim; ++p) {
                          helper += F(k, p) * (get(dlnA_dA, m, o, p, l)
                                                   + get(dlnA_dA, m, o, l, p));
                        }
                        get(K, i, j, k, l) +=
                          0.5 * F_inv(j, n) * get(C, i, n, m, o) * helper;
                      }
                    }
                  }
                }
              }
            }
          }
          T2_t P{tau * F_inv.transpose()};

          return std::make_tuple(std::move(P), std::move(K));
        }
      };

      /**
       * Specialisation for the case where we get Kirchhoff stress (τ) derived
       * with respect to the logarithmic strain (log of left stretch)
       */
      template <Dim_t Dim>
      struct PK1_stress<Dim, StressMeasure::Kirchhoff,
                        StrainMeasure::LogLeftStretch>
          : public PK1_stress<Dim, StressMeasure::Kirchhoff,
                              StrainMeasure::no_strain_> {
        //! short-hand
        using Parent = PK1_stress<Dim, StressMeasure::Kirchhoff,
                                  StrainMeasure::no_strain_>;
        using Parent::compute;
        //! returns the converted stress and stiffness
        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && tau,
                                             Tangent_t && C) {
          using T4_t = muGrid::T4Mat<Real, Dim>;
          using T2_t = Eigen::Matrix<Real, Dim, Dim>;
          using Vec_t = Eigen::Matrix<Real, Dim, 1>;
          T2_t F_inv{F.inverse()};
          T4_t K{T4_t::Zero()};
          T2_t B{F * F.transpose()};

          // compute derivative ∂ln(B)/∂B, see (77) through (80) in Geers 2003
          // (https://doi.org/10.1016/j.cma.2003.07.014)
          T4_t dlnB_dB{T4_t::Zero()};
          const muGrid::SelfAdjointDecomp_t<Dim> spectral_decomp{
                                        muGrid::spectral_decomposition(B)};
          {
            const Vec_t & eig_vals{spectral_decomp.eigenvalues()};
            const Vec_t log_eig_vals{eig_vals.array().log().matrix()};
            const T2_t & eig_vecs{spectral_decomp.eigenvectors()};

            T2_t g_vals{};
            for (int i{0}; i < Dim; ++i) {
              g_vals(i, i) = 1 / eig_vals(i);
              for (int j{i + 1}; j < Dim; ++j) {
                if (std::abs((eig_vals(i) - eig_vals(j)) / eig_vals(i))
                    < 1e-12) {
                  g_vals(i, j) = g_vals(j, i) = g_vals(i, i);
                } else {
                  g_vals(i, j) = ((log_eig_vals(j) - log_eig_vals(i)) /
                                                  (eig_vals(j) - eig_vals(i)));
                  g_vals(j, i) = g_vals(i, j);
                }
              }
            }

            for (int i{0}; i < Dim; ++i) {
              for (int j{0}; j < Dim; ++j) {
                T2_t dyad = eig_vecs.col(i) * eig_vecs.col(j).transpose();
                T4_t outerDyad = Matrices::outer(dyad, dyad.transpose());
                dlnB_dB += g_vals(i, j) * outerDyad;
              }
            }
            }

          double helper;
          // K = 0.5 [I _⊗  F⁻¹] C [∂ln(B)/∂B + [∂ln(B)/∂B]^RT] [I _⊗  F]
          //                                                     - [τF⁻ᵀ ⁻⊗ F⁻¹]
          for (Dim_t i{0}; i < Dim; ++i) {
            for (Dim_t j{0}; j < Dim; ++j) {
              for (Dim_t k{0}; k < Dim; ++k) {
                for (Dim_t l{0}; l < Dim; ++l) {
                  for (Dim_t n{0}; n < Dim; ++n) {
                    get(K, i, j, k, l) -=
                        (tau(i, n) * F_inv(l, n) * F_inv(j, k));
                    for (Dim_t m{0}; m < Dim; ++m) {
                      for (Dim_t o{0}; o < Dim; ++o) {
                        helper = 0;
                        for (Dim_t p{0}; p < Dim; ++p) {
                          helper += F(p, l) * (get(dlnB_dB, m, o, k, p)
                                                   + get(dlnB_dB, m, o, p, k));
                        }
                        get(K, i, j, k, l) +=
                          0.5 * F_inv(j, n) * get(C, i, n, m, o) * helper;
                      }
                    }
                  }
                }
              }
            }
          }
          T2_t P{tau * F_inv.transpose()};

          return std::make_tuple(std::move(P), std::move(K));
        }
      };

      /**
       * Specialisation for the case where we get Kirchhoff stress (τ) and we
       * need PK2(S)
       */
      template <Dim_t Dim, StrainMeasure StrainM>
      struct PK2_stress<Dim, StressMeasure::Kirchhoff, StrainM>
          : public PK2_stress<Dim, StressMeasure::no_stress_,
                              StrainMeasure::no_strain_> {
        //! returns the converted stress
        template <class Strain_t, class Stress_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && tau) {
          return F.inverse() * tau * F.inverse().transpose();
        }
      };

    }  // namespace internal

  }  // namespace MatTB

}  // namespace muSpectre

#endif  // SRC_MATERIALS_STRESS_TRANSFORMATIONS_KIRCHHOFF_IMPL_HH_
