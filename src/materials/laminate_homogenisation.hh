/**
 * @file   laminate_homogenisation.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   28 Sep 2018
 *
 * @brief Laminatehomogenisation enables one to obtain the resulting stress
 *        and stiffness tensors of a laminate pixel that is consisted of two
 *        materialswith a certain normal vector of their interface plane.
 *        note that it is supposed to be used in static way. so it does note
 *        any data member. It is merely a collection of functions used to
 *        calculate effective stress and stiffness.
 *
 *
 * Copyright © 2017 Till Junge, Ali Falsafi
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

#ifndef SRC_MATERIALS_LAMINATE_HOMOGENISATION_HH_
#define SRC_MATERIALS_LAMINATE_HOMOGENISATION_HH_

#include "common/geometry.hh"
#include "common/muSpectre_common.hh"
#include "libmugrid/field_map.hh"
#include "material_linear_anisotropic.hh"
#include "materials_toolbox.hh"
#include "material_muSpectre_base.hh"

#include <tuple>

namespace muSpectre {

  template <Index_t Dim>
  class LamCombination;

  template <Index_t Dim, Formulation Form>
  class LamHomogen {
   public:
    //! typedefs for data handled by this interface
    using Vec_t = Eigen::Matrix<Real, Dim, 1>;

    using Stiffness_t = muGrid::T4Mat<Real, Dim>;
    using Strain_t = Eigen::Matrix<Real, Dim, Dim>;
    using Stress_t = Strain_t;

    using Equation_index_t = std::array<std::array<Index_t, 2>, Dim>;
    using Equation_stiffness_t = Eigen::Matrix<Real, Dim, Dim>;
    using Equation_strain_t = Eigen::Matrix<Real, Dim, 1>;
    using Equation_stress_t = Equation_strain_t;

    using Parallel_index_t = std::conditional_t<
        Form == Formulation::finite_strain,
        std::array<std::array<Index_t, 2>, Dim *(Dim - 1)>,
        std::array<std::array<Index_t, 2>, (Dim - 1) * (Dim - 1)>>;

    using Parallel_strain_t =
        std::conditional_t<Form == Formulation::finite_strain,
                           Eigen::Matrix<Real, Dim *(Dim - 1), 1>,
                           Eigen::Matrix<Real, (Dim - 1) * (Dim - 1), 1>>;
    using Parallel_stress_t = Parallel_strain_t;

    using Function_t = std::function<std::tuple<Stress_t, Stiffness_t>(
        const Eigen::Ref<const Strain_t> &)>;

    // these two functions return the indices in column major strain and stress
    // tensors that the behavior is either serial or parallel concerning
    // combining the laminate layers as we know some indices in a laminate
    // structure act like the two phases are in serial and some others act like
    // two phases are parralell to each other
    inline static constexpr Parallel_index_t get_parallel_indices();
    inline static constexpr Equation_index_t get_equation_indices();

    // these functions return the parts of a stress or strain tensor that
    // behave either serial or parallel in a laminate structure
    // they are used to obtain a certian part of tensor which is needed by the
    // solver used in this struct to calculate the effective stress and
    // stiffness or compose the complete tensor from its sub sets.
    // obtain the serial part of stress or strain tensor
    template <class Derived>
    inline static Equation_strain_t
    get_equation_strain(const Eigen::MatrixBase<Derived> & E_total);
    template <class Derived>
    inline static Equation_stress_t
    get_equation_stress(const Eigen::MatrixBase<Derived> & S_total);
    template <class Derived>
    static Equation_stiffness_t
    get_equation_stiffness(const Eigen::MatrixBase<Derived> & C);

    // obtain the parallel part of stress or strain tensor
    template <class Derived1>
    inline static Parallel_strain_t
    get_parallel_strain(const Eigen::MatrixBase<Derived1> & E);
    template <class Derived1>
    inline static Parallel_stress_t
    get_parallel_stress(const Eigen::MatrixBase<Derived1> & S);

    // compose the complete strain or stress tensor from
    // its serial and parallel parts
    template <class Derived1, class Derived2>
    static Strain_t
    make_total_strain(const Eigen::MatrixBase<Derived1> & E_eq,
                      const Eigen::MatrixBase<Derived2> & E_para);
    template <class Derived1, class Derived2>
    inline static Stress_t
    make_total_stress(const Eigen::MatrixBase<Derived1> & S_eq,
                      const Eigen::MatrixBase<Derived2> & S_para);

    template <class Derived1, class Derived2>
    inline static Equation_strain_t
    linear_eqs(const Real & ratio, const Eigen::MatrixBase<Derived1> & E_0,
               const Eigen::MatrixBase<Derived2> & E_1);

    /**
     * the objective in homogenisation of a single laminate pixel is equating
     * the stress in the serial directions so the difference of stress between
     * their layers should tend to zero. this function return the stress
     * difference and the difference of Stiffness matrices which is used as the
     * Jacobian in the solution process
     */
    template <class Derived1, class Derived2>
    static std::tuple<Equation_stress_t, Equation_stiffness_t, Real>
    delta_equation_stress_stiffness_eval(
        const Function_t & mat_1_stress_eval,
        const Function_t & mat_2_stress_eval,
        const Eigen::MatrixBase<Derived1> & E_1,
        const Eigen::MatrixBase<Derived2> & E_2,
        const RotatorNormal<Dim> & rotator, const Real & ratio);

    template <class Derived1, class Derived2>
    static std::tuple<Equation_stress_t, Equation_stiffness_t, Real>
    delta_equation_stress_stiffness_eval_strain_1(
        const Function_t & mat_1_stress_eval,
        const Function_t & mat_2_stress_eval,
        const Eigen::MatrixBase<Derived1> & E_0,
        const Eigen::MatrixBase<Derived2> & E_1_rot,
        const RotatorNormal<Dim> & rotator, const Real & ratio);
    /**
     * the following functions claculate the energy computation error of the
     * solution. it will be used in each step of the solution to determine the
     * relevant difference that implementation of that step has had on
     * convergence to the solution.
     */
    inline static Real del_energy_eval(const Real & del_E_norm,
                                       const Real & delta_S_norm);
    /**
     *  These functions are used as intrface for combination functions, They are
     *also used for carrying out the stress transformation necessary for
     *combining stifness matrix in Finite-Strain formulation because the
     *combining formula from the bokk "Theory of Composites" are for symmetric
     *stifness matrices such as C and we have to transform stress to PK2 in
     *order to be able to use it
     */
    template <class Derived1, class Derived2>
    static Stress_t
    lam_stress_combine(const Eigen::MatrixBase<Derived1> & stress_1,
                       const Eigen::MatrixBase<Derived2> & stress_2,
                       const Real & ratio);

    static Stiffness_t lam_stiffness_combine(
        const Eigen::Ref<Stiffness_t> & stiffness_1,
        const Eigen::Ref<Stiffness_t> & stiffness_2, const Real & ratio,
        const Eigen::Ref<Strain_t> & F_1, const Eigen::Ref<Stress_t> & F_2,
        const Eigen::Ref<Strain_t> & P_1, const Eigen::Ref<Stress_t> & P_2,
        const Eigen::Ref<Strain_t> & F, const Eigen::Ref<Stress_t> & P);

    /**
     * This is the main solver function that might be called staically from
     * an external file. this will return the resultant stress and stiffness
     * tensor according to interanl "equilibrium" of the lamiante.
     * The inputs are :
     * 1- global Strain
     * 2- stress calculation function of the layer 1
     * 3- stress calculation function of the layer 2
     * 4- the ratio of the first material in the laminate sturucture of the
     * pixel 5- the normal vector of the interface of two layers 6- the
     * tolerance error for the internal solution of the laminate pixel 7- the
     * maximum iterations for the internal solution of the laminate pixel
     */
    static std::tuple<Index_t, Real, Strain_t, Strain_t>
    laminate_solver(const Eigen::Ref<Strain_t> & strain_coord,
                    const Function_t & mat_1_stress_eval,
                    const Function_t & mat_2_stress_eval, const Real & ratio,
                    const Eigen::Ref<Vec_t> & normal_vec,
                    const Real & tol = 1e-10, const Index_t & max_iter = 1000);

    /* ---------------------------------------------------------------------- */
    static Stress_t evaluate_stress(const Eigen::Ref<Strain_t> & strain_coord,
                                    const Function_t & mat_1_stress_eval,
                                    const Function_t & mat_2_stress_eval,
                                    const Real & ratio,
                                    const Eigen::Ref<Vec_t> & normal_vec,
                                    const Real & tol = 1e-10,
                                    const Index_t & max_iter = 1000);

    /* ---------------------------------------------------------------------- */
    static std::tuple<Stress_t, Stiffness_t> evaluate_stress_tangent(
        const Eigen::Ref<Strain_t> & strain_coord,
        const Function_t & mat_1_stress_eval,
        const Function_t & mat_2_stress_eval, const Real & ratio,
        const Eigen::Ref<Vec_t> & normal_vec, const Real & tol = 1e-10,
        const Index_t & max_iter = 1000);
  };  // LamHomogen

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim>
  class LamCombination {
   public:
    using Stiffness_t =
        typename LamHomogen<Dim, Formulation::small_strain>::Stiffness_t;
    using Stress_t =
        typename LamHomogen<Dim, Formulation::small_strain>::Stress_t;
    /**
     * This functions calculate the resultant stress and tangent matrices
     * according to the computed E_1 and E_2 from the solver.
     *
     */
    template <class Derived1, class Derived2>
    inline static Stress_t
    lam_S_combine(const Eigen::MatrixBase<Derived1> & S_1,
                  const Eigen::MatrixBase<Derived2> & S_2, const Real & ratio);
    template <class Derived1, class Derived2>
    static Stiffness_t lam_C_combine(const Eigen::MatrixBase<Derived1> & C_1,
                                     const Eigen::MatrixBase<Derived2> & C_2,
                                     const Real & ratio);
  };
  /* ---------------------------------------------------------------------- */

  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* -----------The following functions are used to obtain----------------- */
  /* -----------parts of the strian/stress.stiffness matrices-------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */

  /*---------------------------------------------------------------------- */
  template <>
  constexpr auto
  LamHomogen<threeD, Formulation::small_strain>::get_equation_indices()
      -> Equation_index_t {
    return Equation_index_t{{{0, 0}, {0, 1}, {0, 2}}};
  }
  template <>
  constexpr auto
  LamHomogen<twoD, Formulation::small_strain>::get_equation_indices()
      -> Equation_index_t {
    return Equation_index_t{{{0, 0}, {0, 1}}};
  }
  template <>
  constexpr auto LamHomogen<threeD, Formulation::native>::get_equation_indices()
      -> Equation_index_t {
    return Equation_index_t{{{0, 0}, {0, 1}, {0, 2}}};
  }
  template <>
  constexpr auto LamHomogen<twoD, Formulation::native>::get_equation_indices()
      -> Equation_index_t {
    return Equation_index_t{{{0, 0}, {0, 1}}};
  }
  template <>
  constexpr auto
  LamHomogen<threeD, Formulation::finite_strain>::get_equation_indices()
      -> Equation_index_t {
    return Equation_index_t{{{0, 0}, {0, 1}, {0, 2}}};
  }
  template <>
  constexpr auto
  LamHomogen<twoD, Formulation::finite_strain>::get_equation_indices()
      -> Equation_index_t {
    return Equation_index_t{{{0, 0}, {0, 1}}};
  }
  /* ---------------------------------------------------------------------- */
  template <>
  constexpr auto
  LamHomogen<threeD, Formulation::small_strain>::get_parallel_indices()
      -> Parallel_index_t {
    return Parallel_index_t{{{1, 1}, {1, 2}, {2, 1}, {2, 2}}};
  }
  template <>
  constexpr auto
  LamHomogen<twoD, Formulation::small_strain>::get_parallel_indices()
      -> Parallel_index_t {
    return Parallel_index_t{{{1, 1}}};
  }

  template <>
  constexpr auto LamHomogen<threeD, Formulation::native>::get_parallel_indices()
      -> Parallel_index_t {
    return Parallel_index_t{{{1, 1}, {1, 2}, {2, 1}, {2, 2}}};
  }
  template <>
  constexpr auto LamHomogen<twoD, Formulation::native>::get_parallel_indices()
      -> Parallel_index_t {
    return Parallel_index_t{{{1, 1}}};
  }

  template <>
  constexpr auto
  LamHomogen<threeD, Formulation::finite_strain>::get_parallel_indices()
      -> Parallel_index_t {
    return Parallel_index_t{{{1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}}};
  }
  template <>
  constexpr auto
  LamHomogen<twoD, Formulation::finite_strain>::get_parallel_indices()
      -> Parallel_index_t {
    return Parallel_index_t{{{1, 0}, {1, 1}}};
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, Formulation Form>
  template <class Derived>
  auto LamHomogen<Dim, Form>::get_equation_stress(
      const Eigen::MatrixBase<Derived> & S_total) -> Equation_stress_t {
    return get_equation_strain(S_total);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, Formulation Form>
  template <class Derived>
  auto LamHomogen<Dim, Form>::get_parallel_stress(
      const Eigen::MatrixBase<Derived> & S_total) -> Parallel_stress_t {
    return get_parallel_strain(S_total);
  }

  /* ---------------------------------------------------------------------- */

  template <Index_t Dim, Formulation Form>
  template <class Derived>
  auto LamHomogen<Dim, Form>::get_parallel_strain(
      const Eigen::MatrixBase<Derived> & E_total) -> Parallel_strain_t {
    Parallel_strain_t E_parallel;
    auto parallel_indices{get_parallel_indices()};
    for (auto && tup : akantu::enumerate(parallel_indices)) {
      auto && index{std::get<1>(tup)};
      auto counter{std::get<0>(tup)};
      E_parallel(counter) = E_total(index[0], index[1]);
    }
    return E_parallel;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, Formulation Form>
  template <class Derived>
  auto LamHomogen<Dim, Form>::get_equation_strain(
      const Eigen::MatrixBase<Derived> & E_total) -> Equation_strain_t {
    Equation_strain_t E_equation;
    auto equation_indices{get_equation_indices()};

    for (auto && tup : akantu::enumerate(equation_indices)) {
      auto && index = std::get<1>(tup);
      auto counter = std::get<0>(tup);
      E_equation(counter) = E_total(index[0], index[1]);
    }
    return E_equation;
  }

  /* ---------------------------------------------------------------------- */
  // linear equation are solved in the rotated coordiantes to obtain strain_2
  // form known strain_1 and then these two are passed to calculate
  // stress in the general corrdinates
  template <Index_t Dim, Formulation Form>
  template <class Derived1, class Derived2>
  auto
  LamHomogen<Dim, Form>::linear_eqs(const Real & ratio,
                                    const Eigen::MatrixBase<Derived1> & E_0_eq,
                                    const Eigen::MatrixBase<Derived2> & E_1_eq)
      -> Equation_strain_t {
    return ((E_0_eq - ratio * E_1_eq) / (1 - ratio));
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, Formulation Form>
  template <class Derived1, class Derived2>
  auto LamHomogen<Dim, Form>::make_total_stress(
      const Eigen::MatrixBase<Derived1> & S_eq,
      const Eigen::MatrixBase<Derived2> & S_para) -> Stress_t {
    return make_total_strain(S_eq, S_para);
  }

  /* ----------------------------------------------------------------------
   */
  template <Index_t Dim>
  template <class Derived1, class Derived2>
  auto
  LamCombination<Dim>::lam_S_combine(const Eigen::MatrixBase<Derived1> & S_1,
                                     const Eigen::MatrixBase<Derived2> & S_2,
                                     const Real & ratio) -> Stress_t {
    return ratio * S_1 + (1 - ratio) * S_2;
  }
}  // namespace muSpectre

#endif  // SRC_MATERIALS_LAMINATE_HOMOGENISATION_HH_
