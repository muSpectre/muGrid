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
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
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

  template <Dim_t Dim>
  class LamCombination;

  template <Dim_t Dim, Formulation Form>
  class LamHomogen {
   public:
    //! typedefs for data handled by this interface
    using Vec_t = Eigen::Matrix<Real, Dim, 1>;

    using Stiffness_t = muGrid::T4Mat<Real, Dim>;
    using Strain_t = Eigen::Matrix<Real, Dim, Dim>;
    using Stress_t = Strain_t;

    using Equation_index_t = std::array<std::array<Dim_t, 2>, Dim>;
    using Equation_stiffness_t = Eigen::Matrix<Real, Dim, Dim>;
    using Equation_strain_t = Eigen::Matrix<Real, Dim, 1>;
    using Equation_stress_t = Equation_strain_t;

    using Parallel_index_t = std::conditional_t<
        Form == Formulation::finite_strain,
        std::array<std::array<Dim_t, 2>, Dim *(Dim - 1)>,
        std::array<std::array<Dim_t, 2>, (Dim - 1) * (Dim - 1)>>;

    using Parallel_strain_t =
        std::conditional_t<Form == Formulation::finite_strain,
                           Eigen::Matrix<Real, Dim *(Dim - 1), 1>,
                           Eigen::Matrix<Real, (Dim - 1) * (Dim - 1), 1>>;
    using Parallel_stress_t = Parallel_strain_t;

    using Function_t = std::function<std::tuple<Stress_t, Stiffness_t>(
        const Eigen::Ref<const Strain_t> &)>;

    // these two functions return the indices in column major strain and stress
    // tensors that the behavior is either serial or parallel concerning
    // combining the laminate layers as we know some indices in a lmainate
    // structure act like the two phases are in serial and some others act like
    // two phases are parralell to each other
    inline static constexpr auto get_parallel_indices() -> Parallel_index_t;
    inline static constexpr auto get_equation_indices() -> Equation_index_t;
    // these functions return the parts of a stress or strain tensor that
    // behave either serial or parallel in a laminate structure
    // they are used to obtain a certian part of tensor which is needed by the
    // solver used in this struct to calculate the effective stress and
    // stiffness or compose the complete tensor from its sub sets.

    // obtain the serial part of stress or strain tensor
    template <class Derived>
    inline static auto
    get_equation_strain(const Eigen::MatrixBase<Derived> & E_total)
        -> Equation_strain_t;
    template <class Derived>
    inline static auto
    get_equation_stress(const Eigen::MatrixBase<Derived> & S_total)
        -> Equation_stress_t;
    template <class Derived>
    inline static auto
    get_equation_stiffness(const Eigen::MatrixBase<Derived> & C)
        -> Equation_stiffness_t;

    // obtain the parallel part of stress or strain tensor
    template <class Derived1>
    inline static auto
    get_parallel_strain(const Eigen::MatrixBase<Derived1> & E)
        -> Parallel_strain_t;
    template <class Derived1>
    inline static auto
    get_parallel_stress(const Eigen::MatrixBase<Derived1> & S)
        -> Parallel_stress_t;

    // compose the complete strain or stress tensor from
    // its serial and parallel parts
    template <class Derived1, class Derived2>
    inline static auto
    make_total_strain(const Eigen::MatrixBase<Derived1> & E_eq,
                      const Eigen::MatrixBase<Derived2> & E_para) -> Strain_t;
    template <class Derived1, class Derived2>
    inline static auto
    make_total_stress(const Eigen::MatrixBase<Derived1> & S_eq,
                      const Eigen::MatrixBase<Derived2> & S_para) -> Stress_t;

    template <class Derived1, class Derived2>
    inline static auto linear_eqs(Real ratio,
                                  const Eigen::MatrixBase<Derived1> & E_0,
                                  const Eigen::MatrixBase<Derived2> & E_1)
        -> Equation_strain_t;

    /**
     * the objective in homogenisation of a single laminate pixel is equating
     * the stress in the serial directions so the difference of stress between
     * their layers should tend to zero. this function return the stress
     * difference and the difference of Stiffness matrices which is used as the
     * Jacobian in the solution process
     */
    template <class Derived1, class Derived2>
    inline static auto delta_equation_stress_stiffness_eval(
        const Function_t & mat_1_stress_eval,
        const Function_t & mat_2_stress_eval,
        const Eigen::MatrixBase<Derived1> & E_1,
        const Eigen::MatrixBase<Derived2> & E_2,
        const RotatorNormal<Dim> & rotator, const Real ratio)
        -> std::tuple<Equation_stress_t, Equation_stiffness_t, Real>;
    template <class Derived1, class Derived2>
    inline static auto delta_equation_stress_stiffness_eval_strain_1(
        const Function_t & mat_1_stress_eval,
        const Function_t & mat_2_stress_eval,
        const Eigen::MatrixBase<Derived1> & E_0,
        const Eigen::MatrixBase<Derived2> & E_1_rot,
        const RotatorNormal<Dim> & rotator, const Real ratio)
        -> std::tuple<Equation_stress_t, Equation_stiffness_t, Real>;
    /**
     * the following functions claculate the energy computation error of the
     * solution. it will be used in each step of the solution to determine the
     * relevant difference that implementation of that step has had on
     * convergence to the solution.
     */
    inline static auto del_energy_eval(const Real del_E_norm,
                                       const Real delta_S_norm) -> Real;
    /**
     *  These functions are used as intrface for combination functions, They are
     *also used for carrying out the stress transformation necessary for
     *combining stifness matrix in Finite-Strain formulation because the
     *combining formula from the bokk "Theory of Composites" are for symmetric
     *stifness matrices such as C and we have to transform stress to PK2 in
     *order to be able to use it
     */
    template <class Derived1, class Derived2>
    inline static auto
    lam_stress_combine(const Eigen::MatrixBase<Derived1> & stress_1,
                       const Eigen::MatrixBase<Derived2> & stress_2,
                       const Real ratio) -> Stress_t;
    inline static auto lam_stiffness_combine(
        const Eigen::Ref<Stiffness_t> & stiffness_1,
        const Eigen::Ref<Stiffness_t> & stiffness_2, const Real ratio,
        const Eigen::Ref<Strain_t> & F_1, const Eigen::Ref<Stress_t> & F_2,
        const Eigen::Ref<Strain_t> & P_1, const Eigen::Ref<Stress_t> & P_2,
        const Eigen::Ref<Strain_t> & F, const Eigen::Ref<Stress_t> & P)
        -> Stiffness_t;

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
    static auto laminate_solver(Eigen::Ref<Strain_t> strain_coord,
                                const Function_t & mat_1_stress_eval,
                                const Function_t & mat_2_stress_eval,
                                Real ratio,
                                const Eigen::Ref<Vec_t> & normal_vec,
                                Real tol = 1e-10, Dim_t max_iter = 1000)
        -> std::tuple<Dim_t, Real, Strain_t, Strain_t>;

    /* ---------------------------------------------------------------------- */
    static auto evaluate_stress(const Eigen::Ref<Strain_t> & strain_coord,
                                const Function_t & mat_1_stress_eval,
                                const Function_t & mat_2_stress_eval,
                                Real ratio,
                                const Eigen::Ref<Vec_t> & normal_vec,
                                Real tol = 1e-10, Dim_t max_iter = 1000)
        -> Stress_t;

    /* ---------------------------------------------------------------------- */
    static auto
    evaluate_stress_tangent(const Eigen::Ref<Strain_t> & strain_coord,
                            const Function_t & mat_1_stress_eval,
                            const Function_t & mat_2_stress_eval, Real ratio,
                            const Eigen::Ref<Vec_t> & normal_vec,
                            Real tol = 1e-10, Dim_t max_iter = 1000)
        -> std::tuple<Stress_t, Stiffness_t>;
  };  // LamHomogen
  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
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
    inline static auto lam_S_combine(const Eigen::MatrixBase<Derived1> & S_1,
                                     const Eigen::MatrixBase<Derived2> & S_2,
                                     const Real ratio) -> Stress_t;
    template <class Derived1, class Derived2>
    inline static auto lam_C_combine(const Eigen::MatrixBase<Derived1> & C_1,
                                     const Eigen::MatrixBase<Derived2> & C_2,
                                     const Real ratio) -> Stiffness_t;
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
  template <Dim_t Dim, Formulation Form>
  template <class Derived>
  auto LamHomogen<Dim, Form>::get_equation_stress(
      const Eigen::MatrixBase<Derived> & S_total) -> Equation_stress_t {
    return get_equation_strain(S_total);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim, Formulation Form>
  template <class Derived>
  auto LamHomogen<Dim, Form>::get_parallel_stress(
      const Eigen::MatrixBase<Derived> & S_total) -> Parallel_stress_t {
    return get_parallel_strain(S_total);
  }

  /* ---------------------------------------------------------------------- */

  template <Dim_t Dim, Formulation Form>
  template <class Derived>
  auto LamHomogen<Dim, Form>::get_parallel_strain(
      const Eigen::MatrixBase<Derived> & E_total) -> Parallel_strain_t {
    Parallel_strain_t E_parallel;
    auto parallel_indices{get_parallel_indices()};
    for (auto && tup : akantu::enumerate(parallel_indices)) {
      auto && index = std::get<1>(tup);
      auto counter = std::get<0>(tup);
      E_parallel(counter) = E_total(index[0], index[1]);
    }
    return E_parallel;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim, Formulation Form>
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
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* -----------The following functions are used to solve the-------------- */
  /* -----------the serial part of the laminate structure------------------ */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */

  // It does not need to be especialised because all functions used init are
  // especialised. this function computes the strain in the second
  // layer of laminate material from strain in the first layer and total
  // strain

  /* ---------------------------------------------------------------------- */
  // this function recieves the strain in general corrdinate and returns ths
  // delta_stress and the jacobian in rotated coordinates
  template <Dim_t Dim, Formulation Form>
  template <class Derived1, class Derived2>
  auto LamHomogen<Dim, Form>::delta_equation_stress_stiffness_eval(
      const Function_t & mat_1_stress_eval,
      const Function_t & mat_2_stress_eval,
      const Eigen::MatrixBase<Derived1> & strain_1,
      const Eigen::MatrixBase<Derived2> & strain_2,
      const RotatorNormal<Dim> & rotator, const Real ratio)
      -> std::tuple<Equation_stress_t, Equation_stiffness_t, Real> {
    auto stress_stiffness_1 = mat_1_stress_eval(strain_1);
    auto stress_stiffness_2 = mat_2_stress_eval(strain_2);

    /**
     *First we obtain the stress and stiffness matrix in the real coordiante
     *axis S₁(E₁) - S₂(E₂(E₁)) => jacobian of the second term is : ∂S₂/∂E₂ *
     *∂E₂/∂E₁ and we know form r * E₁ + (1-r) * E₂ = E_0  => ∂E₂/∂E₁ = -(r /
     *(1-r))
     */
    Stress_t del_stress_coord, del_stress_rot;
    del_stress_coord =
        std::get<0>(stress_stiffness_1) - std::get<0>(stress_stiffness_2);
    Stiffness_t del_stiffness_coord, del_stiffness_rot;
    del_stiffness_coord =
        (std::get<1>(stress_stiffness_1) -
         (-ratio / (1 - ratio)) * std::get<1>(stress_stiffness_2));

    // Then we rotate them into the rotated coordinate (whose x axis is
    // aligned with the noraml of the interface) so we can docompose them into
    // their serial and parallel components. (We only need the elements of
    // them that act in series in the structure of the laminate)
    del_stress_rot = rotator.rotate_back(del_stress_coord).eval();
    del_stiffness_rot = rotator.rotate_back(del_stiffness_coord).eval();

    Equation_stress_t del_stress_eq;
    del_stress_eq = get_equation_strain(del_stress_rot);
    Equation_stiffness_t del_stiffness_eq;
    del_stiffness_eq = get_equation_stiffness(del_stiffness_rot);

    auto del_stress_sum_norm{std::get<0>(stress_stiffness_1).norm() +
                             std::get<0>(stress_stiffness_2).norm()};

    return std::make_tuple(del_stress_eq, del_stiffness_eq,
                           del_stress_sum_norm);
  }  // namespace muSpectre

  /* ---------------------------------------------------------------------- */
  // linear equation are solved in the rotated coordiantes to obtain strain_2
  // form known strain_1 and then these two are passed to calculate
  // stress in the general corrdinates
  template <Dim_t Dim, Formulation Form>
  template <class Derived1, class Derived2>
  auto
  LamHomogen<Dim, Form>::linear_eqs(Real ratio,
                                    const Eigen::MatrixBase<Derived1> & E_0_eq,
                                    const Eigen::MatrixBase<Derived2> & E_1_eq)
      -> Equation_strain_t {
    return ((E_0_eq - ratio * E_1_eq) / (1 - ratio));
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim, Formulation Form>
  template <class Derived1, class Derived2>
  auto LamHomogen<Dim, Form>::delta_equation_stress_stiffness_eval_strain_1(
      const Function_t & mat_1_stress_eval,
      const Function_t & mat_2_stress_eval,
      const Eigen::MatrixBase<Derived1> & strain_0_rot,
      const Eigen::MatrixBase<Derived2> & strain_1_rot,
      const RotatorNormal<Dim> & rotator, Real ratio)
      -> std::tuple<Equation_stress_t, Equation_stiffness_t, Real> {
    // First we claculate strain_1 and strain_2 in rotated coordinates that we
    // have relations (parralel or serial)  between strains in two layers)

    Parallel_strain_t strain_0_par_rot = get_parallel_strain(strain_0_rot);

    Equation_strain_t strain_0_eq_rot = get_equation_strain(strain_0_rot);
    Equation_strain_t strain_1_eq_rot = get_equation_strain(strain_1_rot);
    Equation_strain_t strain_2_eq_rot =
        linear_eqs(ratio, strain_0_eq_rot, strain_1_eq_rot);

    Strain_t strain_2_rot =
        make_total_strain(strain_2_eq_rot, strain_0_par_rot);

    // Then we rotate them back into original coordiantes so we can use the
    // constituive laws for computing their stress
    Strain_t strain_1 = rotator.rotate(strain_1_rot);
    Strain_t strain_2 = rotator.rotate(strain_2_rot);

    // Knowing the strain in each layer we can compute the difference of the
    // stress in layers corrsponding to the strains of the layers
    return delta_equation_stress_stiffness_eval(mat_1_stress_eval,
                                                mat_2_stress_eval, strain_1,
                                                strain_2, rotator, ratio);
  }
  /* ---------------------------------------------------------------------- */

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim, Formulation Form>
  template <class Derived1, class Derived2>
  auto LamHomogen<Dim, Form>::make_total_strain(
      const Eigen::MatrixBase<Derived1> & E_eq,
      const Eigen::MatrixBase<Derived2> & E_para) -> Strain_t {
    Strain_t E_total;

    auto equation_indices{get_equation_indices()};
    auto parallel_indices{get_parallel_indices()};

    for (auto && tup : akantu::enumerate(equation_indices)) {
      auto && index = std::get<1>(tup);
      auto counter = std::get<0>(tup);
      E_total(index[0], index[1]) = E_eq(counter);
      if (Form == Formulation::small_strain) {
        E_total(index[1], index[0]) = E_eq(counter);
      }
    }
    for (auto && tup : akantu::enumerate(parallel_indices)) {
      auto && index = std::get<1>(tup);
      auto counter = std::get<0>(tup);
      E_total(index[0], index[1]) = E_para(counter);
    }
    return E_total;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim, Formulation Form>
  template <class Derived1, class Derived2>
  auto LamHomogen<Dim, Form>::make_total_stress(
      const Eigen::MatrixBase<Derived1> & S_eq,
      const Eigen::MatrixBase<Derived2> & S_para) -> Stress_t {
    return make_total_strain(S_eq, S_para);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim, Formulation Form>
  template <class Derived>
  auto LamHomogen<Dim, Form>::get_equation_stiffness(
      const Eigen::MatrixBase<Derived> & C) -> Equation_stiffness_t {
    Equation_stiffness_t C_equation{};
    auto equation_indices{get_equation_indices()};
    if (Form == Formulation::small_strain) {
      for (auto && tup_col : akantu::enumerate(equation_indices)) {
        auto && index_col = std::get<1>(tup_col);
        auto counter_col = std::get<0>(tup_col);
        for (auto && tup_row : akantu::enumerate(equation_indices)) {
          auto && index_row = std::get<1>(tup_row);
          auto counter_row = std::get<0>(tup_row);
          if (counter_row > 0) {
            C_equation(counter_col, counter_row) =
                muGrid::get(C, index_col[0], index_col[1], index_row[0],
                            index_row[1]) *
                2.0;
          } else {
            C_equation(counter_col, counter_row) = muGrid::get(
                C, index_col[0], index_col[1], index_row[0], index_row[1]);
          }
        }
      }
    } else {
      for (auto && tup_col : akantu::enumerate(equation_indices)) {
        auto && index_col = std::get<1>(tup_col);
        auto counter_col = std::get<0>(tup_col);
        for (auto && tup_row : akantu::enumerate(equation_indices)) {
          auto && index_row = std::get<1>(tup_row);
          auto counter_row = std::get<0>(tup_row);
          C_equation(counter_col, counter_row) = muGrid::get(
              C, index_col[0], index_col[1], index_row[0], index_row[1]);
        }
      }
    }

    return C_equation;
  }
  /* ---------------------------------------------------------------------- */

  /**
   *These functions are used as intrface for combination functions, They are
   *also used for carrying out the stress transformation necessary for
   *combining stifness matrix in Finite-Strain formulation because the
   *combining formula from the bokk "Theory of Composites" written by "Graeme
   *Miltonare" for symmetric stifness matrices such as C and we have to
   *transform stress to PK2 in order to be able to use it
   */

  template <Dim_t Dim, Formulation Form>
  template <class Derived1, class Derived2>
  auto LamHomogen<Dim, Form>::lam_stress_combine(
      const Eigen::MatrixBase<Derived1> & stress_1,
      const Eigen::MatrixBase<Derived2> & stress_2, const Real ratio)
      -> Stress_t {
    return LamCombination<Dim>::lam_S_combine(stress_1, stress_2, ratio);
  }

  template <Dim_t Dim, Formulation Form>
  auto LamHomogen<Dim, Form>::lam_stiffness_combine(
      const Eigen::Ref<Stiffness_t> & stiffness_1,
      const Eigen::Ref<Stiffness_t> & stiffness_2, const Real ratio,
      const Eigen::Ref<Strain_t> & F_1, const Eigen::Ref<Stress_t> & F_2,
      const Eigen::Ref<Strain_t> & P_1, const Eigen::Ref<Stress_t> & P_2,
      const Eigen::Ref<Strain_t> & F, const Eigen::Ref<Stress_t> &
      /*P*/) -> Stiffness_t {
    if (Form == Formulation::finite_strain) {
      auto S_C_1 =
          MatTB::PK2_stress<StressMeasure::PK1, StrainMeasure::Gradient>(
              F_1, P_1, stiffness_1);
      auto S_C_2 =
          MatTB::PK2_stress<StressMeasure::PK1, StrainMeasure::Gradient>(
              F_2, P_2, stiffness_2);
      const Stress_t S_1 = std::get<0>(S_C_1);
      const Stress_t S_2 = std::get<0>(S_C_2);
      Stress_t && S_effective =
          LamCombination<Dim>::lam_S_combine(S_1, S_2, ratio);

      const Stiffness_t C_1 = std::get<1>(S_C_1);
      const Stiffness_t C_2 = std::get<1>(S_C_2);

      Stiffness_t C_effecive =
          LamCombination<Dim>::lam_C_combine(C_1, C_2, ratio);

      auto P_K_effective =
          MatTB::PK1_stress<StressMeasure::PK2, StrainMeasure::GreenLagrange>(
              F, S_effective, C_effecive);

      Stiffness_t K_effective = std::get<1>(P_K_effective);

      return K_effective;
    } else {
      return LamCombination<Dim>::lam_C_combine(stiffness_1, stiffness_2,
                                                ratio);
    }
  }
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* -----------The following functions are used to make the -------------- */
  /* --------resultant laminate stifness/strees form that of the layters----*/
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  template <>
  template <class Derived1, class Derived2>
  auto
  LamCombination<twoD>::lam_C_combine(const Eigen::MatrixBase<Derived1> & C_1,
                                      const Eigen::MatrixBase<Derived2> & C_2,
                                      const Real ratio) -> Stiffness_t {
    using Mat_A_t = Eigen::Matrix<Real, twoD, twoD>;
    using Vec_A_t = Eigen::Matrix<Real, twoD, 1>;
    using Vec_AT_t = Eigen::Matrix<Real, 1, twoD>;

    std::array<double, 3> cf = {1.0, sqrt(2.0), 2.0};
    auto get_A11 = [cf](const Eigen::Ref<const Stiffness_t> & C) {
      Mat_A_t A11 = Mat_A_t::Zero();
      A11 << cf[0] * muGrid::get(C, 0, 0, 0, 0),
          cf[1] * muGrid::get(C, 0, 0, 0, 1),
          cf[1] * muGrid::get(C, 0, 0, 0, 1),
          cf[2] * muGrid::get(C, 0, 1, 0, 1);
      return A11;
    };
    Mat_A_t A11c;
    A11c << cf[0], cf[1], cf[1], cf[2];
    auto get_A12 = [cf](const Eigen::Ref<const Stiffness_t> & C) {
      Vec_A_t A12 = Vec_A_t::Zero();
      A12 << cf[0] * muGrid::get(C, 0, 0, 1, 1),
          cf[1] * muGrid::get(C, 1, 1, 0, 1);
      return A12;
    };
    Vec_A_t A12c = Vec_A_t::Zero();
    A12c << cf[0], cf[1];

    Mat_A_t A11_1{get_A11(C_1)};
    Mat_A_t A11_2{get_A11(C_2)};

    Vec_A_t A12_1 = {get_A12(C_1)};
    Vec_AT_t A21_1 = A12_1.transpose();
    Vec_A_t A12_2 = {get_A12(C_2)};
    Vec_AT_t A21_2 = A12_2.transpose();

    Real A22_1 = muGrid::get(C_1, 1, 1, 1, 1);
    Real A22_2 = muGrid::get(C_2, 1, 1, 1, 1);

    auto get_inverse_average = [&ratio](const Eigen::Ref<Mat_A_t> & matrix_1,
                                        const Eigen::Ref<Mat_A_t> & matrix_2) {
      return ((ratio * matrix_1.inverse() + (1 - ratio) * matrix_2.inverse())
                  .inverse());
    };
    auto get_average = [&ratio](Real A_1, Real A_2) {
      return ratio * A_1 + (1 - ratio) * A_2;
    };

    auto get_average_vec = [&ratio](Vec_A_t A_1, Vec_A_t A_2) {
      return ratio * A_1 + (1 - ratio) * A_2;
    };

    auto get_average_vecT = [&ratio](Vec_AT_t A_1, Vec_AT_t A_2) {
      return ratio * A_1 + (1 - ratio) * A_2;
    };

    // calculating average of A matrices  of the materials
    Mat_A_t A11 = get_inverse_average(A11_1, A11_2);
    Vec_A_t A12 =
        A11 * get_average_vec(A11_1.inverse() * A12_1, A11_2.inverse() * A12_2);
    Real A22 =
        get_average(A22_1 - A21_1 * A11_1.inverse() * A12_1,
                    A22_2 - A21_2 * A11_2.inverse() * A12_2) +
        get_average_vecT(A21_1 * A11_1.inverse(), A21_2 * A11_2.inverse()) *
            A11 *
            get_average_vec(A11_1.inverse() * A12_1, A11_2.inverse() * A12_2);

    std::vector<Real> c_maker_inp = {
        A11(0, 0) / A11c(0, 0), A12(0, 0) / A12c(0, 0),
        A11(0, 1) / A11c(0, 1), A22,
        A12(1, 0) / A12c(1, 0), A11(1, 1) / A11c(1, 1)};
    // now the resultant stiffness is calculated fro 6 elements obtained from
    // A matrices averaging routine:
    Stiffness_t ret_C = Stiffness_t::Zero();
    ret_C = MaterialLinearAnisotropic<twoD, twoD>::c_maker(c_maker_inp);
    return ret_C;
  }
  /*------------------------------------------------------------------*/
  template <>
  template <class Derived1, class Derived2>
  auto
  LamCombination<threeD>::lam_C_combine(const Eigen::MatrixBase<Derived1> & C_1,
                                        const Eigen::MatrixBase<Derived2> & C_2,
                                        const Real ratio) -> Stiffness_t {
    // the combination method is obtained form P. 163 of
    // "Theory of Composites"
    //  Author : Milton_G_W

    // constructing "A" matrices( A11, A12, A21,  A22) according to the
    // procedure from the book:

    // this type of matrix will be used in calculating the combinatio of the
    // Stiffness matrixes
    // using Mat_A_t = Eigen::Matrix<Real, threeD, threeD, Eigen::ColMajor>;
    using Mat_A_t = Eigen::Matrix<Real, threeD, threeD>;

    // these coeffs are used in constructing matrices "A" from matrices "C"
    // and vice versa.
    std::array<double, 3> cf = {1.0, sqrt(2.0), 2.0};

    // These functions make "A" matrices from "C" matrix
    auto get_A11 = [cf](const Eigen::Ref<const Stiffness_t> & C) {
      Mat_A_t A11 = Mat_A_t::Zero();
      A11 << cf[0] * muGrid::get(C, 0, 0, 0, 0),
          cf[1] * muGrid::get(C, 0, 0, 0, 2),
          cf[1] * muGrid::get(C, 0, 0, 0, 1),
          cf[1] * muGrid::get(C, 0, 0, 0, 2),
          cf[2] * muGrid::get(C, 0, 2, 0, 2),
          cf[2] * muGrid::get(C, 0, 2, 0, 1),
          cf[1] * muGrid::get(C, 0, 0, 0, 1),
          cf[2] * muGrid::get(C, 0, 2, 0, 1),
          cf[2] * muGrid::get(C, 0, 1, 0, 1);
      return A11;
    };

    auto get_A12 = [cf](const Eigen::Ref<const Stiffness_t> & C) {
      Mat_A_t A12 = Mat_A_t::Zero();
      A12 << cf[0] * muGrid::get(C, 0, 0, 1, 1),
          cf[0] * muGrid::get(C, 0, 0, 2, 2),
          cf[1] * muGrid::get(C, 0, 0, 1, 2),
          cf[1] * muGrid::get(C, 1, 1, 0, 2),
          cf[1] * muGrid::get(C, 2, 2, 0, 2),
          cf[2] * muGrid::get(C, 1, 2, 0, 2),
          cf[1] * muGrid::get(C, 1, 1, 0, 1),
          cf[1] * muGrid::get(C, 2, 2, 0, 1),
          cf[2] * muGrid::get(C, 1, 2, 0, 1);
      return A12;
    };

    auto get_A22 = [cf](const Eigen::Ref<const Stiffness_t> & C) {
      Mat_A_t A22 = Mat_A_t::Zero();
      A22 << cf[0] * muGrid::get(C, 1, 1, 1, 1),
          cf[0] * muGrid::get(C, 1, 1, 2, 2),
          cf[1] * muGrid::get(C, 1, 1, 2, 1),
          cf[0] * muGrid::get(C, 1, 1, 2, 2),
          cf[0] * muGrid::get(C, 2, 2, 2, 2),
          cf[1] * muGrid::get(C, 2, 2, 1, 2),
          cf[1] * muGrid::get(C, 1, 1, 1, 2),
          cf[1] * muGrid::get(C, 2, 2, 1, 2),
          cf[2] * muGrid::get(C, 2, 1, 2, 1);
      return A22;
    };

    // Here we use the functions defined above to obtain matrices "A"
    Mat_A_t A11_1{get_A11(C_1)};
    Mat_A_t A11_2{get_A11(C_2)};

    Mat_A_t A12_1{get_A12(C_1)};
    Mat_A_t A21_1{A12_1.transpose()};
    Mat_A_t A12_2{get_A12(C_2)};
    Mat_A_t A21_2{A12_2.transpose()};

    Mat_A_t A22_1{get_A22(C_1)};
    Mat_A_t A22_2{get_A22(C_2)};

    // this matrices consists of coeeffs that are used in extraction of "A"s
    Mat_A_t A11c;
    A11c << cf[0], cf[1], cf[1], cf[1], cf[2], cf[2], cf[1], cf[2], cf[2];
    Mat_A_t A12c;
    A12c << cf[0], cf[0], cf[1], cf[1], cf[1], cf[2], cf[1], cf[1], cf[2];
    Mat_A_t A22c;
    A22c << cf[0], cf[0], cf[1], cf[0], cf[0], cf[1], cf[1], cf[1], cf[2];

    // these two functions are routines to compute average of "A" matrices
    auto get_inverse_average = [&ratio](const Eigen::Ref<Mat_A_t> & matrix_1,
                                        const Eigen::Ref<Mat_A_t> & matrix_2) {
      return ((ratio * matrix_1.inverse() + (1 - ratio) * matrix_2.inverse())
                  .inverse());
    };

    auto get_average = [&ratio](const Mat_A_t & matrix_1,
                                const Mat_A_t & matrix_2) {
      return (ratio * matrix_1 + (1 - ratio) * matrix_2);
    };

    // calculating average of A matrices  of the materials according to the
    // book Formulation (9.8) in the book
    Mat_A_t A11 = get_inverse_average(A11_1, A11_2);
    Mat_A_t A12 =
        A11 * get_average(A11_1.inverse() * A12_1, A11_2.inverse() * A12_2);
    Mat_A_t A22 =
        get_average(A22_1 - A21_1 * A11_1.inverse() * A12_1,
                    A22_2 - A21_2 * A11_2.inverse() * A12_2) +
        get_average(A21_1 * A11_1.inverse(), A21_2 * A11_2.inverse()) * A12;

    std::vector<Real> c_maker_inp = {
        A11(0, 0) / A11c(0, 0), A12(0, 0) / A12c(0, 0), A12(0, 1) / A12c(0, 1),
        A12(0, 2) / A12c(0, 2), A11(1, 0) / A11c(1, 0), A11(0, 2) / A11c(0, 2),

        A22(0, 0) / A22c(0, 0), A22(1, 0) / A22c(1, 0), A22(2, 0) / A22c(2, 0),
        A12(1, 0) / A12c(1, 0), A12(2, 0) / A12c(2, 0),

        A22(1, 1) / A22c(1, 1), A22(2, 1) / A22c(2, 1), A12(1, 1) / A12c(1, 1),
        A12(2, 1) / A12c(2, 1),

        A22(2, 2) / A22c(2, 2), A12(1, 2) / A12c(1, 2), A12(2, 2) / A12c(2, 2),

        A11(1, 1) / A11c(1, 1), A11(2, 1) / A11c(2, 1),

        A11(2, 2) / A11c(2, 2)};
    // now the resultant stiffness is calculated for 21 elements obtained from
    // "A" matrices averaging routine :
    Stiffness_t ret_C = Stiffness_t::Zero();
    ret_C = MaterialLinearAnisotropic<threeD, threeD>::c_maker(c_maker_inp);
    return ret_C;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  template <class Derived1, class Derived2>
  auto
  LamCombination<Dim>::lam_S_combine(const Eigen::MatrixBase<Derived1> & S_1,
                                     const Eigen::MatrixBase<Derived2> & S_2,
                                     const Real ratio) -> Stress_t {
    auto ret_S = ratio * S_1 + (1 - ratio) * S_2;
    return ret_S;
  }
}  // namespace muSpectre

#endif  // SRC_MATERIALS_LAMINATE_HOMOGENISATION_HH_
