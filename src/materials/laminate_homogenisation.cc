/**
 * @file   laminate_homogenisation.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   28 Sep 2018
 *
 * @brief : Implementation of functions of internal laminate solver used in
 * MaterialLaminate
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

#include "laminate_homogenisation.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* -----------The following function contains the solution loop  -------- */
  /* ----it recieves the materials normal vector of interface and strain--- */
  /* --it returns the equilvalent resultant stress and stiffness matrices-- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, Formulation Form>
  auto LamHomogen<Dim, Form>::laminate_solver(
      const Eigen::Ref<Strain_t> & strain_coord,
      const Function_t & mat_1_stress_eval,
      const Function_t & mat_2_stress_eval, const Real & ratio,
      const Eigen::Ref<Vec_t> & normal_vec, const Real & tol,
      const Index_t & max_iter)
      -> std::tuple<Index_t, Real, Strain_t, Strain_t> {
    /*
     * here we rotate the strain such that the laminate intersection normal
     * would align with the x-axis. strain_lam is the total strain in the new
     * coordinates.
     */
    Real del_energy;
    Index_t iter{0};
    RotatorNormal<Dim> rotator(normal_vec);

    Strain_t strain_1{strain_coord}, strain_2{strain_coord};
    Strain_t strain_0_rot{rotator.rotate_back(strain_coord)};
    Strain_t strain_1_rot = strain_0_rot;

    Parallel_strain_t strain_0_para{get_parallel_strain(strain_0_rot)};
    Equation_strain_t strain_0_eq{get_equation_strain(strain_0_rot)};
    Equation_strain_t strain_1_new_eq{strain_0_eq};
    Equation_strain_t strain_1_eq{strain_0_eq};

    Equation_stress_t delta_stress_eq{};
    Equation_stiffness_t delta_stiffness_eq{};
    std::tuple<Equation_stress_t, Equation_stiffness_t, Real>
        delta_stress_stiffness_eq;

    Stress_t stress_1{}, stress_2{}, ret_stress{};
    Stiffness_t stiffness_1{}, stiffness_2{}, ret_stiffness{};
    std::tuple<Stress_t, Stiffness_t> stress_stiffness_1, stress_stiffness_2;

    /* serial (stress_1_rot - stress_2_rot) (as function of strain_1_rot)
     * and its Jacobian are calculated here in the rotated_back coordiantes*/
    delta_stress_stiffness_eq = delta_equation_stress_stiffness_eval_strain_1(
        mat_1_stress_eval, mat_2_stress_eval, strain_0_rot, strain_1_rot,
        rotator, ratio);

    delta_stress_eq = std::get<0>(delta_stress_stiffness_eq);
    delta_stiffness_eq = std::get<1>(delta_stress_stiffness_eq);

    // solution loop:
    do {
      // updating variables:
      strain_1_eq = strain_1_new_eq;

      // solving for ∇(δC) * [x_n+1 - x_n] = -δS
      strain_1_new_eq =
          strain_1_eq - delta_stiffness_eq.inverse() * delta_stress_eq;

      // updating strain_1 before stress and stiffness evaluation
      strain_1_rot = make_total_strain(strain_1_new_eq, strain_0_para);

      // stress and stiffenss evaluation in the rotated coordinates
      delta_stress_stiffness_eq = delta_equation_stress_stiffness_eval_strain_1(
          mat_1_stress_eval, mat_2_stress_eval, strain_0_rot, strain_1_rot,
          rotator, ratio);

      delta_stress_eq = std::get<0>(delta_stress_stiffness_eq);
      delta_stiffness_eq = std::get<1>(delta_stress_stiffness_eq);
      // computing variable used for loop termination criterion
      // energy norm of the residual
      del_energy =
          std::abs(delta_stress_eq.dot(strain_1_new_eq - strain_1_eq)) /
          std::get<2>(delta_stress_stiffness_eq);
      iter++;
    } while (del_energy > tol && iter < max_iter);

    // check if the loop has lead in convergence or not:
    if (iter == max_iter) {
      throw muGrid::RuntimeError(
          "Error: The laminate solver has not converged!!!!");
    }

    // computing stress and stiffness of each layer according to it's
    // correspondent strain calculated in the loop

    strain_1_rot = make_total_strain(strain_1_new_eq, strain_0_para);

    // storing the resultant strain in each layer in strain_1 and strain_2
    auto strain_2_new_eq = linear_eqs(ratio, strain_0_eq, strain_1_new_eq);
    auto strain_2_rot = make_total_strain(strain_2_new_eq, strain_0_para);

    Strain_t strain_1_coord{rotator.rotate(strain_1_rot)};
    Strain_t strain_2_coord{rotator.rotate(strain_2_rot)};

    return std::make_tuple(iter, del_energy, strain_1_coord, strain_2_coord);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, Formulation Form>
  auto LamHomogen<Dim, Form>::evaluate_stress(
      const Eigen::Ref<Strain_t> & strain_coord,
      const Function_t & mat_1_stress_eval,
      const Function_t & mat_2_stress_eval, const Real & ratio,
      const Eigen::Ref<Vec_t> & normal_vec, const Real & tol,
      const Index_t & max_iter) -> Stress_t {
    RotatorNormal<Dim> rotator(normal_vec);
    // Using laminate solve to find out the strains in each layer of the
    // lamiante
    auto homogenized =
        laminate_solver(strain_coord, mat_1_stress_eval, mat_2_stress_eval,
                        ratio, normal_vec, tol, max_iter);
    Strain_t strain_1_coord = std::get<2>(homogenized);
    Strain_t strain_2_coord = std::get<3>(homogenized);

    auto stress_stiffness_1 = mat_1_stress_eval(strain_1_coord);
    auto stress_stiffness_2 = mat_2_stress_eval(strain_2_coord);

    Stress_t stress_1 = std::get<0>(stress_stiffness_1);
    stress_1 = rotator.rotate_back(stress_1.eval());
    Stress_t stress_2 = std::get<0>(stress_stiffness_2);
    stress_2 = rotator.rotate_back(stress_2.eval());

    Stress_t ret_stress = lam_stress_combine(stress_1, stress_2, ratio);
    ret_stress = rotator.rotate(ret_stress.eval());
    return ret_stress;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, Formulation Form>
  auto LamHomogen<Dim, Form>::evaluate_stress_tangent(
      const Eigen::Ref<Strain_t> & strain_coord,
      const Function_t & mat_1_stress_eval,
      const Function_t & mat_2_stress_eval, const Real & ratio,
      const Eigen::Ref<Vec_t> & normal_vec, const Real & tol,
      const Index_t & max_iter) -> std::tuple<Stress_t, Stiffness_t> {
    RotatorNormal<Dim> rotator(normal_vec);
    // Using laminate solve to find out the strains in each layer of the
    // lamiante
    auto homogenized =
        laminate_solver(strain_coord, mat_1_stress_eval, mat_2_stress_eval,
                        ratio, normal_vec, tol, max_iter);
    Strain_t strain_1_coord = std::get<2>(homogenized);
    Strain_t strain_2_coord = std::get<3>(homogenized);

    // rotating strains to have them in the rotated coordinates
    Strain_t strain_rot = rotator.rotate_back(strain_coord);
    Strain_t strain_1_rot = rotator.rotate_back(strain_1_coord);
    Strain_t strain_2_rot = rotator.rotate_back(strain_2_coord);

    auto stress_stiffness_1 = mat_1_stress_eval(strain_1_coord);
    auto stress_stiffness_2 = mat_2_stress_eval(strain_2_coord);
    // Here we rotate them so we can combine them according to the formulation
    // that we have in laminate coordiantes
    Stress_t stress_1 = std::get<0>(stress_stiffness_1);
    stress_1 = rotator.rotate_back(stress_1.eval());
    Stress_t stress_2 = std::get<0>(stress_stiffness_2);
    stress_2 = rotator.rotate_back(stress_2.eval());
    Stiffness_t stiffness_1 = std::get<1>(stress_stiffness_1);
    stiffness_1 = rotator.rotate_back(stiffness_1.eval());
    Stiffness_t stiffness_2 = std::get<1>(stress_stiffness_2);
    stiffness_2 = rotator.rotate_back(stiffness_2.eval());
    // combine the computed strains and tangents to have the resultant
    // stress and tangent of the pixel
    Stress_t ret_stress = lam_stress_combine(stress_1, stress_2, ratio);
    Stiffness_t ret_stiffness = lam_stiffness_combine(
        stiffness_1, stiffness_2, ratio, strain_1_rot, strain_2_rot, stress_1,
        stress_2, strain_rot, ret_stress);
    // Then we have to rotate them back to real coordiante system
    ret_stress = rotator.rotate(ret_stress.eval());
    ret_stiffness = rotator.rotate(ret_stiffness.eval());
    return std::make_tuple(ret_stress, ret_stiffness);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, Formulation Form>
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
  template <Index_t Dim, Formulation Form>
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
  template <Index_t Dim, Formulation Form>
  template <class Derived1, class Derived2>
  auto LamHomogen<Dim, Form>::delta_equation_stress_stiffness_eval(
      const Function_t & mat_1_stress_eval,
      const Function_t & mat_2_stress_eval,
      const Eigen::MatrixBase<Derived1> & strain_1,
      const Eigen::MatrixBase<Derived2> & strain_2,
      const RotatorNormal<Dim> & rotator, const Real & ratio)
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
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, Formulation Form>
  template <class Derived1, class Derived2>
  auto LamHomogen<Dim, Form>::delta_equation_stress_stiffness_eval_strain_1(
      const Function_t & mat_1_stress_eval,
      const Function_t & mat_2_stress_eval,
      const Eigen::MatrixBase<Derived1> & strain_0_rot,
      const Eigen::MatrixBase<Derived2> & strain_1_rot,
      const RotatorNormal<Dim> & rotator, const Real & ratio)
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

  /**
   *These functions are used as intrface for combination functions, They are
   *also used for carrying out the stress transformation necessary for
   *combining stifness matrix in Finite-Strain formulation because the
   *combining formula from the bokk "Theory of Composites" written by "Graeme
   *Miltonare" for symmetric stifness matrices such as C and we have to
   *transform stress to PK2 in order to be able to use it
   */

  template <Index_t Dim, Formulation Form>
  template <class Derived1, class Derived2>
  auto LamHomogen<Dim, Form>::lam_stress_combine(
      const Eigen::MatrixBase<Derived1> & stress_1,
      const Eigen::MatrixBase<Derived2> & stress_2, const Real & ratio)
      -> Stress_t {
    return LamCombination<Dim>::lam_S_combine(stress_1, stress_2, ratio);
  }

  template <Index_t Dim, Formulation Form>
  auto LamHomogen<Dim, Form>::lam_stiffness_combine(
      const Eigen::Ref<Stiffness_t> & stiffness_1,
      const Eigen::Ref<Stiffness_t> & stiffness_2, const Real & ratio,
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
                                      const Real & ratio) -> Stiffness_t {
    using Mat_A_t = Eigen::Matrix<Real, twoD, twoD>;
    using Vec_A_t = Eigen::Matrix<Real, twoD, 1>;
    using Vec_AT_t = Eigen::Matrix<Real, 1, twoD>;

    std::array<double, 3> cf = {1.0, sqrt(2.0), 2.0};
    auto get_A11{[&cf](const Eigen::Ref<const Stiffness_t> & C) {
      Mat_A_t A11{(Mat_A_t() << cf[0] * muGrid::get(C, 0, 0, 0, 0),
                   cf[1] * muGrid::get(C, 0, 0, 0, 1),
                   cf[1] * muGrid::get(C, 0, 0, 0, 1),
                   cf[2] * muGrid::get(C, 0, 1, 0, 1))
                      .finished()};

      return A11;
    }};
    const Mat_A_t A11c{(Mat_A_t() << cf[0], cf[1], cf[1], cf[2]).finished()};
    auto get_A12{[&cf](const Eigen::Ref<const Stiffness_t> & C) {
      Vec_A_t A12{(Vec_A_t() << cf[0] * muGrid::get(C, 0, 0, 1, 1),
                   cf[1] * muGrid::get(C, 1, 1, 0, 1))
                      .finished()};
      return A12;
    }};

    Vec_A_t A12c{(Vec_A_t() << cf[0], cf[1]).finished()};

    Mat_A_t A11_1{get_A11(C_1)};
    Mat_A_t A11_2{get_A11(C_2)};

    Vec_A_t A12_1{get_A12(C_1)};
    Vec_AT_t A21_1{A12_1.transpose()};
    Vec_A_t A12_2{get_A12(C_2)};
    Vec_AT_t A21_2{A12_2.transpose()};

    Real A22_1{muGrid::get(C_1, 1, 1, 1, 1)};
    Real A22_2{muGrid::get(C_2, 1, 1, 1, 1)};

    auto get_inverse_average{[&ratio](const Eigen::Ref<Mat_A_t> & matrix_1,
                                      const Eigen::Ref<Mat_A_t> & matrix_2) {
      return ((ratio * matrix_1.inverse() + (1 - ratio) * matrix_2.inverse())
                  .inverse());
    }};
    auto get_average{[&ratio](Real A_1, Real A_2) {
      return ratio * A_1 + (1 - ratio) * A_2;
    }};

    auto get_average_vec{[&ratio](Vec_A_t A_1, Vec_A_t A_2) {
      return ratio * A_1 + (1 - ratio) * A_2;
    }};

    auto get_average_vecT{[&ratio](Vec_AT_t A_1, Vec_AT_t A_2) {
      return ratio * A_1 + (1 - ratio) * A_2;
    }};

    // calculating average of A matrices  of the materials
    Mat_A_t A11{get_inverse_average(A11_1, A11_2)};
    Vec_A_t A12{A11 * get_average_vec(A11_1.inverse() * A12_1,
                                      A11_2.inverse() * A12_2)};
    Real A22{
        get_average(A22_1 - A21_1 * A11_1.inverse() * A12_1,
                    A22_2 - A21_2 * A11_2.inverse() * A12_2) +
        get_average_vecT(A21_1 * A11_1.inverse(), A21_2 * A11_2.inverse()) *
            A11 *
            get_average_vec(A11_1.inverse() * A12_1, A11_2.inverse() * A12_2)};

    std::vector<Real> c_maker_inp{
        {A11(0, 0) / A11c(0, 0), A12(0, 0) / A12c(0, 0), A11(0, 1) / A11c(0, 1),
         A22, A12(1, 0) / A12c(1, 0), A11(1, 1) / A11c(1, 1)}};
    // now the resultant stiffness is calculated fro 6
    // elements obtained from A matrices averaging routine:
    Stiffness_t ret_C{MaterialLinearAnisotropic<twoD>::c_maker(c_maker_inp)};
    return ret_C;
  }

  /*------------------------------------------------------------------*/
  template <>
  template <class Derived1, class Derived2>
  auto
  LamCombination<threeD>::lam_C_combine(const Eigen::MatrixBase<Derived1> & C_1,
                                        const Eigen::MatrixBase<Derived2> & C_2,
                                        const Real & ratio) -> Stiffness_t {
    // the combination method is obtained form P. 163 of
    // "Theory of Composites"
    //  Author : Milton_G_W

    // constructing "A" matrices( A11, A12, A21,  A22)
    // according to the procedure from the book:

    // this type of matrix will be used in calculating
    // the combinatio of the Stiffness matrixes using
    // Mat_A_t = Eigen::Matrix<Real, threeD, threeD,
    // Eigen::ColMajor>;
    using Mat_A_t = Eigen::Matrix<Real, threeD, threeD>;

    // these coeffs are used in constructing matrices
    // "A" from matrices "C" and vice versa.
    std::array<double, 3> cf{1.0, sqrt(2.0), 2.0};

    // These functions make "A" matrices from "C"
    // matrix
    auto get_A11{[&cf](const Eigen::Ref<const Stiffness_t> & C) {
      Mat_A_t A11{(Mat_A_t() << cf[0] * muGrid::get(C, 0, 0, 0, 0),
                   cf[1] * muGrid::get(C, 0, 0, 0, 2),
                   cf[1] * muGrid::get(C, 0, 0, 0, 1),
                   cf[1] * muGrid::get(C, 0, 0, 0, 2),
                   cf[2] * muGrid::get(C, 0, 2, 0, 2),
                   cf[2] * muGrid::get(C, 0, 2, 0, 1),
                   cf[1] * muGrid::get(C, 0, 0, 0, 1),
                   cf[2] * muGrid::get(C, 0, 2, 0, 1),
                   cf[2] * muGrid::get(C, 0, 1, 0, 1))
                      .finished()};
      return A11;
    }};

    auto get_A12{[&cf](const Eigen::Ref<const Stiffness_t> & C) {
      Mat_A_t A12{(Mat_A_t() << cf[0] * muGrid::get(C, 0, 0, 1, 1),
                   cf[0] * muGrid::get(C, 0, 0, 2, 2),
                   cf[1] * muGrid::get(C, 0, 0, 1, 2),
                   cf[1] * muGrid::get(C, 1, 1, 0, 2),
                   cf[1] * muGrid::get(C, 2, 2, 0, 2),
                   cf[2] * muGrid::get(C, 1, 2, 0, 2),
                   cf[1] * muGrid::get(C, 1, 1, 0, 1),
                   cf[1] * muGrid::get(C, 2, 2, 0, 1),
                   cf[2] * muGrid::get(C, 1, 2, 0, 1))
                      .finished()};
      return A12;
    }};

    auto get_A22{[&cf](const Eigen::Ref<const Stiffness_t> & C) {
      Mat_A_t A22{(Mat_A_t() << cf[0] * muGrid::get(C, 1, 1, 1, 1),
                   cf[0] * muGrid::get(C, 1, 1, 2, 2),
                   cf[1] * muGrid::get(C, 1, 1, 2, 1),
                   cf[0] * muGrid::get(C, 1, 1, 2, 2),
                   cf[0] * muGrid::get(C, 2, 2, 2, 2),
                   cf[1] * muGrid::get(C, 2, 2, 1, 2),
                   cf[1] * muGrid::get(C, 1, 1, 1, 2),
                   cf[1] * muGrid::get(C, 2, 2, 1, 2),
                   cf[2] * muGrid::get(C, 2, 1, 2, 1))
                      .finished()};

      return A22;
    }};

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
    Mat_A_t A11c{(Mat_A_t() << cf[0], cf[1], cf[1], cf[1], cf[2], cf[2], cf[1],
                  cf[2], cf[2])
                     .finished()};
    Mat_A_t A12c{(Mat_A_t() << cf[0], cf[0], cf[1], cf[1], cf[1], cf[2], cf[1],
                  cf[1], cf[2])
                     .finished()};
    Mat_A_t A22c{(Mat_A_t() << cf[0], cf[0], cf[1], cf[0], cf[0], cf[1], cf[1],
                  cf[1], cf[2])
                     .finished()};

    // these two functions are routines to compute average of "A" matrices
    auto get_inverse_average{[&ratio](const Eigen::Ref<Mat_A_t> & matrix_1,
                                      const Eigen::Ref<Mat_A_t> & matrix_2) {
      return ((ratio * matrix_1.inverse() + (1 - ratio) * matrix_2.inverse())
                  .inverse());
    }};

    auto get_average{
        [&ratio](const Mat_A_t & matrix_1, const Mat_A_t & matrix_2) {
          return (ratio * matrix_1 + (1 - ratio) * matrix_2);
        }};

    // calculating average of A matrices  of the materials according to the
    // book Formulation (9.8) in the book
    Mat_A_t A11{get_inverse_average(A11_1, A11_2)};
    Mat_A_t A12{A11 *
                get_average(A11_1.inverse() * A12_1, A11_2.inverse() * A12_2)};
    Mat_A_t A22{get_average(A22_1 - A21_1 * A11_1.inverse() * A12_1,
                            A22_2 - A21_2 * A11_2.inverse() * A12_2) +
                get_average(A21_1 * A11_1.inverse(), A21_2 * A11_2.inverse()) *
                    A12};

    std::vector<Real> c_maker_inp{
        {A11(0, 0) / A11c(0, 0), A12(0, 0) / A12c(0, 0), A12(0, 1) / A12c(0, 1),
         A12(0, 2) / A12c(0, 2), A11(1, 0) / A11c(1, 0), A11(0, 2) / A11c(0, 2),

         A22(0, 0) / A22c(0, 0), A22(1, 0) / A22c(1, 0), A22(2, 0) / A22c(2, 0),
         A12(1, 0) / A12c(1, 0), A12(2, 0) / A12c(2, 0),

         A22(1, 1) / A22c(1, 1), A22(2, 1) / A22c(2, 1), A12(1, 1) / A12c(1, 1),
         A12(2, 1) / A12c(2, 1),

         A22(2, 2) / A22c(2, 2), A12(1, 2) / A12c(1, 2), A12(2, 2) / A12c(2, 2),

         A11(1, 1) / A11c(1, 1), A11(2, 1) / A11c(2, 1),

         A11(2, 2) / A11c(2, 2)}};
    // now the resultant stiffness is calculated for 21 elements
    // obtained from "A" matrices averaging routine :
    Stiffness_t ret_C{MaterialLinearAnisotropic<threeD>::c_maker(c_maker_inp)};
    return ret_C;
  }

  /* ---------------------------------------------------------------------- */
  template class LamHomogen<twoD, Formulation::finite_strain>;
  template class LamHomogen<threeD, Formulation::finite_strain>;
  template class LamHomogen<twoD, Formulation::small_strain>;
  template class LamHomogen<threeD, Formulation::small_strain>;
  template class LamHomogen<twoD, Formulation::native>;
  template class LamHomogen<threeD, Formulation::native>;

  template class LamCombination<twoD>;
  template class LamCombination<threeD>;

}  // namespace muSpectre
