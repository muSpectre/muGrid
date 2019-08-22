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
 * µSpectreis free software; you can redistribute it and/or
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
  template <Dim_t Dim, Formulation Form>
  auto LamHomogen<Dim, Form>::laminate_solver(
      Eigen::Ref<Strain_t> strain_coord, const Function_t & mat_1_stress_eval,
      const Function_t & mat_2_stress_eval, Real ratio,
      const Eigen::Ref<Vec_t> & normal_vec, Real tol, Dim_t max_iter)
      -> std::tuple<Dim_t, Real, Strain_t, Strain_t> {
    /*
     * here we rotate the strain such that the laminate intersection normal
     * would align with the x-axis. strain_lam is the total strain in the new
     * coordinates.
     */
    Real del_energy;
    Dim_t iter{0};
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
      throw std::runtime_error(
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
  template <Dim_t Dim, Formulation Form>
  auto LamHomogen<Dim, Form>::evaluate_stress(
      const Eigen::Ref<Strain_t> & strain_coord,
      const Function_t & mat_1_stress_eval,
      const Function_t & mat_2_stress_eval, Real ratio,
      const Eigen::Ref<Vec_t> & normal_vec, Real tol, Dim_t max_iter)
      -> Stress_t {
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
  template <Dim_t Dim, Formulation Form>
  auto LamHomogen<Dim, Form>::evaluate_stress_tangent(
      const Eigen::Ref<Strain_t> & strain_coord,
      const Function_t & mat_1_stress_eval,
      const Function_t & mat_2_stress_eval, Real ratio,
      const Eigen::Ref<Vec_t> & normal_vec, Real tol, Dim_t max_iter)
      -> std::tuple<Stress_t, Stiffness_t> {
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
  template class LamHomogen<twoD, Formulation::finite_strain>;
  template class LamHomogen<threeD, Formulation::finite_strain>;
  template class LamHomogen<twoD, Formulation::small_strain>;
  template class LamHomogen<threeD, Formulation::small_strain>;

}  // namespace muSpectre
