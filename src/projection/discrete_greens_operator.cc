/**
 * @file   discrete_greens_operator.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   09 Jul 2020
 *
 * @brief  implementation for inverse circulant operator
 *
 * Copyright © 2020 Till Junge, Martin Ladecký
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

#include "discrete_greens_operator.hh"

#include <libmugrid/field_map.hh>
#include <libmugrid/eigen_tools.hh>
#include <libmugrid/iterators.hh>

#include <Eigen/IterativeLinearSolvers>
#include <utility>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  DiscreteGreensOperator::DiscreteGreensOperator(
      muFFT::FFTEngine_ptr engine, const RealSpaceField_t & impulse_response,
      const Index_t & displacement_rank)
      : engine{std::move(engine)},
        nb_dof_per_pixel{static_cast<Index_t>(
            muGrid::ct_sqrt(impulse_response.get_nb_dof_per_pixel()))},
        displacement_rank{displacement_rank},
        diagonals(this->engine->register_fourier_space_field(
            this->engine->get_fourier_field_collection().generate_unique_name(),
            this->nb_dof_per_pixel * this->nb_dof_per_pixel)),
        field_values(this->engine->register_fourier_space_field(
            this->engine->get_fourier_field_collection().generate_unique_name(),
            this->nb_dof_per_pixel)) {
    this->engine->fft(impulse_response, this->diagonals);

    using FieldMap_t = muGrid::FieldMap<Complex, Mapping::Mut>;
    FieldMap_t map{this->diagonals, this->nb_dof_per_pixel, IterUnit::Pixel};
    bool first{true};
    for (auto && acoustic : map) {
      if (first) {
        // determine rank
        const Index_t rank{this->nb_dof_per_pixel -
                           muGrid::ipow(this->engine->get_spatial_dim(),
                                        this->displacement_rank)};
        if (rank == 0) {
          acoustic.setZero();
        } else {
          Eigen::ConjugateGradient<FieldMap_t::PlainType> cg;
          cg.compute(acoustic);
          // for the zero-frequency k-space pixel, we need to take the Penrose
          // pseudo-inverse of the stiffness matrix, as this matrix is
          // rank-deficient by order of the spatial dimension, but is not
          // entirely zero rank. This means that setting its acoustic tensor to
          // zero would block some degrees of freedom that should not be
          // blocked. The pseudo-inverse algorithm is highly unstable, however,
          // so the simplest solution we found is to use a conjugate gradient
          // solver to find the pseudo-inverse. This may or may not be the most
          // efficient and robust way.
          auto && tmp{cg.solve(
              Eigen::MatrixXcd::Identity(acoustic.rows(), acoustic.cols()))};
          acoustic = tmp;
        }
        first = false;
      } else {
        acoustic = acoustic.inverse().eval();
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  void DiscreteGreensOperator::apply(RealSpaceField_t & field) {
    muGrid::FieldMap<Complex, Mapping::Const> greens_fun_map{
        this->diagonals, this->nb_dof_per_pixel, IterUnit::Pixel};

    this->engine->fft(field, this->field_values);
    muGrid::FieldMap<Complex, Mapping::Mut> field_value_map{
        this->field_values, this->nb_dof_per_pixel, IterUnit::Pixel};
    auto && normalisation{this->engine->normalisation()};

    for (auto && tup : akantu::zip(greens_fun_map, field_value_map)) {
      auto && greens_mat{std::get<0>(tup)};
      auto && field_value{std::get<1>(tup)};
      field_value = normalisation * (greens_mat * field_value).eval();
    }

    this->engine->ifft(this->field_values, field);
  }

  /* ---------------------------------------------------------------------- */
  void DiscreteGreensOperator::apply(const RealSpaceField_t & input_field,
                                     RealSpaceField_t & output_field) {
    muGrid::FieldMap<Complex, Mapping::Const> greens_fun_map{
        this->diagonals, this->nb_dof_per_pixel, IterUnit::Pixel};

    auto && normalisation{this->engine->normalisation()};

    this->engine->fft(input_field, this->field_values);
    muGrid::FieldMap<Complex, Mapping::Mut> field_value_map{
        this->field_values, this->nb_dof_per_pixel, IterUnit::Pixel};

    for (auto && tup : akantu::zip(greens_fun_map, field_value_map)) {
      auto && greens_mat{std::get<0>(tup)};
      auto && field_value{std::get<1>(tup)};
      field_value = (normalisation * greens_mat * field_value).eval();
    }

    this->engine->ifft(this->field_values, output_field);
  }

  /* ---------------------------------------------------------------------- */
  void
  DiscreteGreensOperator::apply_increment(const RealSpaceField_t & input_field,
                                          const Real & alpha,
                                          RealSpaceField_t & output_field) {
    auto & output_copy{this->engine->fetch_or_register_real_space_field(
        "Temporary_copy_for_greens_operator", this->nb_dof_per_pixel)};
    output_copy = output_field;
    this->apply(input_field, output_field);
    output_field.eigen_vec() =
        (output_copy.eigen_vec() + alpha * output_field.eigen_vec()).eval();
  }

  /* ---------------------------------------------------------------------- */
  Index_t DiscreteGreensOperator::get_nb_dof() const {
    return this->nb_dof_per_pixel *
           muGrid::CcoordOps::get_size(this->engine->get_nb_domain_grid_pts());
  }

  /* ---------------------------------------------------------------------- */
  void DiscreteGreensOperator::action_increment(EigenCVecRef delta_grad,
                                                const Real & alpha,
                                                EigenVecRef delta_flux) {
    auto delta_grad_field_ptr{muGrid::WrappedField<Real>::make_const(
        "Greens_operator_input_field",
        this->engine->get_real_field_collection(), this->nb_dof_per_pixel,
        delta_grad, PixelTag)};
    muGrid::WrappedField<Real> delta_flux_field{
        "Greens_operator_output_field",
        this->engine->get_real_field_collection(), this->nb_dof_per_pixel,
        delta_flux, PixelTag};
    this->apply_increment(*delta_grad_field_ptr, alpha, delta_flux_field);
  }

  /* ---------------------------------------------------------------------- */
  const muFFT::Communicator & DiscreteGreensOperator::get_communicator() const {
    return this->engine->get_communicator();
  }

}  // namespace muSpectre
