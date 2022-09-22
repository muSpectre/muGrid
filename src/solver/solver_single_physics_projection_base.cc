/**
 * @file   solver_single_physics_projection_base.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   13 Feb 2022
 *
 * @brief  implementation of SolverSinglePhysicsProjectionBase
 *
 * Copyright © 2022 Ali Falsafi
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

#include "solver_single_physics_projection_base.hh"

#include "projection/projection_finite_strain_fast.hh"
#include "projection/projection_small_strain.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  SolverSinglePhysicsProjectionBase::SolverSinglePhysicsProjectionBase(
      std::shared_ptr<CellData> cell_data, const muGrid::Verbosity & verbosity,
      const Real & newton_tol, const Real & equil_tol, const Uint & max_iter,
      const Gradient_t & gradient, const Weights_t & weights,
      const MeanControl & mean_control)
      : Parent{cell_data, verbosity, SolverType::Spectral},
        newton_tol{newton_tol}, equil_tol{equil_tol}, max_iter{max_iter},
        gradient{std::make_shared<Gradient_t>(gradient)},
        weights{std::make_shared<Weights_t>(weights)},
        nb_quad_pts{static_cast<Index_t>(gradient.size()) /
                    (this->cell_data->get_domain_lengths().get_dim())},
        mean_control{mean_control} {};

  /* ---------------------------------------------------------------------- */
  SolverSinglePhysicsProjectionBase::SolverSinglePhysicsProjectionBase(
      std::shared_ptr<CellData> cell_data, const muGrid::Verbosity & verbosity,
      const Real & newton_tol, const Real & equil_tol, const Uint & max_iter,
      const MeanControl & mean_control)
      : Parent{cell_data, verbosity, SolverType::Spectral},
        newton_tol{newton_tol}, equil_tol{equil_tol}, max_iter{max_iter},
        gradient{std::make_shared<Gradient_t>(
            muFFT::make_fourier_gradient(this->cell_data->get_spatial_dim()))},
        weights{std::make_shared<Weights_t>(Weights_t{1})},
        mean_control{mean_control} {};

  /* ---------------------------------------------------------------------- */
  void SolverSinglePhysicsProjectionBase::initialise_eigen_strain_storage() {
    if (not this->has_eigen_strain_storage()) {
      //! store solver fields in cell
      auto & field_collection{this->cell_data->get_fields()};

      this->eval_grad = std::make_shared<MappedField_t>(
          this->fetch_or_register_field("eval_grad", this->grad_shape[0],
                                        this->grad_shape[1], field_collection,
                                        QuadPtTag),
          this->grad_shape[0], IterUnit::SubPt);
      this->eval_grads[this->domain] = this->eval_grad;
    }
  }

  /* ---------------------------------------------------------------------- */
  void SolverSinglePhysicsProjectionBase::initialise_cell_worker() {
    // make sure the number of subpoints is correct
    this->cell_data->set_nb_quad_pts(this->nb_quad_pts);
    this->cell_data->set_nb_nodal_pts(OneNode);

    //! check whether formulation has been set
    if (this->is_mechanics()) {
      if (this->get_formulation() == Formulation::not_set) {
        throw SolverError(
            "Can't run a mechanics calculation without knowing the "
            "formulation. please use the `set_formulation()` to "
            "choose between finite and small strain");
      } else {
        for (auto && mat :
             this->cell_data->get_domain_materials().at(this->domain)) {
          auto mech{std::dynamic_pointer_cast<MaterialMechanicsBase>(mat)};
          if (this->get_formulation() == Formulation::small_strain) {
            mech->check_small_strain_capability();
          }
          mech->set_formulation(this->get_formulation());
        }
      }
    }

    this->grad_shape =
        gradient_shape(this->domain.rank(), this->cell_data->get_spatial_dim(),
                       this->is_mechanics(), this->get_formulation());

    //! store solver fields in cell
    auto & field_collection{this->cell_data->get_fields()};

    // Corresponds to symbol δF or δε
    this->grad_incr = std::make_shared<MappedField_t>(
        this->fetch_or_register_field("incrF", this->grad_shape[0],
                                      this->grad_shape[1], field_collection,
                                      QuadPtTag),
        this->grad_shape[0], IterUnit::SubPt);

    // Corresponds to symbol F or ε
    this->grad = std::make_shared<MappedField_t>(
        this->fetch_or_register_field("grad", this->grad_shape[0],
                                      this->grad_shape[1], field_collection,
                                      QuadPtTag),
        this->grad_shape[0], IterUnit::SubPt);

    this->eval_grad = this->grad;

    this->eval_grads[this->domain] = this->eval_grad;
    this->grads[this->domain] = this->grad;

    // Corresponds to symbol P or σ
    this->flux = std::make_shared<MappedField_t>(
        this->fetch_or_register_field("flux", this->grad_shape[0],
                                      this->grad_shape[1], field_collection,
                                      QuadPtTag),
        this->grad_shape[0], IterUnit::SubPt);
    this->fluxes[this->domain] = this->flux;

    // Corresponds to symbol K or C
    Index_t tangent_size{this->grad_shape[0] * this->grad_shape[1]};
    this->tangent = std::make_shared<MappedField_t>(
        this->fetch_or_register_field("tangent", tangent_size, tangent_size,
                                      field_collection, QuadPtTag),
        this->grad_shape[0], IterUnit::SubPt);
    this->tangents[this->domain] = this->tangent;

    // field to store the rhs for cg calculations
    this->rhs = std::make_shared<MappedField_t>(
        this->fetch_or_register_field("rhs", this->grad_shape[0],
                                      this->grad_shape[1], field_collection,
                                      QuadPtTag),
        this->grad_shape[0], IterUnit::SubPt);

    Eigen::MatrixXd default_grad_val{};
    if (this->is_mechanics()) {
      switch (this->get_formulation()) {
      case Formulation::finite_strain: {
        default_grad_val =
            Eigen::MatrixXd::Identity(this->grad_shape[0], this->grad_shape[1]);
        break;
      }
      case Formulation::small_strain: {
        default_grad_val =
            Eigen::MatrixXd::Zero(this->grad_shape[0], this->grad_shape[1]);
        break;
      }
      default:
        std::stringstream error_msg{};
        error_msg << "I don't know how to handle the Formulation '"
                  << this->get_formulation() << "'.";
        throw SolverError{error_msg.str()};
        break;
      }
    } else {
      default_grad_val =
          Eigen::MatrixXd::Zero(this->grad_shape[0], this->grad_shape[1]);
    }
    this->grad->get_map() = default_grad_val;

    this->previous_macro_load.setZero(this->grad_shape[0], this->grad_shape[1]);

    // make sure all materials are initialised and every pixel is covered
    this->cell_data->check_material_coverage();

    if (this->is_mechanics()) {
      this->create_mechanics_projection();
    } else {
      this->create_gradient_projection();
    }
  }

  /* ---------------------------------------------------------------------- */
  bool SolverSinglePhysicsProjectionBase::has_eigen_strain_storage() const {
    return this->eval_grad != this->grad;
  }

  /* ---------------------------------------------------------------------- */
  Index_t SolverSinglePhysicsProjectionBase::get_nb_dof() const {
    if (not this->is_initialised) {
      throw SolverError{"Can't determine the number of degrees of freedom "
                        "until I have been "
                        "initialised!"};
    }
    return this->cell_data->get_pixels().size() *
           this->cell_data->get_nb_quad_pts() * this->grad_shape[0] *
           this->grad_shape[1];
  }

  /* ---------------------------------------------------------------------- */
  void SolverSinglePhysicsProjectionBase::action_increment(
      EigenCVecRef delta_grad, const Real & alpha, EigenVecRef del_flux) {
    auto && grad_size{this->grad_shape[0] * this->grad_shape[1]};
    auto delta_grad_ptr{muGrid::WrappedField<Real>::make_const(
        "delta Grad", this->cell_data->get_fields(), grad_size, delta_grad,
        QuadPtTag)};

    muGrid::WrappedField<Real> del_flux_field{"delta_flux",
                                              this->cell_data->get_fields(),
                                              grad_size, del_flux, QuadPtTag};

    switch (this->cell_data->get_material_dim()) {
    case twoD: {
      this->template action_increment_worker_prep<twoD>(
          *delta_grad_ptr, this->tangent->get_field(), alpha, del_flux_field,
          this->get_displacement_rank());
      break;
    }
    case threeD: {
      this->template action_increment_worker_prep<threeD>(
          *delta_grad_ptr, this->tangent->get_field(), alpha, del_flux_field,
          this->get_displacement_rank());
      break;
    }
    default:
      std::stringstream err{};
      err << "unknown dimension " << this->cell_data->get_material_dim()
          << std::endl;
      throw SolverError(err.str());
      break;
    }
    this->projection->apply_projection(del_flux_field);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM>
  void SolverSinglePhysicsProjectionBase::action_increment_worker_prep(
      const muGrid::TypedFieldBase<Real> & delta_grad,
      const muGrid::TypedFieldBase<Real> & tangent, const Real & alpha,
      muGrid::TypedFieldBase<Real> & delta_flux,
      const Index_t & displacement_rank) {
    switch (displacement_rank) {
    case zerothOrder: {
      // this is a scalar problem, e.g., heat equation
      SolverSinglePhysicsProjectionBase::template action_increment_worker<
          DimM, zerothOrder>(delta_grad, tangent, alpha, delta_flux);
      break;
    }
    case firstOrder: {
      // this is a vectorial problem, e.g., mechanics
      SolverSinglePhysicsProjectionBase::template action_increment_worker<
          DimM, firstOrder>(delta_grad, tangent, alpha, delta_flux);
      break;
    }
    default:
      throw SolverError("Can only handle scalar and vectorial problems");
      break;
    }
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM, Index_t DisplacementRank>
  void SolverSinglePhysicsProjectionBase::action_increment_worker(
      const muGrid::TypedFieldBase<Real> & delta_grad,
      const muGrid::TypedFieldBase<Real> & tangent, const Real & alpha,
      muGrid::TypedFieldBase<Real> & delta_flux) {
    constexpr Index_t GradRank{DisplacementRank + 1};
    static_assert(
        (GradRank == firstOrder) or (GradRank == secondOrder),
        "Can only handle vectors and second-rank tensors as gradients");
    constexpr Index_t TangentRank{GradRank + GradRank};
    muGrid::TensorFieldMap<Real, muGrid::Mapping::Const, GradRank, DimM,
                           IterUnit::SubPt>
        grad_map{delta_grad};
    muGrid::TensorFieldMap<Real, muGrid::Mapping::Const, TangentRank, DimM,
                           IterUnit::SubPt>
        tangent_map{tangent};
    muGrid::TensorFieldMap<Real, muGrid::Mapping::Mut, GradRank, DimM,
                           IterUnit::SubPt>
        flux_map{delta_flux};
    for (auto && tup : akantu::zip(grad_map, tangent_map, flux_map)) {
      auto & df = std::get<0>(tup);
      auto & k = std::get<1>(tup);
      auto & dp = std::get<2>(tup);
      Eigen::MatrixXd tmp{alpha * Matrices::tensmult(k, df)};
      dp += tmp;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  void SolverSinglePhysicsProjectionBase::create_mechanics_projection_worker() {
    if (static_cast<Index_t>(this->gradient->size()) % Dim != 0) {
      std::stringstream error{};
      error << "There are " << this->gradient->size()
            << " derivative operators in "
            << "the gradient. This number must be divisible by the system "
            << "dimension " << Dim << ".";
      throw std::runtime_error(error.str());
    }
    // Deduce number of quad points from the gradient
    auto fft_ptr{this->cell_data->get_FFT_engine()};
    // fft_ptr->create_plan(this->nb_quad_pts);
    // fft_ptr->create_plan(this->gradient->size());

    const DynRcoord_t lengths{this->cell_data->get_domain_lengths()};
    switch (this->get_formulation()) {
    case Formulation::finite_strain: {
      switch (this->nb_quad_pts) {
      case OneQuadPt: {
        using Projection = ProjectionFiniteStrainFast<Dim, OneQuadPt>;
        this->projection = std::make_shared<Projection>(
            std::move(fft_ptr), lengths, *this->gradient, *this->weights,
            this->mean_control);
        break;
      }
      case TwoQuadPts: {
        using Projection = ProjectionFiniteStrainFast<Dim, TwoQuadPts>;
        this->projection = std::make_shared<Projection>(
            std::move(fft_ptr), lengths, *this->gradient, *this->weights,
            this->mean_control);
        break;
      }
      case FourQuadPts: {
        using Projection = ProjectionFiniteStrainFast<Dim, FourQuadPts>;
        this->projection = std::make_shared<Projection>(
            std::move(fft_ptr), lengths, *this->gradient, *this->weights,
            this->mean_control);
        break;
      }
      case FiveQuadPts: {
        using Projection = ProjectionFiniteStrainFast<Dim, FiveQuadPts>;
        this->projection = std::make_shared<Projection>(
            std::move(fft_ptr), lengths, *this->gradient, *this->weights,
            this->mean_control);
        break;
      }
      case SixQuadPts: {
        using Projection = ProjectionFiniteStrainFast<Dim, SixQuadPts>;
        this->projection = std::make_shared<Projection>(
            std::move(fft_ptr), lengths, *this->gradient, *this->weights,
            this->mean_control);
        break;
      }
      default: {
        std::stringstream error;
        error << this->nb_quad_pts << " quadrature points are presently "
              << "unsupported.";
        throw std::runtime_error(error.str());
        break;
      }
      }
      break;
    }
    case Formulation::small_strain: {
      switch (this->nb_quad_pts) {
      case OneQuadPt: {
        using Projection = ProjectionSmallStrain<Dim, OneQuadPt>;
        this->projection = std::make_shared<Projection>(
            std::move(fft_ptr), lengths, *this->gradient, *this->weights,
            this->mean_control);
        break;
      }
      case TwoQuadPts: {
        using Projection = ProjectionSmallStrain<Dim, TwoQuadPts>;
        this->projection = std::make_shared<Projection>(
            std::move(fft_ptr), lengths, *this->gradient, *this->weights,
            this->mean_control);
        break;
      }
      case FourQuadPts: {
        using Projection = ProjectionSmallStrain<Dim, FourQuadPts>;
        this->projection = std::make_shared<Projection>(
            std::move(fft_ptr), lengths, *this->gradient, *this->weights,
            this->mean_control);
        break;
      }
      case FiveQuadPts: {
        using Projection = ProjectionSmallStrain<Dim, FiveQuadPts>;
        this->projection = std::make_shared<Projection>(
            std::move(fft_ptr), lengths, *this->gradient, *this->weights,
            this->mean_control);
        break;
      }
      case SixQuadPts: {
        using Projection = ProjectionSmallStrain<Dim, SixQuadPts>;
        this->projection = std::make_shared<Projection>(
            std::move(fft_ptr), lengths, *this->gradient, *this->weights,
            this->mean_control);
        break;
      }
      default:
        std::stringstream error;
        error << this->nb_quad_pts << " quadrature points are presently "
              << "unsupported.";
        throw std::runtime_error(error.str());
        break;
      }
      break;
    }
    default: {
      throw std::runtime_error("Unknown formulation (in projection creation).");
      break;
    }
    }
  }

  /* ---------------------------------------------------------------------- */
  void SolverSinglePhysicsProjectionBase::create_mechanics_projection() {
    switch (this->cell_data->get_spatial_dim()) {
    case twoD: {
      this->template create_mechanics_projection_worker<twoD>();
      break;
    }
    case threeD: {
      this->template create_mechanics_projection_worker<threeD>();
      break;
    }
    default:
      std::stringstream error_message{};
      error_message << "Only 2- and 3-dimensional projections are currently "
                       "supported, you chose "
                    << this->cell_data->get_spatial_dim() << '.';
      throw SolverError{error_message.str()};
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  void SolverSinglePhysicsProjectionBase::create_gradient_projection() {
    switch (this->cell_data->get_spatial_dim()) {
    case twoD: {
      this->projection = std::make_shared<ProjectionGradient<twoD, firstOrder>>(
          this->cell_data->get_FFT_engine(),
          this->cell_data->get_domain_lengths(), this->mean_control);
      break;
    }
    case threeD: {
      this->projection =
          std::make_shared<ProjectionGradient<threeD, firstOrder>>(
              this->cell_data->get_FFT_engine(),
              this->cell_data->get_domain_lengths(), this->mean_control);
      break;
    }
    default:
      std::stringstream error_message{};
      error_message << "generic gradient projection is not implemented for "
                    << this->cell_data->get_spatial_dim()
                    << "-dimensional problems.";
      throw SolverError{error_message.str()};
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  Index_t SolverSinglePhysicsProjectionBase::get_displacement_rank() const {
    return this->domain.rank() - 1;
  }

  /* ---------------------------------------------------------------------- */
  auto SolverSinglePhysicsProjectionBase::get_projection()
      -> const ProjectionBase & {
    if (this->projection == nullptr) {
      throw SolverError("Projection is not yet defined.");
    }
    return *this->projection;
  }

  /* ---------------------------------------------------------------------- */
  auto SolverSinglePhysicsProjectionBase::get_rhs() -> MappedField_t & {
    return *this->rhs;
  }

  /* ---------------------------------------------------------------------- */
  auto SolverSinglePhysicsProjectionBase::get_incr() -> MappedField_t & {
    return *this->grad_incr;
  }

  /* ---------------------------------------------------------------------- */
  auto SolverSinglePhysicsProjectionBase::compute_effective_stiffness()
      -> EigenMat_t {
    if (not this->is_initialised) {
      this->initialise_cell();
    }

    if (not(this->mean_control == MeanControl::StrainControl)) {
      std::stringstream err{};
      err << "This function is currently only usable for solvers"
          << " with a strain control projection operator" << std::endl
          << "the algorithm derived needs to use a strain control"
          << " projection operator to zero out σ_eq" << std::endl
          << " NOTE: Please try to define an additional similar solver with"
          << " MeanControl::StrainControl if you are using any other mean"
          << " control for your main solver" << std::endl
          << ", NOTE: you also need to define a new"
          << " linear(Krylov) solver and pass it to the constructor of the"
          << " newly defined solver" << std::endl;
      throw SolverError(err.str());
    }

    auto && comm{this->cell_data->get_communicator()};
    auto && dim{this->cell_data->get_spatial_dim()};
    Dim_t dim_square{dim * dim};

    // unit test strains
    EigenMat_t unit_test_loads{this->create_unit_test_strain()};

    // The return of this method :C_eff
    EigenMat_t effective_tangent{EigenMat_t::Identity(dim_square, dim_square)};

    // extracting fields
    auto && rhs_field{this->get_rhs().get_field()};
    auto && grad_incr_field{this->grad_incr->get_field()};
    auto && tangent_field{this->tangent->get_field()};

    auto && tangnet_map{tangent_field.eigen_map(
        dim_square, dim_square * tangent_field.get_nb_entries())};
    auto && grad_incr_vec{grad_incr_field.eigen_vec()};

    for (Index_t i = 0; i < unit_test_loads.cols(); ++i) {
      auto && unit_test_load{unit_test_loads.col(i)};

      // calculation of -CδEᵖᵉʳᵗ:
      rhs_field.eigen_vec() = -tangnet_map.transpose() * unit_test_load;

      // rhs = -G (C : δEᵖᵉʳᵗ)
      this->projection->apply_projection(rhs_field);

      try {
        // solve G(C : δε) = -G(C : δEᵖᵉʳᵗ) for δε
        grad_incr_field =
            this->get_krylov_solver().solve(rhs_field.eigen_vec());
      } catch (ConvergenceError & error) {
        std::stringstream err{};
        err << "Failure at calculating effective stiffness"
            << "The applied boundary condition Δ" << this->strain_symb() << "="
            << unit_test_load << std::endl;
        throw ConvergenceError(err.str());
      }

      Eigen::Map<EigenMat_t> delE(unit_test_load.data(), dim, dim);
      // δσᵖᵉʳᵗ = Cw : (δε + δEᵖᵉʳᵗ)
      this->grad_incr->get_map() += delE;
      EigenMat_t sum_stress_loc{tangnet_map * grad_incr_vec};

      // δσ̄ᵖᵉʳᵗ = <δσᵖᵉʳᵗ>
      EigenMat_t sum_stress_pert{comm.sum(sum_stress_loc)};
      Index_t size_stress_pert{
          Index_t(comm.sum(this->grad_incr->get_map().size()))};
      EigenMat_t mean_stress_pert{sum_stress_pert / size_stress_pert};

      /* δσ̄ᵖᵉʳᵗ=C_eff * δEᵖᵉʳᵗ ----------------|
       *                                        >⇒ C_eff[:,i]=σ̄ᵖᵉʳᵗ-σ̄⁰
       * δEᵖᵉʳᵗ=[0..0,1,0..0], 1 at iᵗʰ element|
       *note: δσ̄s are vectorized*/

      // vectorizing the δσ̄
      EigenVec_t mean_stress_pert_vectorized{Eigen::Map<EigenVec_t>(
          mean_stress_pert.data(),
          mean_stress_pert.cols() * mean_stress_pert.rows())};

      // replacing the δσ̄ in iᵗʰ row of C_eff
      effective_tangent.row(i) = mean_stress_pert_vectorized;
    }
    return effective_tangent;
  }

  /* ---------------------------------------------------------------------- */
  auto SolverSinglePhysicsProjectionBase::create_unit_test_strain()
      -> EigenMat_t {
    auto && dim{this->cell_data->get_spatial_dim()};
    Dim_t dim_square{dim * dim};
    // return of this function  δEᵖᵉʳᵗ s
    EigenMat_t unit_test_loads{EigenMat_t::Identity(dim_square, dim_square)};

    // symmetrize lambda function
    auto && symmetric{[&dim](Eigen::MatrixXd vec) -> Eigen::MatrixXd {
      Eigen::Map<EigenMat_t> mat(vec.data(), dim, dim);
      EigenMat_t sym_mat{0.5 * (mat + mat.transpose()).eval()};
      Eigen::Map<EigenMat_t> sym_vec(sym_mat.data(), dim * dim, 1);
      return sym_vec;
    }};

    // symmetrize the strain in case of small strain formulation
    auto && Form{this->get_formulation()};
    switch (Form) {
    case Formulation::small_strain: {
      for (Dim_t i{0}; i < dim_square; i++) {
        auto && unit_test_load{unit_test_loads.col(i)};
        unit_test_load = symmetric(unit_test_load);
      }
      break;
    }
    case Formulation::finite_strain: {
      // do nothing an just return the identity as the matrix containing the
      // test unit strain at its columns
      break;
    }
    default: {
      throw SolverError(
          "Only can create unit test loads for mechanics problems with either "
          "small strain or finite strain formulation");
      break;
    }
    }
    return unit_test_loads;
  }

  /* ---------------------------------------------------------------------- */
  auto SolverSinglePhysicsProjectionBase::fetch_or_register_field(
      const std::string & unique_name, const Index_t & nb_rows,
      const Index_t & nb_cols, muGrid::FieldCollection & collection,
      const std::string & sub_division_tag) -> RealField & {
    if (collection.field_exists(unique_name)) {
      auto & field{
          dynamic_cast<RealField &>(collection.get_field(unique_name))};
      return field;
    } else {
      return collection.register_field<Real>(unique_name, {nb_rows, nb_cols},
                                             sub_division_tag);
    }
  }

  /* ---------------------------------------------------------------------- */
}  // namespace muSpectre
