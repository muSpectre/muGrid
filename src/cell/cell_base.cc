/**
 * @file   cell_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Nov 2017
 *
 * @brief  Implementation for cell base class
 *
 * Copyright © 2017 Till Junge
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

#include "cell/cell_base.hh"
#include "common/ccoord_operations.hh"
#include "common/iterators.hh"
#include "common/tensor_algebra.hh"

#include <sstream>
#include <algorithm>


namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  CellBase<DimS, DimM>::CellBase(Projection_ptr projection_)
    :subdomain_resolutions{projection_->get_subdomain_resolutions()},
     subdomain_locations{projection_->get_subdomain_locations()},
     domain_resolutions{projection_->get_domain_resolutions()},
     pixels(subdomain_resolutions, subdomain_locations),
     domain_lengths{projection_->get_domain_lengths()},
     fields{std::make_unique<FieldCollection_t>()},
     F{make_field<StrainField_t>("Gradient", *this->fields)},
     P{make_field<StressField_t>("Piola-Kirchhoff-1", *this->fields)},
     projection{std::move(projection_)}
  { }

  /**
   * turns out that the default move container in combination with
   * clang segfaults under certain (unclear) cicumstances, because the
   * move constructor of the optional appears to be busted in gcc
   * 7.2. Copying it (K) instead of moving it fixes the issue, and
   * since it is a reference, the cost is practically nil
   */
  template <Dim_t DimS, Dim_t DimM>
  CellBase<DimS, DimM>::CellBase(CellBase && other):
    subdomain_resolutions{std::move(other.subdomain_resolutions)},
    subdomain_locations{std::move(other.subdomain_locations)},
    domain_resolutions{std::move(other.domain_resolutions)},
    pixels{std::move(other.pixels)},
    domain_lengths{std::move(other.domain_lengths)},
    fields{std::move(other.fields)},
    F{other.F},
    P{other.P},
    K{other.K}, // this seems to segfault under clang if it's not a move
    materials{std::move(other.materials)},
    projection{std::move(other.projection)}
  { }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::Material_t &
  CellBase<DimS, DimM>::add_material(Material_ptr mat) {
    this->materials.push_back(std::move(mat));
    return *this->materials.back();
  }


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto CellBase<DimS, DimM>::get_strain_vector() -> Vector_ref {
    return this->get_strain().eigenvec();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto CellBase<DimS, DimM>::get_stress_vector() const -> ConstVector_ref {
    return this->get_stress().eigenvec();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void CellBase<DimS, DimM>::
  set_uniform_strain(const Eigen::Ref<const Matrix_t> & strain) {
    this->F.get_map() = strain;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto CellBase<DimS, DimM>::evaluate_stress() -> ConstVector_ref {
    if (not this->initialised) {
      this->initialise();
    }
    for (auto & mat: this->materials) {
      mat->compute_stresses(this->F, this->P, this->get_formulation());
    }

    return this->P.const_eigenvec();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto CellBase<DimS, DimM>::
  evaluate_stress_tangent() -> std::array<ConstVector_ref, 2> {
    if (not this->initialised) {
      this->initialise();
    }

    constexpr bool create_tangent{true};
    this->get_tangent(create_tangent);

    for (auto & mat: this->materials) {
      mat->compute_stresses_tangent(this->F, this->P, this->K.value(),
                                    this->get_formulation());
    }
    const TangentField_t & k = this->K.value();
    return std::array<ConstVector_ref, 2>{
      this->P.const_eigenvec(), k.const_eigenvec()};

  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto CellBase<DimS, DimM>::
  evaluate_projected_directional_stiffness
  (Eigen::Ref<const Vector_t> delF) -> Vector_ref {
    // the following const_cast should be safe, as long as the
    // constructed delF_field is const itself
    const TypedField<FieldCollection_t, Real> delF_field
      ("Proxied raw memory for strain increment",
       *this->fields,
       Eigen::Map<Vector_t>(const_cast<Real *>(delF.data()), delF.size()),
       this->F.get_nb_components());

    if (!this->K) {
      throw std::runtime_error
        ("currently only implemented for cases where a stiffness matrix "
         "exists");
    }

    if (delF.size() != this->get_nb_dof()) {
      std::stringstream err{};
      err << "input should be of size ndof = ¶(" << this->subdomain_resolutions
          << ") × " << DimS << "² = "<< this->get_nb_dof() << " but I got "
          << delF.size();
      throw std::runtime_error(err.str());
    }

    const std::string out_name{"δP; temp output for directional stiffness"};
    auto & delP = this->get_managed_field(out_name);

    auto Kmap{this->K.value().get().get_map()};
    auto delPmap{delP.get_map()};
    MatrixFieldMap<FieldCollection_t, Real, DimM, DimM, true> delFmap(delF_field);

    for (auto && tup:
           akantu::zip(Kmap, delFmap, delPmap)) {
      auto & k = std::get<0>(tup);
      auto & df = std::get<1>(tup);
      auto & dp = std::get<2>(tup);
      dp = Matrices::tensmult(k, df);
    }

    return Vector_ref(this->project(delP).data(), this->get_nb_dof());

  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  std::array<Dim_t, 2> CellBase<DimS, DimM>::get_strain_shape() const {
    return this->projection->get_strain_shape();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void CellBase<DimS, DimM>::apply_projection(Eigen::Ref<Vector_t> vec) {
    TypedField<FieldCollection_t, Real> field("Proxy for projection",
                                              *this->fields,
                                              vec,
                                              this->F.get_nb_components());
    this->projection->apply_projection(field);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::FullResponse_t
  CellBase<DimS, DimM>::evaluate_stress_tangent(StrainField_t & grad) {
    if (this->initialised == false) {
      this->initialise();
    }
    //! High level compatibility checks
    if (grad.size() != this->F.size()) {
      throw std::runtime_error("Size mismatch");
    }
    constexpr bool create_tangent{true};
    this->get_tangent(create_tangent);

    for (auto & mat: this->materials) {
      mat->compute_stresses_tangent(grad, this->P, this->K.value(),
                                    this->get_formulation());
    }
    return std::tie(this->P, this->K.value());
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::StressField_t &
  CellBase<DimS, DimM>::directional_stiffness(const TangentField_t &K,
                                              const StrainField_t &delF,
                                              StressField_t &delP) {
    for (auto && tup:
           akantu::zip(K.get_map(), delF.get_map(), delP.get_map())){
      auto & k = std::get<0>(tup);
      auto & df = std::get<1>(tup);
      auto & dp = std::get<2>(tup);
      dp = Matrices::tensmult(k, df);
    }
    return this->project(delP);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::Vector_ref
  CellBase<DimS, DimM>::directional_stiffness_vec(const Eigen::Ref<const Vector_t> &delF) {
    if (!this->K) {
      throw std::runtime_error
        ("currently only implemented for cases where a stiffness matrix "
         "exists");
    }
    if (delF.size() != this->get_nb_dof()) {
      std::stringstream err{};
      err << "input should be of size ndof = ¶(" << this->subdomain_resolutions
          << ") × " << DimS << "² = "<< this->get_nb_dof() << " but I got "
          << delF.size();
      throw std::runtime_error(err.str());
    }
    const std::string out_name{"temp output for directional stiffness"};
    const std::string in_name{"temp input for directional stiffness"};

    auto & out_tempref = this->get_managed_field(out_name);
    auto & in_tempref = this->get_managed_field(in_name);
    Vector_ref(in_tempref.data(), this->get_nb_dof()) = delF;

    this->directional_stiffness(this->K.value(), in_tempref, out_tempref);
    return Vector_ref(out_tempref.data(), this->get_nb_dof());

  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  Eigen::ArrayXXd
  CellBase<DimS, DimM>::
  directional_stiffness_with_copy
    (Eigen::Ref<Eigen::ArrayXXd> delF) {
    if (!this->K) {
      throw std::runtime_error
        ("currently only implemented for cases where a stiffness matrix "
         "exists");
    }
    const std::string out_name{"temp output for directional stiffness"};
    const std::string in_name{"temp input for directional stiffness"};

    auto & out_tempref = this->get_managed_field(out_name);
    auto & in_tempref = this->get_managed_field(in_name);
    in_tempref.eigen() = delF;
    this->directional_stiffness(this->K.value(), in_tempref, out_tempref);
    return out_tempref.eigen();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::StressField_t &
  CellBase<DimS, DimM>::project(StressField_t &field) {
    this->projection->apply_projection(field);
    return field;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::StrainField_t &
  CellBase<DimS, DimM>::get_strain() {
    if (this->initialised == false) {
      this->initialise();
    }
    return this->F;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  const typename CellBase<DimS, DimM>::StressField_t &
  CellBase<DimS, DimM>::get_stress() const {
    return this->P;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  const typename CellBase<DimS, DimM>::TangentField_t &
  CellBase<DimS, DimM>::get_tangent(bool create) {
    if (!this->K) {
      if (create) {
        this->K = make_field<TangentField_t>("Tangent Stiffness", *this->fields);
      } else {
        throw std::runtime_error
          ("K does not exist");
      }
    }
    return this->K.value();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::StrainField_t &
  CellBase<DimS, DimM>::get_managed_field(std::string unique_name) {
    if (!this->fields->check_field_exists(unique_name)) {
      return make_field<StressField_t>(unique_name, *this->fields);
    } else {
      return static_cast<StressField_t&>(this->fields->at(unique_name));
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void CellBase<DimS, DimM>::initialise(FFT_PlanFlags flags) {
    // check that all pixels have been assigned exactly one material
    this->check_material_coverage();
    for (auto && mat: this->materials) {
      mat->initialise();
    }
    // resize all global fields (strain, stress, etc)
    this->fields->initialise(this->subdomain_resolutions, this->subdomain_locations);
    // initialise the projection and compute the fft plan
    this->projection->initialise(flags);
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void CellBase<DimS, DimM>::save_history_variables() {
    for (auto && mat: this->materials) {
      mat->save_history_variables();
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::iterator
  CellBase<DimS, DimM>::begin() {
    return this->pixels.begin();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::iterator
  CellBase<DimS, DimM>::end() {
    return this->pixels.end();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto CellBase<DimS, DimM>::get_adaptor() -> Adaptor {
    return Adaptor(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void CellBase<DimS, DimM>::check_material_coverage() {
    auto nb_pixels = CcoordOps::get_size(this->subdomain_resolutions);
    std::vector<MaterialBase<DimS, DimM>*> assignments(nb_pixels, nullptr);
    for (auto & mat: this->materials) {
      for (auto & pixel: *mat) {
        auto index = CcoordOps::get_index(this->subdomain_resolutions,
                                          this->subdomain_locations,
                                          pixel);
        auto& assignment{assignments.at(index)};
        if (assignment != nullptr) {
          std::stringstream err{};
          err << "Pixel " << pixel << "is already assigned to material '"
              << assignment->get_name()
              << "' and cannot be reassigned to material '" << mat->get_name();
          throw std::runtime_error(err.str());
        } else {
          assignments[index] = mat.get();
        }
      }
    }

    // find and identify unassigned pixels
    std::vector<Ccoord> unassigned_pixels;
    for (size_t i = 0; i < assignments.size(); ++i) {
      if (assignments[i] == nullptr) {
        unassigned_pixels.push_back(
          CcoordOps::get_ccoord(this->subdomain_resolutions,
                                this->subdomain_locations, i));
      }
    }

    if (unassigned_pixels.size() != 0) {
      std::stringstream err {};
      err << "The following pixels have were not assigned a material: ";
      for (auto & pixel: unassigned_pixels) {
        err << pixel << ", ";
      }
      err << "and that cannot be handled";
      throw std::runtime_error(err.str());
    }
  }

  template class CellBase<twoD, twoD>;
  template class CellBase<threeD, threeD>;

}  // muSpectre
