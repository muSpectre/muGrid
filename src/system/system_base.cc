/**
 * file   system_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Nov 2017
 *
 * @brief  Implementation for system base class
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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

#include <sstream>
#include <algorithm>

#include "system/system_base.hh"
#include "common/ccoord_operations.hh"
#include "common/iterators.hh"
#include "common/tensor_algebra.hh"


namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  SystemBase<DimS, DimM>::SystemBase(Projection_ptr projection_)
    :resolutions{projection_->get_resolutions()},
     pixels(resolutions),
     lengths{projection_->get_lengths()},
     fields{},
     F{make_field<StrainField_t>("Gradient", this->fields)},
     P{make_field<StressField_t>("Piola-Kirchhoff-1", this->fields)},
     K_ptr{nullptr}, projection{std::move(projection_)}
  { }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void SystemBase<DimS, DimM>::add_material(Material_ptr mat) {
    this->materials.push_back(std::move(mat));
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename SystemBase<DimS, DimM>::FullResponse_t
  SystemBase<DimS, DimM>::evaluate_stress_tangent(StrainField_t & grad) {
    if (this->is_initialised == false) {
      this->initialise();
    }
    //! High level compatibility checks
    if (grad.size() != this->F.size()) {
      throw std::runtime_error("Size mismatch");
    }
    if (this->K_ptr == nullptr) {
      K_ptr = &make_field<TangentField_t>("Tangent Stiffness", this->fields);
    }

    for (auto & mat: this->materials) {
      mat->compute_stresses_tangent(grad, this->P, *this->K_ptr, form);
    }
    return std::tie(this->P, *this->K_ptr);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void SystemBase<DimS, DimM>::directional_stiffness(const TangentField_t &K,
                                                     const StrainField_t &delF,
                                                     StressField_t &delP) {
    // for (auto && tup:
    //        akantu::zip(K.get_map(), delF.get_map(), delP.get_map())){
    //   auto & k = std::get<0>(tup);
    //   auto & df = std::get<1>(tup);
    //   auto & dp = std::get<2>(tup);
    //   dp = Matrices::tensmult(k, df);
    // }
    K.get_map();
    delF.get_map();
    this->projection->apply_projection(delP);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void SystemBase<DimS, DimM>::convolve(StressField_t &field) {
    this->projection->apply_projection(field);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename SystemBase<DimS, DimM>::StrainField_t &
  SystemBase<DimS, DimM>::get_strain() {
    if (this->is_initialised == false) {
      this->initialise();
    }
    return this->F;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  const typename SystemBase<DimS, DimM>::StressField_t &
  SystemBase<DimS, DimM>::get_stress() const {
    return this->P;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void SystemBase<DimS, DimM>::initialise(FFT_PlanFlags flags) {
    // check that all pixels have been assigned exactly one material
    this->check_material_coverage();
    // resize all global fields (strain, stress, etc)
    this->fields.initialise(this->resolutions);
    // initialise the projection and compute the fft plan
    this->projection->initialise(flags);
    this->is_initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void SystemBase<DimS, DimM>::initialise_materials(bool stiffness) {
    for (auto && mat: this->materials) {
      mat->initialise(stiffness);
    }
  }


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename SystemBase<DimS, DimM>::iterator
  SystemBase<DimS, DimM>::begin() {
    return this->pixels.begin();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename SystemBase<DimS, DimM>::iterator
  SystemBase<DimS, DimM>::end() {
    return this->pixels.end();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void SystemBase<DimS, DimM>::check_material_coverage() {
    auto nb_pixels = CcoordOps::get_size(this->resolutions);
    std::vector<MaterialBase<DimS, DimM>*> assignments(nb_pixels, nullptr);
    for (auto & mat: this->materials) {
      for (auto & pixel: *mat) {
        auto index = CcoordOps::get_index(this->resolutions, pixel);
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
        unassigned_pixels.push_back(CcoordOps::get_ccoord(this->resolutions, i));
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

  template class SystemBase<twoD, twoD>;
  template class SystemBase<threeD, threeD>;

}  // muSpectre
