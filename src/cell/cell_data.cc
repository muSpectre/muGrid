/**
 * @file   cell_data.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Jun 2020
 *
 * @brief  implementation for CellData member function
 *
 * Copyright © 2020 Till Junge
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

#include "cell_data.hh"

#include <libmufft/fftw_engine.hh>
#ifdef WITH_MPI
#include <libmufft/fftwmpi_engine.hh>
#endif

namespace muSpectre {
  /* ---------------------------------------------------------------------- */
  CellData::CellData(std::shared_ptr<muFFT::FFTEngineBase> engine,
                     const DynRcoord_t & domain_lengths)
      : fft_engine{engine}, domain_lengths{domain_lengths},
        fields{std::make_unique<muGrid::GlobalFieldCollection>(
            engine->get_nb_domain_grid_pts().get_dim(),
            engine->get_nb_subdomain_grid_pts(),
            engine->get_nb_subdomain_grid_pts(),
            engine->get_subdomain_locations())},
        communicator{engine->get_communicator()} {
    if (this->get_spatial_dim() != this->domain_lengths.get_dim()) {
      std::stringstream error_message{};
      error_message << "Dimension mismatch: you provided a "
                    << this->get_spatial_dim() << "-dimensional grid ("
                    << this->get_nb_domain_grid_pts() << "), but a "
                    << this->domain_lengths.get_dim()
                    << "-dimensional geometry (" << this->domain_lengths
                    << ").";
      throw CellDataError{error_message.str()};
    }
  }

  /* ---------------------------------------------------------------------- */
  std::shared_ptr<CellData>
  CellData::make(std::shared_ptr<muFFT::FFTEngineBase> engine,
                 const DynRcoord_t & domain_lengths) {
    return std::shared_ptr<CellData>{new CellData{engine, domain_lengths}};
  }

  /* ---------------------------------------------------------------------- */
  std::shared_ptr<CellData>
  CellData::make(const DynCcoord_t & nb_domain_grid_pts,
                 const DynRcoord_t & domain_lengths) {
    auto fft_ptr{std::make_shared<muFFT::FFTWEngine>(nb_domain_grid_pts)};
    return CellData::make(fft_ptr, domain_lengths);
  }

  /* ---------------------------------------------------------------------- */
#ifdef WITH_MPI
  std::shared_ptr<CellData>
  CellData::make_parallel(const DynCcoord_t & nb_domain_grid_pts,
                          const DynRcoord_t & domain_lengths,
                          const muFFT::Communicator & communicator) {
    auto fft_ptr{std::make_shared<muFFT::FFTWMPIEngine>(nb_domain_grid_pts,
                                                        communicator)};
    return CellData::make(fft_ptr, domain_lengths);
  }
#endif

  /* ---------------------------------------------------------------------- */
  muGrid::GlobalFieldCollection & CellData::get_fields() {
    return *this->fields;
  }

  /* ---------------------------------------------------------------------- */
  const muGrid::GlobalFieldCollection & CellData::get_fields() const {
    return *this->fields;
  }

  /* ---------------------------------------------------------------------- */
  const muFFT::Communicator & CellData::get_communicator() const {
    return this->communicator;
  }

  /* ---------------------------------------------------------------------- */
  auto CellData::get_domain_materials() -> DomainMaterialsMap_t & {
    return this->domain_materials;
  }

  /* ---------------------------------------------------------------------- */
  MaterialBase & CellData::add_material(Material_ptr mat) {
    if (mat->get_material_dimension() != this->get_spatial_dim()) {
      throw CellDataError("this cell class only accepts materials with the "
                          "same dimensionality as the spatial problem.");
    }
    if (this->material_dim == muGrid::Unknown) {
      this->material_dim = mat->get_material_dimension();
    } else {
      if (this->material_dim != mat->get_material_dimension()) {
        std::stringstream error_message{};
        error_message
            << "You're trying do add a material with a material dimension of "
            << mat->get_material_dimension()
            << ", but based on previously added materials, the material "
               "dimension of this problem should be "
            << this->material_dim;
        throw CellDataError{error_message.str()};
      }
    }

    auto && domain{mat->get_physics_domain()};
    this->domain_materials[domain].push_back(mat);
    return *mat;
  }

  /* ---------------------------------------------------------------------- */
  void CellData::check_material_coverage() const {
    auto nb_pixels{
        muGrid::CcoordOps::get_size(this->get_nb_subdomain_grid_pts())};
    for (auto && domain_map : this->domain_materials) {
      auto && domain{std::get<0>(domain_map)};
      auto && materials{std::get<1>(domain_map)};

      std::vector<MaterialBase *> assignments(nb_pixels, nullptr);

      for (auto & mat : materials) {
        mat->initialise();
        for (auto & index : mat->get_pixel_indices()) {
          auto & assignment{assignments.at(index)};
          if (assignment != nullptr) {
            std::stringstream err{};
            err << "Pixel " << index << "is already assigned to material '"
                << assignment->get_name()
                << "' and cannot be reassigned to material '" << mat->get_name()
                << " for Domain '" << domain << "': ";
            throw CellDataError{err.str()};
          } else {
            assignments[index] = mat.get();
          }
        }
      }

      // find and identify unassigned pixels
      std::vector<DynCcoord_t> unassigned_pixels;
      for (size_t i = 0; i < assignments.size(); ++i) {
        if (assignments[i] == nullptr) {
          unassigned_pixels.push_back(this->fields->get_ccoord(i));
        }
      }

      if (unassigned_pixels.size() != 0) {
        std::stringstream err{};
        err << "The following pixels have were not assigned a material for "
               "Domain '"
            << domain << "': ";
        for (auto & pixel : unassigned_pixels) {
          muGrid::operator<<(err, pixel) << ", ";
        }
        err << "and that cannot be handled";
        throw CellDataError{err.str()};
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  const Dim_t & CellData::get_spatial_dim() const {
    return this->get_nb_domain_grid_pts().get_dim();
  }

  /* ---------------------------------------------------------------------- */
  const Dim_t & CellData::get_material_dim() const {
    return this->material_dim;
  }

  /* ---------------------------------------------------------------------- */
  bool CellData::was_last_eval_non_linear() const {
    for (auto && domain : this->domain_materials) {
      for (auto && material : std::get<1>(domain)) {
        if (material->was_last_step_nonlinear()) {
          return true;
        }
      }
    }
    return false;
  }

  /* ---------------------------------------------------------------------- */
  const DynCcoord_t & CellData::get_nb_domain_grid_pts() const {
    return this->fft_engine->get_nb_domain_grid_pts();
  }

  /* ---------------------------------------------------------------------- */
  const DynCcoord_t & CellData::get_nb_subdomain_grid_pts() const {
    return this->fft_engine->get_nb_subdomain_grid_pts();
  }

  /* ---------------------------------------------------------------------- */
  const DynCcoord_t & CellData::get_subdomain_locations() const {
    return this->fft_engine->get_subdomain_locations();
  }

  /* ---------------------------------------------------------------------- */
  const DynRcoord_t & CellData::get_domain_lengths() const {
    return this->domain_lengths;
  }

  /* ---------------------------------------------------------------------- */
  std::shared_ptr<muFFT::FFTEngineBase> CellData::get_FFT_engine() {
    return this->fft_engine;
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & CellData::get_nb_quad_pts() const {
    return this->get_fields().get_nb_sub_pts(QuadPtTag);
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & CellData::get_nb_nodal_pts() const {
    return this->get_fields().get_nb_sub_pts(NodalPtTag);
  }

  /* ---------------------------------------------------------------------- */
  void CellData::set_nb_quad_pts(const Index_t & nb_quad_pts) {
    this->get_fields().set_nb_sub_pts(QuadPtTag, nb_quad_pts);
  }

  /* ---------------------------------------------------------------------- */
  void CellData::set_nb_nodal_pts(const Index_t & nb_nodal_pts) {
    this->get_fields().set_nb_sub_pts(NodalPtTag, nb_nodal_pts);
  }

  /* ---------------------------------------------------------------------- */
  bool CellData::has_nb_quad_pts() const {
    return this->get_fields().has_nb_sub_pts(QuadPtTag);
  }

  /* ---------------------------------------------------------------------- */
  bool CellData::has_nb_nodal_pts() const {
    return this->get_fields().has_nb_sub_pts(NodalPtTag);
  }

  /* ---------------------------------------------------------------------- */
  muGrid::FieldCollection::IndexIterable CellData::get_quad_pt_indices() const {
    return this->fields->get_sub_pt_indices(QuadPtTag);
  }

  /* ---------------------------------------------------------------------- */
  muGrid::FieldCollection::PixelIndexIterable
  CellData::get_pixel_indices() const {
    return this->fields->get_pixel_indices_fast();
  }

  /* ---------------------------------------------------------------------- */
  const muGrid::CcoordOps::DynamicPixels & CellData::get_pixels() const {
    return this->get_fields().get_pixels();
  }

  /* ---------------------------------------------------------------------- */
  void CellData::save_history_variables() {
    for (auto && domain_mat : this->domain_materials) {
      auto && materials{std::get<1>(domain_mat)};
      for (auto && mat : materials) {
        mat->save_history_variables();
      }
    }
  }

}  // namespace muSpectre
