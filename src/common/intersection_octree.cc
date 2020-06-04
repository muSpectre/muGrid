/**
 * @file   intersection_octree.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   May 2018
 *
 * @brief  Oct tree for obtaining and calculating the intersection with pixels
 *
 * Copyright © 2018 Ali Falsafi
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

#include <libmugrid/exception.hh>

#include "common/intersection_octree.hh"

using muGrid::RuntimeError;

namespace muSpectre {

  /* ---------------------------------------------------------------------- */

  template <SplitCell IsSplit>
  Node<IsSplit>::Node(const Index_t & dim, const DynRcoord_t & new_origin,
                      const DynCcoord_t & new_lengths, const Index_t & depth,
                      const Index_t & max_depth, RootNode_t & root,
                      const bool & is_root)
      : dim{dim}, root_node{root}, origin{new_origin}, Clengths{new_lengths},
        depth{depth}, is_pixel{depth == max_depth},
        children_no{(is_pixel) ? 0 : muGrid::ipow(2, this->dim)} {
    for (int i{0}; i < this->dim; i++) {
      this->Rlengths[i] = this->Clengths[i] * this->root_node.pixel_lengths[i];
    }
    if (not is_root) {
      this->check_node();
    }
  }

  /* ---------------------------------------------------------------------- */
  template <SplitCell IsSplit>
  void Node<IsSplit>::check_node() {
    switch (this->dim) {
    case twoD: {
      this->template check_node_helper<twoD>();
      break;
    }
    case threeD: {
      this->template check_node_helper<threeD>();
      break;
    }
    default: {
      std::stringstream err;
      err << "Input dimesnion is not correct. Valid dimnensions are only twoD "
             "or threeD";
      throw(RuntimeError(err.str()));
      break;
    }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <SplitCell IsSplit>
  template <Index_t DimS>
  void Node<IsSplit>::check_node_helper() {
    Real intersection_ratio = 0.0;

    // this volume should be calculated by CORKPP as the intersection volume of
    // the precipitate and the Node
    auto && intersect = PrecipitateIntersectBase<DimS>::intersect_precipitate(
        this->root_node.precipitate_vertices, this->origin.template get<DimS>(),
        this->Rlengths.template get<DimS>());

    intersection_ratio = intersect.volume_ratio;

    if (intersect.status == corkpp::IntersectionState::enclosing) {
      // pixel-box is inside precipitate
      Real pix_num = muGrid::ipow(2, (this->root_node.max_depth - this->depth));
      DynCcoord_t origin_point, pixels_number;
      for (int i{0}; i < this->dim; i++) {
        origin_point[i] =
            std::round(this->origin[i] / this->root_node.pixel_lengths[i]);
        pixels_number[i] = pix_num;
      }

      muGrid::CcoordOps::Pixels<DimS> pixels(pixels_number, origin_point);

      if (IsSplit != SplitCell::simple) {
        for (auto && pix : pixels) {
          auto pix_id{muGrid::CcoordOps::get_index<DimS>(
              this->root_node.cell.get_projection()
                  .get_nb_domain_grid_pts()
                  .template get<DimS>(),
              Ccoord_t<DimS>(), pix)};
          this->root_node.intersected_pixels.emplace_back(pix);
          this->root_node.intersected_pixels_id.emplace_back(pix_id);
          this->root_node.intersection_ratios.emplace_back(1.0);
          this->root_node.intersection_state.emplace_back(
              corkpp::IntersectionState::enclosing);
          this->root_node.intersection_normals.push_back(
              Vector_t::Zero(this->dim));
        }
      } else {
        for (auto && pix : pixels) {
          auto pix_id{muGrid::CcoordOps::get_index<DimS>(
              this->root_node.cell.get_projection()
                  .get_nb_domain_grid_pts()
                  .template get<DimS>(),
              Ccoord_t<DimS>(), pix)};
          this->root_node.intersected_pixels.emplace_back(pix);
          this->root_node.intersected_pixels_id.emplace_back(pix_id);
          this->root_node.intersection_ratios.emplace_back(1.0);
        }
      }
    } else if (intersect.status ==
                   corkpp::IntersectionState::completely_inside or
               intersect.status == corkpp::IntersectionState::intersecting) {
      if (IsSplit != SplitCell::simple) {
        this->split_node(intersection_ratio, intersect.normal_vector,
                         intersect.status);
      } else {
        this->split_node(intersection_ratio, intersect.status);
      }
    } else {
    }
  }
  /* ---------------------------------------------------------------------- */
  template <SplitCell IsSplit>
  void Node<IsSplit>::split_node(const Real & intersection_ratio,
                                 const corkpp::IntersectionState & state) {
    switch (this->dim) {
    case twoD: {
      this->template split_node_helper<twoD>(intersection_ratio, state);
      break;
    }
    case threeD: {
      this->template split_node_helper<threeD>(intersection_ratio, state);
      break;
    }
    default: {
      std::stringstream err;
      err << "Input dimesnion is not correct. Valid dimnensions are only twoD "
             "or threeD";
      throw(RuntimeError(err.str()));
      break;
    }
    }
  }
  /* ---------------------------------------------------------------------- */
  template <SplitCell IsSplit>
  template <Index_t DimS>
  void
  Node<IsSplit>::split_node_helper(const Real & intersection_ratio,
                                   const corkpp::IntersectionState & state) {
    if (this->depth == this->root_node.max_depth) {
      DynCcoord_t pixel;
      for (int i{0}; i < this->dim; i++) {
        pixel[i] =
            std::round(this->origin[i] / this->root_node.pixel_lengths[i]);
      }
      auto pix_id{muGrid::CcoordOps::get_index<DimS>(
          this->root_node.cell.get_projection()
              .get_nb_domain_grid_pts()
              .template get<DimS>(),
          Ccoord_t<DimS>(), pixel.template get<DimS>())};
      this->root_node.intersected_pixels.emplace_back(
          pixel.template get<DimS>());
      this->root_node.intersected_pixels_id.emplace_back(pix_id);
      this->root_node.intersection_ratios.emplace_back(intersection_ratio);
      this->root_node.intersection_state.emplace_back(state);
    } else {
      this->divide_node();
    }
  }

  /* ---------------------------------------------------------------------- */
  template <SplitCell IsSplit>
  void Node<IsSplit>::split_node(const Real & intersection_ratio,
                                 const corkpp::vector_t & normal_vector,
                                 const corkpp::IntersectionState & state) {
    switch (this->dim) {
    case twoD: {
      this->template split_node_helper<twoD>(intersection_ratio, normal_vector,
                                             state);
      break;
    }
    case threeD: {
      this->template split_node_helper<threeD>(intersection_ratio,
                                               normal_vector, state);
      break;
    }
    default: {
      std::stringstream err;
      err << "Input dimesnion is not correct. Valid dimnensions are only twoD "
             "or threeD";
      RuntimeError(err.str());
      break;
    }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <SplitCell IsSplit>
  template <Index_t DimS>
  void
  Node<IsSplit>::split_node_helper(const Real & intersection_ratio,
                                   const corkpp::vector_t & normal_vector,
                                   const corkpp::IntersectionState & state) {
    if (this->depth == this->root_node.max_depth) {
      DynCcoord_t pixel;
      for (int i{0}; i < this->dim; i++) {
        pixel[i] =
            std::round(this->origin[i] / this->root_node.pixel_lengths[i]);
      }
      auto pix_id{muGrid::CcoordOps::get_index<DimS>(
          this->root_node.cell.get_projection()
              .get_nb_domain_grid_pts()
              .template get<DimS>(),
          Ccoord_t<DimS>(), pixel.template get<DimS>())};
      this->root_node.intersected_pixels.emplace_back(
          pixel.template get<DimS>());

      this->root_node.intersected_pixels_id.emplace_back(pix_id);
      this->root_node.intersection_ratios.emplace_back(intersection_ratio);
      this->root_node.intersection_normals.push_back(
          normal_vector.head(this->dim));
      this->root_node.intersection_state.emplace_back(state);
    } else {
      this->divide_node();
    }
  }

  /* ---------------------------------------------------------------------- */
  template <SplitCell IsSplit>
  int RootNode<IsSplit>::compute_squared_circum_square(
      const Cell & cell) const {
    auto && max_grid_size_at{
        std::max_element(cell.get_projection().get_nb_domain_grid_pts().begin(),
                         cell.get_projection().get_nb_domain_grid_pts().end())};
    auto && max_grid_size{
        cell.get_projection().get_nb_domain_grid_pts()[std::distance(
            cell.get_projection().get_nb_domain_grid_pts().begin(),
            max_grid_size_at)]};
    // retrun the smallest muGrid::ipower of which is greater than the maximum
    // nb of grid points in all directions
    return muGrid::ipow(
        2, static_cast<size_t>(std::ceil(std::log2(max_grid_size))));
  }

  /* ---------------------------------------------------------------------- */
  template <SplitCell IsSplit>
  void Node<IsSplit>::divide_node() {
    switch (this->dim) {
    case twoD: {
      this->template divide_node_helper<twoD>();
      break;
    }
    case threeD: {
      this->template divide_node_helper<threeD>();
      break;
    }
    default: {
      std::stringstream err;
      err << "Input dimesnion is not correct. Valid dimnensions are only twoD "
             "or threeD";
      throw(RuntimeError(err.str()));
      break;
    }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <SplitCell IsSplit>
  template <Index_t DimS>
  void Node<IsSplit>::divide_node_helper() {
    DynRcoord_t new_origin;
    DynCcoord_t new_length;
    // this->children.reserve(children_no);
    muGrid::CcoordOps::Pixels<DimS> sub_nodes(
        muGrid::CcoordOps::get_cube<DimS>(Index_t{2}));
    for (auto && sub_node : sub_nodes) {
      for (int i{0}; i < this->dim; i++) {
        new_length[i] = std::round(this->Clengths[i] * 0.5);
        new_origin[i] = this->origin[i] + sub_node[i] * Rlengths[i] * 0.5;
      }
      this->children.emplace_back(this->dim, new_origin, new_length,
                                  this->depth + 1, this->root_node.max_depth,
                                  this->root_node, false);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <SplitCell IsSplit>
  DynRcoord_t RootNode<IsSplit>::make_root_origin(const Cell & cell) const {
    return DynRcoord_t(cell.get_spatial_dim());
  }

  /* ---------------------------------------------------------------------- */
  template <SplitCell IsSplit>
  Index_t RootNode<IsSplit>::make_max_resolution(const Cell & cell) const {
    auto && max_grid_size_at{
        std::max_element(cell.get_projection().get_nb_domain_grid_pts().begin(),
                         cell.get_projection().get_nb_domain_grid_pts().end())};
    return cell.get_projection().get_nb_domain_grid_pts()[std::distance(
        cell.get_projection().get_nb_domain_grid_pts().begin(),
        max_grid_size_at)];
  }

  /* ---------------------------------------------------------------------- */
  template <SplitCell IsSplit>
  Index_t RootNode<IsSplit>::make_max_depth(const Cell & cell) const {
    auto && max_grid_size_at{
        std::max_element(cell.get_projection().get_nb_domain_grid_pts().begin(),
                         cell.get_projection().get_nb_domain_grid_pts().end())};
    return std::ceil(
        std::log2(cell.get_projection().get_nb_domain_grid_pts()[std::distance(
            cell.get_projection().get_nb_domain_grid_pts().begin(),
            max_grid_size_at)]));
  }

  /* ---------------------------------------------------------------------- */
  template <SplitCell IsSplit>
  RootNode<IsSplit>::RootNode(const Cell & cell,
                              const std::vector<DynRcoord_t> & vert_precipitate)
      /*Calling parent constructing method simply by argumnets of (coordinates
         of origin, 2^depth_max in each direction, 0 ,*this)*/

      : Parent(cell.get_spatial_dim(), this->make_root_origin(cell),
               DynCcoord_t{muGrid::CcoordOps::get_cube(
                   cell.get_spatial_dim(),
                   this->compute_squared_circum_square(cell))},
               Index_t{0}, make_max_depth(cell), *this, true),
        cell{cell}, cell_length{cell.get_projection().get_domain_lengths()},
        pixel_lengths{cell.get_projection().get_pixel_lengths()},
        cell_resolution{cell.get_projection().get_nb_domain_grid_pts()},
        max_resolution{make_max_resolution(cell)}, max_depth{make_max_depth(
                                                       cell)},
        precipitate_vertices{vert_precipitate}, intersection_normals{
                                                    cell.get_spatial_dim()} {
    for (auto && vertex : vert_precipitate) {
      auto && dynvert{DynRcoord_t(vertex)};
      auto && is_vertex_inside = cell.is_point_inside(dynvert);
      if (!is_vertex_inside) {
        throw RuntimeError(
            "The precipitate introduced does not lie inside the cell");
      }
    }
    for (int i{0}; i < this->dim; i++) {
      this->Rlengths[i] = this->Clengths[i] * this->root_node.pixel_lengths[i];
    }
    this->check_node();
  }

  /* ---------------------------------------------------------------------- */
  template class RootNode<SplitCell::simple>;
  template class RootNode<SplitCell::laminate>;

  template class Node<SplitCell::simple>;
  template class Node<SplitCell::laminate>;

}  // namespace muSpectre
