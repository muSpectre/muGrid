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

#include "libmugrid/ccoord_operations.hh"
#include "common/muSpectre_common.hh"
#include "cell/cell_base.hh"
#include "materials/material_base.hh"
#include "common/intersection_octree.hh"
#include "common/intersection_volume_calculator_corkpp.hh"

#include <vector>
#include <array>
#include <algorithm>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */

  template <Dim_t DimS, SplitCell is_split>
  Node<DimS, is_split>::Node(const Rcoord & new_origin,
                             const Ccoord & new_lengths, int depth,
                             RootNode_t & root, bool is_root)
      : root_node(root), origin(new_origin), Clengths(new_lengths),
        depth(depth), is_pixel((depth == root.max_depth)),
        children_no(((is_pixel) ? 0 : pow(2, DimS))) {
    for (int i = 0; i < DimS; i++) {
      this->Rlengths[i] = this->Clengths[i] * this->root_node.pixel_lengths[i];
    }
    if (not is_root) {
      this->check_node();
    }
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, SplitCell is_split>
  void Node<DimS, is_split>::check_node() {
    Real intersection_ratio = 0.0;

    // this volume should be calculated by CGAL as the intersection volume of
    // the precipitate and the Node
    auto && intersect = PrecipitateIntersectBase<DimS>::intersect_precipitate(
        this->root_node.precipitate_vertices, this->origin, this->Rlengths);

    intersection_ratio = intersect.volume_ratio;

    if (intersect.status == corkpp::IntersectionState::enclosing) {
      // pixel-box is inside precipitate
      Real pix_num = pow(2, (this->root_node.max_depth - this->depth));
      Ccoord origin_point, pixels_number;
      for (int i = 0; i < DimS; i++) {
        origin_point[i] =
            std::round(this->origin[i] / this->root_node.pixel_lengths[i]);
        pixels_number[i] = pix_num;
      }

      muGrid::CcoordOps::Pixels<DimS> pixels(pixels_number, origin_point);

      if (is_split != SplitCell::simple) {
        for (auto && pix : pixels) {
          this->root_node.intersected_pixels.push_back(pix);
          this->root_node.intersection_ratios.push_back(1.0);
          this->root_node.intersection_state.push_back(
              corkpp::IntersectionState::enclosing);
          this->root_node.intersection_normals.push_back(Vector_t::Zero());
        }
      } else {
        for (auto && pix : pixels) {
          this->root_node.intersected_pixels.push_back(pix);
          this->root_node.intersection_ratios.push_back(1.0);
        }
      }
    } else if (intersect.status ==
                   corkpp::IntersectionState::completely_inside or
               intersect.status == corkpp::IntersectionState::intersecting) {
      if (is_split != SplitCell::simple) {
        this->split_node(intersection_ratio, intersect.normal_vector,
                         intersect.status);
      } else {
        this->split_node(intersection_ratio, intersect.status);
      }
    } else {
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, SplitCell is_split>
  void Node<DimS, is_split>::split_node(Real intersection_ratio,
                                        corkpp::IntersectionState state) {
    if (this->depth == this->root_node.max_depth) {
      Ccoord pixel;
      for (int i = 0; i < DimS; i++) {
        pixel[i] =
            std::round(this->origin[i] / this->root_node.pixel_lengths[i]);
      }

      this->root_node.intersected_pixels.push_back(pixel);
      this->root_node.intersection_ratios.push_back(intersection_ratio);
      this->root_node.intersection_state.push_back(state);
    } else {
      this->divide_node();
    }
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, SplitCell is_split>
  void Node<DimS, is_split>::split_node(Real intersection_ratio,
                                        corkpp::vector_t normal_vector,
                                        corkpp::IntersectionState state) {
    if (this->depth == this->root_node.max_depth) {
      Ccoord pixel;
      for (int i = 0; i < DimS; i++) {
        pixel[i] =
            std::round(this->origin[i] / this->root_node.pixel_lengths[i]);
      }
      this->root_node.intersected_pixels.push_back(pixel);
      this->root_node.intersection_ratios.push_back(intersection_ratio);
      this->root_node.intersection_normals.push_back(normal_vector.head(DimS));
      this->root_node.intersection_state.push_back(state);
    } else {
      this->divide_node();
    }
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, SplitCell is_split>
  void Node<DimS, is_split>::divide_node() {
    Rcoord new_origin;
    Ccoord new_length;
    // this->children.reserve(children_no);
    muGrid::CcoordOps::Pixels<DimS> sub_nodes(
        muGrid::CcoordOps::get_cube<DimS>(2));
    for (auto && sub_node : sub_nodes) {
      for (int i = 0; i < DimS; i++) {
        new_length[i] = std::round(this->Clengths[i] * 0.5);
        new_origin[i] = this->origin[i] + sub_node[i] * Rlengths[i] * 0.5;
      }
      this->children.emplace_back(new_origin, new_length, this->depth + 1,
                                  this->root_node, false);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, SplitCell is_split>
  RootNode<DimS, is_split>::RootNode(CellBase<DimS, DimS> & cell,
                                     std::vector<Rcoord> vert_precipitate)
      /*Calling parent constructing method simply by argumnets of (coordinates
         of origin, 2^depth_max in each direction, 0 ,*this)*/

      : Node<DimS, is_split>(
            Rcoord{},
            muGrid::CcoordOps::get_cube<DimS, Dim_t>(pow(
                2,
                std::ceil(
                    std::log2(cell.get_nb_domain_grid_pts().at(std::distance(
                        std::max_element(cell.get_nb_domain_grid_pts().begin(),
                                         cell.get_nb_domain_grid_pts().end()),
                        cell.get_nb_domain_grid_pts().begin())))))),
            0, *this, true),

        cell(cell), cell_length(cell.get_domain_lengths()),
        pixel_lengths(cell.get_pixel_lengths()),
        cell_resolution(cell.get_nb_domain_grid_pts()),
        max_resolution(this->cell_resolution.at(
            std::distance(std::max_element(this->cell_resolution.begin(),
                                           this->cell_resolution.end()),
                          this->cell_resolution.begin()))),
        max_depth(std::ceil(std::log2(this->max_resolution))),
        precipitate_vertices(vert_precipitate) {
    for (auto && vertice : vert_precipitate) {
      auto && is_vertice_inside = cell.is_inside(vertice);
      if (!is_vertice_inside) {
        throw std::runtime_error(
            "The precipitate introduced does not lie inside the cell");
      }
    }
    for (int i = 0; i < DimS; i++) {
      this->Rlengths[i] = this->Clengths[i] * this->root_node.pixel_lengths[i];
    }
    this->check_node();
  }

  /* ---------------------------------------------------------------------- */
  template class RootNode<threeD, SplitCell::simple>;
  template class RootNode<threeD, SplitCell::laminate>;
  template class RootNode<twoD, SplitCell::simple>;
  template class RootNode<twoD, SplitCell::laminate>;

  template class Node<threeD, SplitCell::simple>;
  template class Node<threeD, SplitCell::laminate>;
  template class Node<twoD, SplitCell::simple>;
  template class Node<twoD, SplitCell::laminate>;

}  // namespace muSpectre
