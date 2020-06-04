/**
 * @file   intersection_volume_calculator_corkpp.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   04 June 2018
 *
 * @brief  Calculation of the intersection volume of percipitates and pixles
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

#ifndef SRC_COMMON_INTERSECTION_VOLUME_CALCULATOR_CORKPP_HH_
#define SRC_COMMON_INTERSECTION_VOLUME_CALCULATOR_CORKPP_HH_

#include "cork_interface.hh"

#include "libmugrid/grid_common.hh"

#include <vector>
#include <fstream>
#include <math.h>

namespace muSpectre {

  using muGrid::Ccoord_t;
  using muGrid::Index_t;
  using muGrid::DynCcoord_t;
  using muGrid::DynRcoord_t;
  using muGrid::Rcoord_t;
  using muGrid::Real;

  template <Index_t DimS>
  class Correction {
   public:
    static Rcoord_t<3> correct_origin(const Rcoord_t<DimS> & array);
    static Rcoord_t<3> correct_length(const Rcoord_t<DimS> & array);
    static std::vector<Rcoord_t<3>>
    correct_vector(const std::vector<Rcoord_t<DimS>> & vector);
  };

  template <>
  class Correction<3> {
   public:
    static Rcoord_t<3> correct_origin(const Rcoord_t<3> & array) {
      return array;
    }
    static Rcoord_t<3> correct_length(const Rcoord_t<3> & array) {
      return array;
    }
    static std::vector<Rcoord_t<3>>
    correct_vector(const std::vector<Rcoord_t<3>> & vertices) {
      // std::vector<Rcoord_t<3>> corrected_convex_poly_vertices;
      return vertices;
    }
  };

  template <>
  class Correction<2> {
   public:
    static std::vector<Rcoord_t<3>>
    correct_vector(const std::vector<Rcoord_t<2>> & vertices) {
      std::vector<corkpp::point_t> corrected_convex_poly_vertices;
      for (auto && vertice : vertices) {
        corrected_convex_poly_vertices.push_back({vertice[0], vertice[1], 0.0});
      }
      for (auto && vertice : vertices) {
        corrected_convex_poly_vertices.push_back({vertice[0], vertice[1], 1.0});
      }
      return corrected_convex_poly_vertices;
    }
    static Rcoord_t<3> correct_origin(const Rcoord_t<2> & array) {
      return Rcoord_t<3>{array[0], array[1], 0.0};
    }

    static Rcoord_t<3> correct_length(const Rcoord_t<2> & array) {
      return Rcoord_t<3>{array[0], array[1], 1.0};
    }
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  class PrecipitateIntersectBase {
   public:
    static std::tuple<std::vector<corkpp::point_t>,
                      std::vector<corkpp::point_t>>
    correct_dimension(const std::vector<Rcoord_t<DimS>> & convex_poly_vertices,
                      const Rcoord_t<DimS> & origin,
                      const Rcoord_t<DimS> & lengths);

    //! this function is the palce that CORK is called to analyze the geometry
    //! and make the intersection of the precipitate with a grid
    static corkpp::VolNormStateIntersection
    intersect_precipitate(const std::vector<DynRcoord_t> & convex_poly_vertices,
                          const Rcoord_t<DimS> & origin,
                          const Rcoord_t<DimS> & lengths);
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  std::tuple<std::vector<corkpp::point_t>, std::vector<corkpp::point_t>>
  PrecipitateIntersectBase<DimS>::correct_dimension(
      const std::vector<Rcoord_t<DimS>> & convex_poly_vertices,
      const Rcoord_t<DimS> & origin, const Rcoord_t<DimS> & lengths) {
    std::vector<corkpp::point_t> corrected_convex_poly_vertices(
        Correction<DimS>::correct_vector(convex_poly_vertices));
    corkpp::point_t corrected_origin(Correction<DimS>::correct_origin(origin));
    corkpp::point_t corrected_lengths(
        Correction<DimS>::correct_length(lengths));
    std::vector<corkpp::point_t> vertices_pixel{
        corkpp::cube_vertice_maker(corrected_origin, corrected_lengths)};
    return std::make_tuple(corrected_convex_poly_vertices, vertices_pixel);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  corkpp::VolNormStateIntersection
  PrecipitateIntersectBase<DimS>::intersect_precipitate(
      const std::vector<DynRcoord_t> & convex_poly_vertices,
      const Rcoord_t<DimS> & origin, const Rcoord_t<DimS> & lengths) {
    std::vector<Rcoord_t<DimS>> converted_poly_vertices;
    for (auto && poly_vert : convex_poly_vertices) {
      converted_poly_vertices.push_back(poly_vert.get<DimS>());
    }

    auto && precipitate_pixel{
        correct_dimension(converted_poly_vertices, origin, lengths)};
    auto && precipitate{std::get<0>(precipitate_pixel)};
    auto && pixel{std::get<1>(precipitate_pixel)};
    auto && intersect{corkpp::calculate_intersection_volume_normal_state(
        precipitate, pixel, DimS)};
    return std::move(intersect);
  }

  /* ---------------------------------------------------------------------- */

}  // namespace muSpectre
#endif  // SRC_COMMON_INTERSECTION_VOLUME_CALCULATOR_CORKPP_HH_
