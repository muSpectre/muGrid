/**
 * @file   cell_split.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   10 Dec 2019
 *
 * @brief Base class representing a unit cell able to handle
 *        split material assignments
 *
 * Copyright © 2017 Till Junge
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

#ifndef SRC_CELL_CELL_SPLIT_HH_
#define SRC_CELL_CELL_SPLIT_HH_

#include "cell/cell.hh"

#include "common/muSpectre_common.hh"
#include "common/intersection_octree.hh"
#include "materials/material_base.hh"
#include "projection/projection_base.hh"

#include "libmugrid/ccoord_operations.hh"
#include "libmugrid/field.hh"

#include <vector>
#include <memory>
#include <tuple>
#include <functional>
#include <sstream>
#include <algorithm>

namespace muSpectre {

  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  /* This class handles the cells that has
     splitly assigned material to their pixels */
  class CellSplit : public Cell {
    friend class Cell;

   public:
    using Parent = Cell;  //!< base class

    //! projections handled through `std::unique_ptr`s
    using Projection_ptr = std::unique_ptr<ProjectionBase>;

    //! combined stress and tangent field
    using FullResponse_t =
        std::tuple<const muGrid::RealField &, const muGrid::RealField &>;

    //! Default constructor
    CellSplit() = delete;

    //! constructor using sizes and resolution
    explicit CellSplit(Projection_ptr projection);

    //! Copy constructor
    CellSplit(const CellSplit & other) = delete;

    //! Move constructor
    CellSplit(CellSplit && other) = default;

    //! Destructor
    virtual ~CellSplit() = default;

    //! Copy assignment operator
    CellSplit & operator=(const CellSplit & other) = delete;

    //! Move assignment operator
    CellSplit & operator=(CellSplit && other) = delete;

    /**
     * add a new material to the cell
     */
    MaterialBase & add_material(Material_ptr mat) final;

    /**
     *completes the assignmnet of material with a specific material so
     *all the under-assigned pixels would be assigned to a material.
     */

    void complete_material_assignment(MaterialBase & material);

    // returns the assigend ratios to each pixel
    std::vector<Real> get_assigned_ratios();

    // Assigns pixels according to the coordiantes of the
    // precipitate
    void make_automatic_precipitate_split_pixels(
        const std::vector<DynRcoord_t> & preciptiate_vertices,
        MaterialBase & material);

    // Returns the unassigend portion of the pixels whose material
    // assignemnt are not complete
    std::vector<Real> get_unassigned_ratios_incomplete_pixels() const;

    // This function returns the index of the pixels whose material
    // assignemnt are not complete
    std::vector<Index_t> get_index_incomplete_pixels() const;

    // This function returns the Ccoord of the pixels whose material
    // assignemnt are not complete
    std::vector<DynCcoord_t> get_unassigned_pixels();

    // this class is designed to handle the pixels with incomplete material
    // assignment and itereating over them
    class IncompletePixels {
     public:
      //! constructor
      explicit IncompletePixels(const CellSplit & cell);
      //! copy constructor
      IncompletePixels(const IncompletePixels & other) = default;
      //! move constructor
      IncompletePixels(IncompletePixels & other) = default;
      // Deconstructor
      virtual ~IncompletePixels() = default;

      //! iterator type over all incompletetedly assigned pixel's
      class iterator {
       public:
        using value_type = std::tuple<DynCcoord_t, Real>;  //!< stl conformance

        //! constructor
        iterator(const IncompletePixels & pixels, Index_t dim,
                 bool begin = true);

        // deconstructor
        virtual ~iterator() = default;

        //! dereferencing
        value_type operator*() const;

        template <Index_t DimS>
        value_type deref_helper() const;

        //! pre-increment
        iterator & operator++();

        //! inequality
        bool operator!=(const iterator & other);

        //! equality
        inline bool operator==(const iterator & other) const;

       protected:
        const IncompletePixels & incomplete_pixels;
        Index_t dim;
        size_t index;
      };

      //! stl conformance
      inline iterator begin() const {
        return iterator(*this, this->cell.get_spatial_dim());
      }

      //! stl conformance
      inline iterator end() const {
        return iterator(*this, this->cell.get_spatial_dim(), false);
      }

      //! stl conformance
      inline size_t size() const {
        return muGrid::CcoordOps::get_size(
            this->cell.projection->get_nb_subdomain_grid_pts());
      }

     protected:
      const CellSplit & cell;
      std::vector<Real> incomplete_assigned_ratios;
      std::vector<Index_t> index_incomplete_pixels;
    };

    IncompletePixels make_incomplete_pixels();

    //! makes sure every pixel has been assigned to materials whose ratios add
    //! up to 1.0
    void check_material_coverage() const final;

    // full resppnse is composed of the stresses and tangent matrix
    const muGrid::RealField & evaluate_stress() final;

    // stress and the tangent of the response of the cell_split
    std::tuple<const muGrid::RealField &, const muGrid::RealField &>
    evaluate_stress_tangent() final;

   protected:
    // this function make the initial magnitude of the
    // stress and stiffness tensors to zero
    void set_p_k_zero();

   private:
  };

}  // namespace muSpectre

#endif  // SRC_CELL_CELL_SPLIT_HH_
