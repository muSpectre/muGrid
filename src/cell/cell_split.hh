/**
 * @file   cell_split.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   19 Apr 2018
 *
 * @brief Base class representing a unit cell able to handle
 *        split material assignments
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

#ifndef SRC_CELL_CELL_SPLIT_HH_
#define SRC_CELL_CELL_SPLIT_HH_

#include "common/muSpectre_common.hh"
#include "libmugrid/ccoord_operations.hh"
#include "libmugrid/field.hh"
#include "materials/material_base.hh"
#include "projection/projection_base.hh"
#include "cell/cell_traits.hh"
#include "cell/cell_base.hh"
#include "common/intersection_octree.hh"

#include <vector>
#include <memory>
#include <tuple>
#include <functional>
namespace muSpectre {

  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  /* This class handles the cells that has
     splitly assigned material to their pixels */
  template <Dim_t DimS, Dim_t DimM = DimS>
  class CellSplit : public CellBase<DimS, DimM> {
    friend class CellBase<DimS, DimM>;

   public:
    using Parent = CellBase<DimS, DimM>;  //!< base class
    //! global field collection
    using FieldCollection_t = muGrid::GlobalFieldCollection<DimS>;
    using Projection_t = ProjectionBase<DimS, DimM>;
    //! projections handled through `std::unique_ptr`s
    using Projection_ptr = std::unique_ptr<Projection_t>;
    using StrainField_t =
        muGrid::TensorField<FieldCollection_t, Real, secondOrder, DimM>;
    //! expected type for stress fields
    using StressField_t =
        muGrid::TensorField<FieldCollection_t, Real, secondOrder, DimM>;
    //! expected type for tangent stiffness fields
    using TangentField_t =
        muGrid::TensorField<FieldCollection_t, Real, fourthOrder, DimM>;
    //! combined stress and tangent field
    using FullResponse_t =
        std::tuple<const StressField_t &, const TangentField_t &>;

    //! ref to constant vector
    using ConstVector_ref = typename Parent::ConstVector_ref;

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
    CellSplit & operator=(CellSplit && other) = default;

    /**
     *completes the assignmnet of material with a specific material so
     *all the under-assigned pixels would be assigned to a material.
     */
    //
    void complete_material_assignment(MaterialBase<DimS, DimM> & material);

    // returns the assigend ratios to each pixel
    std::vector<Real> get_assigned_ratios();

    //
    // template<class MaterialType>
    void make_automatic_precipitate_split_pixels(
        std::vector<Rcoord_t<DimS>> preciptiate_vertices,
        MaterialBase<DimS, DimM> & material);
    //
    std::vector<Real> get_unassigned_ratios_incomplete_pixels();
    std::vector<int> get_index_incomplete_pixels();
    std::vector<Ccoord_t<DimS>> get_unassigned_pixels();

    class IncompletePixels {
     public:
      //! constructor
      explicit IncompletePixels(CellSplit<DimS, DimM> & cell);
      //! copy constructor
      IncompletePixels(const IncompletePixels & other) = default;
      //! move constructor
      IncompletePixels(IncompletePixels & other) = default;
      // Deconstructor
      virtual ~IncompletePixels() = default;

      //! iterator type over all incompletetedly assigned pixel's
      class iterator {
       public:
        using value_type =
            std::tuple<Ccoord_t<DimS>, Real>;       //!< stl conformance
        using const_value_type = const value_type;  //!< stl conformance
        using pointer = value_type *;               //!< stl conformance
        using difference_type = std::ptrdiff_t;     //!< stl conformance
        using iterator_category =
            std::forward_iterator_tag;  //!< stl conformance
        using reference = value_type;   //!< stl conformance
        //! constructor
        explicit iterator(const IncompletePixels & pixels, bool begin = true);
        // deconstructor
        virtual ~iterator() = default;
        //! dereferencing
        auto operator*() -> value_type const;
        //! pre-increment
        auto operator++() -> iterator &;
        //! inequality
        auto operator!=(const iterator & other) -> bool;
        //! equality
        inline bool operator==(const iterator & other) const;

       private:
        const IncompletePixels & incomplete_pixels;
        size_t index;
      };
      inline iterator begin() const { return iterator(*this); }
      //! stl conformance
      inline iterator end() const { return iterator(*this, false); }
      //! stl conformance
      inline size_t size() const {
        return muGrid::CcoordOps::get_size(this->cell.get_nb_domain_grid_pts());
      }

     private:
      CellSplit<DimS, DimM> & cell;
      std::vector<Real> incomplete_assigned_ratios;
      std::vector<int> index_incomplete_pixels;
    };

    auto make_incomplete_pixels() -> IncompletePixels;

   protected:
    void check_material_coverage() final;
    // this function make the initial magnitude of the
    // stress and stiffness tensors to zero
    void set_p_k_zero();
    // full resppnse is composed of the stresses and tangent matrix
    FullResponse_t evaluate_stress_tangent(StrainField_t & F) override;
    std::array<ConstVector_ref, 2> evaluate_stress_tangent() override;
    ConstVector_ref evaluate_stress() override;

   private:
  };

}  // namespace muSpectre

#endif  // SRC_CELL_CELL_SPLIT_HH_
