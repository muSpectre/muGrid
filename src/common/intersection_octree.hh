/**
 * @file   intersection_octree.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   May 2018
 *
 * @brief  octree algorithm employed to accelerate precipitate pixel assignment
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

#ifndef SRC_COMMON_INTERSECTION_OCTREE_HH_
#define SRC_COMMON_INTERSECTION_OCTREE_HH_

#include <vector>

#include "libmugrid/ccoord_operations.hh"
#include "common/muSpectre_common.hh"
#include "cell/cell_base.hh"
#include "materials/material_base.hh"
#include "common/intersection_volume_calculator_corkpp.hh"

namespace muSpectre {
  template <Dim_t DimS, SplitCell is_split>
  class RootNode;

  // this class is defined to be used instead of std::Vector<Eigen::vector>
  template <Dim_t DimS>
  class Vectors_t {
    using Vector_t = Eigen::Matrix<Real, DimS, 1>;

   public:
    Eigen::Map<Vector_t> operator[](Dim_t id) {
      return Eigen::Map<Vector_t>(&data[DimS * id]);
    }

    inline void push_back(Vector_t vector) {
      for (int i = 0; i < DimS; ++i) {
        data.push_back(vector[i]);
      }
    }
    /* --------------------------------------------------------------------- */
    // be careful that the index increments with DimS instead of one two make
    // the end() function easy to write via the constructor
    class iterator {
     public:
      using value_type = Eigen::Map<Vector_t>;
      //! constructor
      explicit iterator(Vectors_t & data, bool begin = true)
          : vectors{data}, index{begin ? 0 : this->vectors.data.size()} {}
      // deconstructor
      virtual ~iterator() = default;
      //! dereferencing
      value_type operator*() {
        auto ret_val = value_type(&this->vectors.data[this->index]);
        return ret_val;
      }
      //! pre-increment
      auto operator++() -> iterator & {
        this->index += DimS;
        return *this;
      }
      auto operator--() -> iterator & {
        this->index -= DimS;
        return *this;
      }
      //! inequality
      inline bool operator!=(const iterator & other) {
        return (this->index != other.index);
      }
      //! equality
      inline bool operator==(const iterator & other) const {
        return (this->index == other.index);
      }

     private:
      Vectors_t & vectors;
      size_t index;
    };
    /* --------------------------------------------------------------------- */

    inline iterator begin() { return iterator(*this); }
    inline iterator end() { return iterator(*this, false); }
    size_t size() const { return std::floor(this->data.size() / DimS); }
    /* --------------------------------------------------------------------- */
   protected:
    std::vector<Real> data{};
  };

  template <Dim_t DimS, SplitCell is_split>
  class Node {
   public:
    using Rcoord = Rcoord_t<DimS>;  //!< physical coordinates type
    using Ccoord = Ccoord_t<DimS>;  //!< cell coordinates type
    using RootNode_t = RootNode<DimS, is_split>;
    using Vector_t = Eigen::Matrix<Real, DimS, 1>;
    // using Vectors_t = std::vector<Vector_t>;
    //! Default constructor
    Node() = delete;

    // Construct by origin, lenghts, and depth
    Node(const Rcoord & new_origin, const Ccoord & new_lenghts, int depth,
         RootNode_t & root, bool is_root);

    // This function is put here as a comment, so we could add a octree by
    // giving a cell as its input that seems to be of possible use in the future
    // Construct by cell
    // Node(const CellBase<DimS, DimS>& cell, RootNode_t& root);

    //! Copy constructor
    Node(const Node & other) = delete;

    //! Move constructor
    Node(Node && other) = default;

    //! Destructor
    virtual ~Node() = default;

    // This function checks the status of the node and orders its devision into
    // smaller nodes or asssign material to it
    virtual void check_node();

    // This function gives the ratio of the node which happens to be inside the
    // precipitate and assign materila to it if node is a pixel or divide it
    // furhter if it is not
    void split_node(Real ratio, corkpp::IntersectionState state);
    void split_node(Real intersection_ratio, corkpp::vector_t normal_vector,
                    corkpp::IntersectionState state);

    // this function constructs children of a node
    void divide_node();

   protected:
    //
    RootNode_t & root_node;
    Rcoord origin, Rlengths{};
    Ccoord Clengths{};
    int depth;
    bool is_pixel;
    int children_no;
    std::vector<Node> children{};
  };

  template <Dim_t DimS, SplitCell is_split>
  class RootNode : public Node<DimS, is_split> {
    friend class Node<DimS, is_split>;

   public:
    using Rcoord = Rcoord_t<DimS>;        //!< physical coordinates type
    using Ccoord = Ccoord_t<DimS>;        //!< cell coordinates type
    using Parent = Node<DimS, is_split>;  //!< base class
    using Vector_t = typename Parent::Vector_t;
    // using Vectors_t = typename Parent::Vectors_t;
    //! Default Constructor
    RootNode() = delete;

    //! Constructing a root node for a cell and a preticipate inside that cell
    RootNode(CellBase<DimS, DimS> & cell, std::vector<Rcoord> vert_precipitate);

    //! Copy constructor
    RootNode(const RootNode & other) = delete;

    //! Move constructor
    RootNode(RootNode && other) = default;

    //! Destructor
    ~RootNode() = default;

    // returns the pixels which have intersection raio with the preipitate
    inline auto get_intersected_pixels() -> std::vector<Ccoord> {
      return this->intersected_pixels;
    }

    // return the intersection ratios of corresponding to the pixels returned by
    // get_intersected_pixels()
    inline auto get_intersection_ratios() -> std::vector<Real> {
      return this->intersection_ratios;
    }

    // return the normal vector of intersection surface of corresponding to the
    // pixels returned by get_intersected_pixels()
    inline auto get_intersection_normals() -> Vectors_t<DimS> {
      return this->intersection_normals;
    }

    inline auto get_intersection_status()
        -> std::vector<corkpp::IntersectionState> {
      return this->intersection_state;
    }

    // checking rootnode:
    void check_root_node();

   protected:
    CellBase<DimS, DimS> & cell;
    Rcoord cell_length, pixel_lengths;
    Ccoord cell_resolution;
    int max_resolution;
    int max_depth;
    std::vector<Rcoord> precipitate_vertices{};
    std::vector<Ccoord> intersected_pixels{};
    std::vector<Real> intersection_ratios{};
    Vectors_t<DimS> intersection_normals{};
    std::vector<corkpp::IntersectionState> intersection_state{};
  };

}  // namespace muSpectre

#endif  // SRC_COMMON_INTERSECTION_OCTREE_HH_
