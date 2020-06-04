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

#include "common/muSpectre_common.hh"
#include "cell/cell.hh"
#include "materials/material_base.hh"
#include "common/intersection_volume_calculator_corkpp.hh"

#include "libmugrid/ccoord_operations.hh"

#include <vector>
#include <array>
#include <algorithm>
namespace muSpectre {
  template <SplitCell IsSplit>
  class RootNode;

  // this class is defined to be used instead of std::Vector<Eigen::vector>
  class Vectors_t {
    using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

   public:
    //! constructor
    explicit Vectors_t(const Index_t & dim) : dim{dim} {}

    //! constructor
    Vectors_t(const std::vector<Real> & data, const Index_t & dim)
        : data{data}, dim{dim} {}

    //! access operator:
    Eigen::Map<const Vector_t> operator[](const Index_t & id) const {
      return Eigen::Map<const Vector_t>(&this->data.data()[this->dim * id],
                                        this->dim, 1);
    }

    //! access operator:
    Eigen::Map<Vector_t> operator[](const Index_t & id) {
      return Eigen::Map<Vector_t>(&this->data.data()[this->dim * id], this->dim,
                                  1);
    }

    //! access to staic sized map of the vectors:
    template <Index_t DimS>
    Eigen::Map<Eigen::Matrix<Real, DimS, 1>> at(const Index_t & id) {
      return Eigen::Map<Eigen::Matrix<Real, DimS, 1>>(
          &this->data.data()[DimS * id], this->dim, 1);
    }

    //! push back for adding new vector to the data of the class
    inline void push_back(const Vector_t & vector) {
      for (int i{0}; i < this->dim; ++i) {
        this->data.push_back(vector[i]);
      }
    }

    //! push back for adding new vector to the data of the class
    inline void push_back(const Eigen::Map<Vector_t, 0> & vector) {
      for (int i{0}; i < this->dim; ++i) {
        this->data.push_back(vector[i]);
      }
    }

    //! push back for adding new vector to the data of the class
    inline void push_back(const Eigen::Map<const Vector_t, 0> & vector) {
      for (int i{0}; i < this->dim; ++i) {
        this->data.push_back(vector[i]);
      }
    }

    //! push back for adding new vector from DynRcoord
    inline void push_back(const DynRcoord_t & vector) {
      for (int i{0}; i < this->dim; ++i) {
        this->data.push_back(vector[i]);
      }
    }

    inline std::vector<Real> get_a_vector(const Index_t & id) {
      std::vector<Real> ret(this->dim);
      for (Index_t i{id * dim}; i < this->dim * (id + 1); ++i) {
        ret.push_back(this->data[i]);
      }
      return ret;
    }

    inline const Index_t & get_dim() { return this->dim; }

    /* --------------------------------------------------------------------- */
    // be careful that the index increments with DimS instead of one make
    // the end() function easy to write via the constructor
    class iterator {
     public:
      using value_type = Eigen::Map<Vector_t>;
      using value_type_const = Eigen::Map<const Vector_t>;
      //! constructor
      explicit iterator(const Vectors_t & data, const Index_t & dim,
                        bool begin = true)
          : vectors(data), dim{dim}, index{begin ? 0
                                                 : this->vectors.data.size()} {}
      // deconstructor
      virtual ~iterator() = default;

      //! dereferencing
      value_type_const operator*() const { return this->vectors[this->index]; }

      //! pre-increment
      iterator & operator++() {
        this->index++;
        return *this;
      }

      // !decremment
      iterator & operator--() {
        this->index--;
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

     protected:
      const Vectors_t & vectors;
      Index_t dim;
      size_t index;
    };

    /* --------------------------------------------------------------------- */

    inline iterator begin() { return iterator(*this, this->dim, true); }
    inline iterator end() { return iterator(*this, this->dim, false); }
    size_t size() const {
      return static_cast<size_t>(this->data.size() / this->dim);
    }

    /* --------------------------------------------------------------------- */
   protected:
    std::vector<Real> data{};
    Index_t dim;
  };

  template <SplitCell IsSplit>
  class Node {
   public:
    using RootNode_t = RootNode<IsSplit>;
    using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    //! Default constructor
    Node() = delete;

    // Construct by origin, lenghts, and depth
    Node(const Index_t & dim, const DynRcoord_t & new_origin,
         const DynCcoord_t & new_lenghts, const Index_t & depth,
         const Index_t & max_depth, RootNode_t & root, const bool & is_root);

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
    template <Index_t DimS>
    void check_node_helper();

    void check_node();

    // This function gives the ratio of the node which happens to be inside the
    // precipitate and assign materila to it if node is a pixel or divide it
    // furhter if it is not
    template <Index_t DimS>
    void split_node_helper(const Real & ratio,
                           const corkpp::IntersectionState & state);

    template <Index_t DimS>
    void split_node_helper(const Real & intersection_ratio,
                           const corkpp::vector_t & normal_vector,
                           const corkpp::IntersectionState & state);

    void split_node(const Real & ratio,
                    const corkpp::IntersectionState & state);
    void split_node(const Real & intersection_ratio,
                    const corkpp::vector_t & normal_vector,
                    const corkpp::IntersectionState & state);

    template <Index_t DimS>
    void divide_node_helper();
    // this function constructs children of a node
    void divide_node();

   protected:
    //
    Index_t dim;
    RootNode_t & root_node;
    DynRcoord_t origin, Rlengths{};
    DynCcoord_t Clengths{};
    Index_t depth;
    bool is_pixel;
    Index_t children_no;
    std::vector<Node> children{};
  };

  template <SplitCell IsSplit>
  class RootNode : public Node<IsSplit> {
    friend class Node<IsSplit>;

   public:
    using Parent = Node<IsSplit>;  //!< base class
    using Vector_t = typename Parent::Vector_t;
    // using Vectors_t = typename Parent::Vectors_t;
    //! Default Constructor
    RootNode() = delete;

    //! Constructing a root node for a cell and a preticipate inside that cell
    RootNode(const Cell & cell,
             const std::vector<DynRcoord_t> & vert_precipitate);

    //! Copy constructor
    RootNode(const RootNode & other) = delete;

    //! Move constructor
    RootNode(RootNode && other) = default;

    //! Destructor
    ~RootNode() = default;

    // returns the pixels which have intersection raio with the preipitate
    inline std::vector<DynCcoord_t> get_intersected_pixels() {
      return this->intersected_pixels;
    }

    // returns the index of the pixels which have intersection raio with the
    // preipitate
    inline std::vector<size_t> get_intersected_pixels_id() {
      return this->intersected_pixels_id;
    }

    // return the intersection ratios of corresponding to the pixels returned by
    // get_intersected_pixels()
    inline std::vector<Real> get_intersection_ratios() {
      return this->intersection_ratios;
    }

    // return the normal vector of intersection surface of corresponding to the
    // pixels returned by get_intersected_pixels()
    inline Vectors_t get_intersection_normals() {
      return this->intersection_normals;
    }

    inline std::vector<corkpp::IntersectionState> get_intersection_status() {
      return this->intersection_state;
    }

    // Returns the maximum of the nb_grid_pts in all directions
    Index_t make_max_resolution(const Cell & cell) const;

    // Retruns the maximum depth of the branches in the OctTree
    Index_t make_max_depth(const Cell & cell) const;

    // checking rootnode:
    void check_root_node();

    // computes the smallest muGrid::ipower of which is greater than the maximum
    // nb of grid points in all directions which is the size of the OctTree for
    // checking intersections
    int compute_squared_circum_square(const Cell & cell) const;

    // make Rcoord of the origin of the root node
    DynRcoord_t make_root_origin(const Cell & cell) const;

   protected:
    const Cell & cell;            //! the cell to be intersected
    DynRcoord_t cell_length;      //! The Real size of the cell
    DynRcoord_t pixel_lengths;    //! The Real size of each pixel
    DynCcoord_t cell_resolution;  //! The nb_grid_pts for the
    Index_t
        max_resolution;  //! The maximum of the nb_grid_pts in all directions
    Index_t max_depth;   //! The maximum depth of the branches in the OctTree
    std::vector<DynRcoord_t>
        precipitate_vertices{};  //! The coordinates of the vertices of the
                                 //! perticpiate
    std::vector<DynCcoord_t>
        intersected_pixels{};  //! The pixels of the cell which intersect with
                               //! the percipitate
    std::vector<size_t>
        intersected_pixels_id{};  //! The index of the intersecting pixels
    std::vector<Real>
        intersection_ratios{};  //! The intesrction ratio of intersecting pixels
    Vectors_t intersection_normals;  //! The normal vectors of the interface in
                                     //! the intersecting pixels
    std::vector<corkpp::IntersectionState>
        intersection_state{};  //! The state of the interface in
                               //! the intersecting pixels
  };

}  // namespace muSpectre

#endif  // SRC_COMMON_INTERSECTION_OCTREE_HH_
