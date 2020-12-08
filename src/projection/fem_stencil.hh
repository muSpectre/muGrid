/**
 * @file   fem_stencil.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   02 Aug 2020
 *
 * @brief  Finite element stencil, inherits from a  gradient operator.
 * In addition has quadrature weights
 *
 * Copyright © 2020 Till Junge, Martin Ladecký
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "libmugrid/gradient_operator_default.hh"
#include "common/muSpectre_common.hh"
#include <cell/cell_data.hh>

#include <Eigen/Dense>

#ifndef SRC_PROJECTION_FEM_STENCIL_HH_
#define SRC_PROJECTION_FEM_STENCIL_HH_

namespace muSpectre {
  class FEMStencilBase {
   public:
    //! Default constructor
    FEMStencilBase() = delete;

    //! Constructor with weights vector and cell data
    FEMStencilBase(const std::vector<Real> & quadrature_weights,
                   std::shared_ptr<CellData> cell);

    //! Copy constructor
    FEMStencilBase(const FEMStencilBase & other) = delete;

    //! Move constructor
    FEMStencilBase(FEMStencilBase && other) = default;

    //! Destructor
    virtual ~FEMStencilBase() = default;

    //! Copy assignment operator
    FEMStencilBase & operator=(const FEMStencilBase & other) = delete;

    //! Move assignment operator
    FEMStencilBase & operator=(FEMStencilBase && other) = delete;

    //! Return the quadrature weights vector
    const std::vector<Real> & get_quadrature_weights() const;

    //! return the pointer to the cell
    std::shared_ptr<CellData> get_cell_ptr() const;

    //! return a ref to the gradient operator
    virtual std::shared_ptr<muGrid::GradientOperatorBase>
    get_gradient_operator() = 0;

    //! return a const ref to the gradient operator
    virtual const std::shared_ptr<muGrid::GradientOperatorBase>
    get_gradient_operator() const = 0;

    //! return the number of quadrature points per pixel
    virtual Index_t get_nb_pixel_quad_pts() const = 0;

    //! return the number of nodal points per pixel
    virtual Index_t get_nb_pixel_nodal_pts() const = 0;

   protected:
    std::vector<Real> quadrature_weights;
    std::shared_ptr<CellData> cell;
  };

  /**
   * base class  defining the interface of a FEM discretisation. It allows you
   * to create any FEM discretisation (even at run-time) by providing the shape
   * functions, gradients, and quadrature weights. The flexibility comes at the
   * cost of an unoptimized runtime cost.
   */
  template <class GradientOperator = muGrid::GradientOperatorDefault>
  class FEMStencil : public FEMStencilBase {
   public:
    using Parent = FEMStencilBase;
    //! Default constructor
    FEMStencil() = delete;

    /**
     * constructor with all the parameters
     *
     * @param spatial_dimension spatial dimension of the stencil
     * @param nb_quad_pts number of quadrature points per element
     * @param nb_elements number of elements per pixel
     * @param nb_elemenodal_pts number of nodal points per element
     * @param nb_pixelnodal_pts number of nodal points per pixel
     * @param shape_fn_gradients per quadrature point and element, one matrix
     * of shape function gradients (evaluated on the quadrature point)
     * @param nodal_pts nodal point indices composed of nodal point index
     * within a pixel and pixel coordinate offset. E.g. the second nodal point
     * in pixel (i+1, j) gets (1, (1, 0))
     * @quadrature_weights quadrature weights associated to quadrature points
     * including the jacobian (det of Jacobian matrix)
     */
    FEMStencil(
        const Index_t & nb_quad_pts_per_element, const Index_t & nb_elements,
        const Index_t & nb_element_nodal_pts,
        const Index_t & nb_pixel_nodal_pts,
        const std::vector<std::vector<Eigen::MatrixXd>> & shape_fn_gradients,
        const std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>> &
            nodal_pts,
        const std::vector<Real> & quadrature_weights,
        std::shared_ptr<CellData> cell);

    //! Copy constructor
    FEMStencil(const FEMStencil & other) = delete;

    //! Move constructor
    FEMStencil(FEMStencil && other) = default;

    //! Destructor
    virtual ~FEMStencil() = default;

    //! Copy assignment operator
    FEMStencil & operator=(const FEMStencil & other) = delete;

    //! Move assignment operator
    FEMStencil & operator=(FEMStencil && other) = delete;

    //! return a ref to the gradient operator
    std::shared_ptr<muGrid::GradientOperatorBase> get_gradient_operator() final;

    //! return a const ref to the gradient operator
    const std::shared_ptr<muGrid::GradientOperatorBase>
    get_gradient_operator() const final;

    //! return the number of quadrature points per pixel
    Index_t get_nb_pixel_quad_pts() const final;

    //! return the number of nodal points per pixel
    Index_t get_nb_pixel_nodal_pts() const final;

   protected:
    std::shared_ptr<GradientOperator> gradient_operator;
  };
}  // namespace muSpectre

#endif  // SRC_PROJECTION_FEM_STENCIL_HH_
