/**
 * @file   solver_base.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   26 Jun 2020
 *
 * @brief  Base class to define the interface for solvers based on CellData
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

#include "solver_common.hh"
#include "matrix_adaptor.hh"

#include "common/muSpectre_common.hh"
#include "cell/cell_data.hh"

#include <libmugrid/grid_common.hh>
#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/units.hh>

#include <map>

#ifndef SRC_SOLVER_SOLVER_BASE_HH_
#define SRC_SOLVER_SOLVER_BASE_HH_

namespace muSpectre {

  class SolverBase : public MatrixAdaptable {
   public:
    using EigenStrainFunc_t =
        typename std::function<void(muGrid::TypedFieldBase<Real> &)>;

    using CellExtractFieldFunc_t = typename std::function<void(
        const std::shared_ptr<muSpectre::CellData>)>;

    using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
#ifdef NO_EXPERIMENTAL
    using EigenStrainFunc_ref = typename muGrid::optional<EigenStrainFunc_t &>;
    using CellExtractFieldFunc_ref =
        typename muGrid::optional<CellExtractFieldFunc_t>;
#else
    using EigenStrainFunc_ref =
        typename muGrid::optional<std::reference_wrapper<EigenStrainFunc_t>>;
    using CellExtractFieldFunc_ref =
        typename muGrid::optional<CellExtractFieldFunc_t>;
#endif

    using Parent = MatrixAdaptable;
    using MappedField_t =
        muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>>;
    using LoadStep = std::map<muGrid::PhysicsDomain, Eigen::MatrixXd>;
    using FilePath = std::string;

    //! Default constructor
    SolverBase() = delete;

    //! Explicit constructor
    SolverBase(std::shared_ptr<CellData> cell_data,
               const muGrid::Verbosity & verbosity,
               const SolverType & solver_type);

    //! Copy constructor
    SolverBase(const SolverBase & other) = delete;

    //! Move constructor
    SolverBase(SolverBase && other) = default;

    //! Destructor
    virtual ~SolverBase() = default;

    //! Copy assignment operator
    SolverBase & operator=(const SolverBase & other) = delete;

    //! Move assignment operator
    SolverBase & operator=(SolverBase && other) = delete;

    /**
     * solve for a single increment of load. This function cannot handle
     * eigenload. If you have an eigenload problem, use
     * `apply_load_increment(const LoadStep & load_step)`, thenn apply your
     * eigen load followed by `solve_load_increment()` (without argument).
     */
    virtual OptimizeResult
    solve_load_increment(const LoadStep & load_step,
                         EigenStrainFunc_ref eigen_strain_func,
                         CellExtractFieldFunc_ref cell_extract_func) = 0;

    /**
     * set formulation (small vs finite strain) used fo
r mechanics domain.
     * Setting this is only necessary if at least one Material in the list
     * is a mechanics material.
     */
    void set_formulation(const Formulation & formulation);

    //! returns formulation used for mechanics domain
    const Formulation & get_formulation() const;

    //! resets the load step counter to zero
    void reset_counter_load_step();

    //! get current load step counter
    const Int & get_counter_load_step() const;

    //! resets the Newton iteration counter to zero
    void reset_counter_iteration();

    //! get current newton iteration counter
    const Int & get_counter_iteration() const;

    /** reset the linear/nonlinear status of all the materials in the given
    domain
    */
    void clear_last_step_nonlinear(const muGrid::PhysicsDomain & domain);

    /**
     * evaluates and returns the stress for the currently set strain
     */
    virtual const MappedField_t &
    evaluate_stress(const muGrid::PhysicsDomain & domain);

    /**
     * evaluates and returns the stress and tangent moduli for the currently set
     * strain
     */
    virtual std::tuple<const MappedField_t &, const MappedField_t &>
    evaluate_stress_tangent(const muGrid::PhysicsDomain & domain);

    virtual void initialise_cell() = 0;

    const muFFT::Communicator & get_communicator() const;

    //! getter of the cell data
    const std::shared_ptr<CellData> get_cell_data() const;

    //! calculates the squared norm of a vector distributed memory safe
    template <class T>
    Real squared_norm(const T & vec) {
      auto && comm{this->cell_data->get_communicator()};
      return comm.sum(vec.squaredNorm());
    }

    template <class T>
    Real inf_norm(const T & field) {
      auto && comm{this->cell_data->get_communicator()};
      return comm.max(std::accumulate(
          field->begin(), field->end(), 0.0,
          [](Real max, auto && field_entry) -> Real {
            auto && field_entry_norm{field_entry.squaredNorm()};
            return field_entry_norm > max ? field_entry_norm : max;
          }));
    }

    //! calculates the dot product of  two vectors distributed memory safe
    Real dot(const Vector_t & vec_a, const Vector_t & vec_b);

    const SolverType & get_solver_type() const;

   protected:
    std::shared_ptr<CellData> cell_data;
    muGrid::Verbosity verbosity;
    Formulation formulation{Formulation::not_set};
    //! Gradient (or strain) field
    std::map<muGrid::PhysicsDomain, std::shared_ptr<MappedField_t>> grads{};
    /**
     * Gradient (or strain) field evaluated by the materials this pointer points
     * usually to the same field as `grad` does, but in the case of problems
     * with an eigen strain, these point to different fields
     */
    std::map<muGrid::PhysicsDomain, std::shared_ptr<MappedField_t>>
        eval_grads{};
    //! Flux (or stress) field
    std::map<muGrid::PhysicsDomain, std::shared_ptr<MappedField_t>> fluxes{};
    //! Tangent moduli field
    std::map<muGrid::PhysicsDomain, std::shared_ptr<MappedField_t>> tangents{};
    Int counter_load_step{};         //!< load step counter
    Int counter_iteration{};  //!< Newton iteration counter
    Int default_count_width{3};
    bool is_initialised{false};
    SolverType solver_type;  //!< Type of the solver either
                             //! Spectral or FiniteElements
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_SOLVER_BASE_HH_
