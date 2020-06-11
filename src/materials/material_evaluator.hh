/**
 * @file   material_evaluator.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   12 Dec 2018
 *
 * @brief  Helper to evaluate material laws
 *
 * Copyright © 2018 Till Junge
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

#ifndef SRC_MATERIALS_MATERIAL_EVALUATOR_HH_
#define SRC_MATERIALS_MATERIAL_EVALUATOR_HH_

#include "common/muSpectre_common.hh"
#include "materials/materials_toolbox.hh"

#include <libmugrid/T4_map_proxy.hh>
#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/mapped_field.hh>

#include <exception>
#include <memory>
#include <sstream>

namespace muSpectre {

  /**
   * forward declaration to avoid including material_base.hh
   */
  class MaterialBase;

  /**
   * Small convenience class providing a common interface to evaluate materials
   * without the need to set up an entire homogenisation problem. Useful for
   * debugging material laws.
   *
   * \tparam DimM Dimensionality of the material
   */
  template <Index_t DimM>
  class MaterialEvaluator {
   public:
    //! shorthand for second-rank tensors
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;

    //! shorthand for fourth-rank tensors
    using T4_t = muGrid::T4Mat<Real, DimM>;

    //! map of a second-rank tensor
    using T2_map = Eigen::Map<T2_t>;

    //! map of a fourth-rank tensor
    using T4_map = muGrid::T4MatMap<Real, DimM>;

    //! const map of a second-rank tensor
    using T2_const_map = Eigen::Map<const T2_t>;

    //! const map of a fourth-rank tensor
    using T4_const_map = muGrid::T4MatMap<Real, DimM, true>;

    //! convenience alias
    using FieldColl_t = muGrid::GlobalFieldCollection;

    //! Default constructor
    MaterialEvaluator() = delete;

    /**
     * constructor with a shared pointer to a Material
     */
    explicit MaterialEvaluator(std::shared_ptr<MaterialBase> material)
        : material{material}, collection{std::make_unique<FieldColl_t>(
                                  DimM,
                                  []() {
                                    muGrid::FieldCollection::SubPtMap_t map{};
                                    map[QuadPtTag] = OneQuadPt;
                                    return map;
                                  }())},
          strain{"gradient", *this->collection, QuadPtTag},
          stress{"stress", *this->collection, QuadPtTag}, tangent{
                                                              "tangent",
                                                              *this->collection,
                                                              QuadPtTag} {
      this->collection->initialise(
          muGrid::CcoordOps::get_cube<DimM>(Index_t{1}), Ccoord_t<DimM>{});
    }

    //! Copy constructor
    MaterialEvaluator(const MaterialEvaluator & other) = delete;

    //! Move constructor
    MaterialEvaluator(MaterialEvaluator && other) = default;

    //! Destructor
    virtual ~MaterialEvaluator() = default;

    //! Copy assignment operator
    MaterialEvaluator & operator=(const MaterialEvaluator & other) = delete;

    //! Move assignment operator
    MaterialEvaluator & operator=(MaterialEvaluator && other) = default;

    /**
     * for materials with state variables. See `muSpectre::MaterialBase` for
     * details
     */
    void save_history_variables() { this->material->save_history_variables(); }

    /**
     * Evaluates the underlying materials constitutive law and returns the
     * stress P or σ as a function of the placement gradient F or small strain
     * tensor ε depending on the formulation
     * (`muSpectre::Formulation::small_strain` for σ(ε),
     * `muSpectre::Formulation::finite_strain` for P(F))
     */
    inline T2_const_map evaluate_stress(const Eigen::Ref<const T2_t> & grad,
                                        const Formulation & form);

    /**
     * Evaluates the underlying materials constitutive law and returns the the
     * stress P or σ and the tangent moduli K as a function of the placement
     * gradient F or small strain tensor ε depending on the formulation
     * (`muSpectre::Formulation::small_strain` for σ(ε),
     * `muSpectre::Formulation::finite_strain` for P(F))
     */
    inline std::tuple<T2_const_map, T4_const_map>
    evaluate_stress_tangent(const Eigen::Ref<const T2_t> & grad,
                            const Formulation & form);

    /**
     * estimate the tangent using finite difference
     */
    inline T4_t
    estimate_tangent(const Eigen::Ref<const T2_t> & grad,
                     const Formulation & form, const Real step,
                     const FiniteDiff diff_type = FiniteDiff::centred);

    /**
     * initialise the material and the fields
     */
    inline void initialise();

   protected:
    /**
     * throws a runtime error if the material's per-pixel data has not been set.
     */
    void check_init();

    /**
     * storage of the material is managed through a shared pointer
     */
    std::shared_ptr<MaterialBase> material;

    /**
     * storage of the strain, stress and tangent fields is managed through a
     * unique pointer
     */
    std::unique_ptr<FieldColl_t> collection;

    //! strain field (independent variable)
    muGrid::MappedT2Field<Real, Mapping::Mut, DimM, IterUnit::SubPt> strain;

    //! stress field (result)
    muGrid::MappedT2Field<Real, Mapping::Mut, DimM, IterUnit::SubPt> stress;

    //! field of tangent moduli (result)
    muGrid::MappedT4Field<Real, Mapping::Mut, DimM, IterUnit::SubPt> tangent;

    //! whether the evaluator has been initialised
    bool is_initialised{false};
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto
  MaterialEvaluator<DimM>::evaluate_stress(const Eigen::Ref<const T2_t> & grad,
                                           const Formulation & form)
      -> T2_const_map {
    this->check_init();
    this->strain.get_map()[0] = grad;
    this->material->compute_stresses(this->strain.get_field(),
                                     this->stress.get_field(), form);
    return T2_const_map(this->stress.get_map()[0].data());
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialEvaluator<DimM>::evaluate_stress_tangent(
      const Eigen::Ref<const T2_t> & grad, const Formulation & form)
      -> std::tuple<T2_const_map, T4_const_map> {
    this->check_init();
    this->strain.get_map()[0] = grad;
    this->material->compute_stresses_tangent(this->strain.get_field(),
                                             this->stress.get_field(),
                                             this->tangent.get_field(), form);
    return std::make_tuple(T2_const_map(this->stress.get_map()[0].data()),
                           T4_const_map(this->tangent.get_map()[0].data()));
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialEvaluator<DimM>::check_init() {
    if (not this->is_initialised) {
      this->initialise();
    }
    const auto & size{this->material->size()};
    if (size < 1) {
      throw muGrid::RuntimeError(
          "Not initialised! You have to call `add_pixel(...)` on your material "
          "exactly one time before you can evaluate it.");
    } else if (size > 1) {
      std::stringstream error{};
      error << "The material to be evaluated should have exactly one pixel "
               "with one quadrature point added. You've added "
            << size << " quadrature points.";
      throw muGrid::RuntimeError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialEvaluator<DimM>::estimate_tangent(
      const Eigen::Ref<const T2_t> & grad, const Formulation & form,
      const Real delta, const FiniteDiff diff_type) -> T4_t {
    using T2_vec = Eigen::Map<Eigen::Matrix<Real, DimM * DimM, 1>>;

    T4_t tangent{T4_t::Zero()};

    const T2_t stress{this->evaluate_stress(grad, form)};

    auto fun{[&form, this](const auto & grad_) {
      return this->evaluate_stress(grad_, form);
    }};

    auto symmetrise{
        [](auto & eps) { eps = .5 * (eps + eps.transpose().eval()); }};

    static_assert(Int(decltype(tangent.col(0))::SizeAtCompileTime) ==
                      Int(T2_t::SizeAtCompileTime),
                  "wrong column size");

    for (Index_t i{}; i < DimM * DimM; ++i) {
      T2_t strain2{grad};
      T2_vec strain2_vec{strain2.data()};

      switch (diff_type) {
      case FiniteDiff::forward: {
        strain2_vec(i) += delta;
        if (form == Formulation::small_strain) {
          symmetrise(strain2);
        }
        T2_t del_f_del{(fun(strain2) - stress) / delta};

        tangent.col(i) = T2_vec(del_f_del.data());
        break;
      }
      case FiniteDiff::backward: {
        strain2_vec(i) -= delta;
        if (form == Formulation::small_strain) {
          symmetrise(strain2);
        }
        T2_t del_f_del{(stress - fun(strain2)) / delta};

        tangent.col(i) = T2_vec(del_f_del.data());
        break;
      }
      case FiniteDiff::centred: {
        T2_t strain1{grad};
        T2_vec strain1_vec{strain1.data()};
        strain1_vec(i) += delta;
        strain2_vec(i) -= delta;

        if (form == Formulation::small_strain) {
          symmetrise(strain1);
          symmetrise(strain2);
        }

        T2_t del_f_del{(fun(strain1).eval() - fun(strain2).eval()) /
                       (2 * delta)};

        tangent.col(i) = T2_vec(del_f_del.data());
        break;
      }
      default: {
        throw muGrid::RuntimeError("Unknown FiniteDiff value");
        break;
      }
      }

      static_assert(Int(decltype(tangent.col(i))::SizeAtCompileTime) ==
                        Int(T2_t::SizeAtCompileTime),
                    "wrong column size");
    }
    return tangent;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialEvaluator<DimM>::initialise() {
    this->material->initialise();
    this->is_initialised = true;
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_EVALUATOR_HH_
