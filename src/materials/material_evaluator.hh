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
 * General Public License for more details.
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
 */

#ifndef SRC_MATERIALS_MATERIAL_EVALUATOR_HH_
#define SRC_MATERIALS_MATERIAL_EVALUATOR_HH_

#include "common/common.hh"
#include "common/T4_map_proxy.hh"
#include "common/ccoord_operations.hh"
#include "materials/materials_toolbox.hh"
#include "common/field.hh"

#include <exception>
#include <memory>
#include <sstream>

namespace muSpectre {

  /**
   * forward declaration to avoid incluning material_base.hh
   */
  template <Dim_t DimS, Dim_t DimM>
  class MaterialBase;

  template <Dim_t DimM>
  class MaterialEvaluator {
   public:
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    using T4_t = T4Mat<Real, DimM>;

    using T2_map = Eigen::Map<T2_t>;
    using T4_map = T4MatMap<Real, DimM>;

    using T2_const_map = Eigen::Map<const T2_t>;
    using T4_const_map = T4MatMap<Real, DimM, true>;

    using FieldColl_t = GlobalFieldCollection<DimM>;
    using T2Field_t = TensorField<FieldColl_t, Real, secondOrder, DimM>;
    using T4Field_t = TensorField<FieldColl_t, Real, fourthOrder, DimM>;

    //! Default constructor
    MaterialEvaluator() = delete;

    /**
     * constructor with a shared pointer to a Material
     */
    explicit MaterialEvaluator(
        std::shared_ptr<MaterialBase<DimM, DimM>> material)
        : material{material},
          collection{std::make_unique<FieldColl_t>()},
          strain{make_field<T2Field_t>("gradient", *this->collection)},
          stress{make_field<T2Field_t>("stress", *this->collection)},
          tangent{make_field<T4Field_t>("tangent", *this->collection)} {
      this->collection->initialise(CcoordOps::get_cube<DimM>(1), {0});
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

   protected:
    /**
     * throws a runtime error if the material's per-pixel data has not been set.
     */
    void check_init() const;

    std::shared_ptr<MaterialBase<DimM, DimM>> material;
    std::unique_ptr<FieldColl_t> collection;
    T2Field_t & strain;
    T2Field_t & stress;
    T4Field_t & tangent;
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM>
  auto MaterialEvaluator<DimM>::evaluate_stress(
      const Eigen::Ref<const T2_t> & grad, const Formulation & form)
      -> T2_const_map {
    this->check_init();
    this->strain.get_map()[0] = grad;
    this->material->compute_stresses(this->strain, this->stress, form);
    return T2_const_map(this->stress.get_map()[0].data());
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM>
  auto MaterialEvaluator<DimM>::evaluate_stress_tangent(
      const Eigen::Ref<const T2_t> & grad, const Formulation & form)
      -> std::tuple<T2_const_map, T4_const_map> {
    this->check_init();
    this->strain.get_map()[0] = grad;
    this->material->compute_stresses_tangent(this->strain, this->stress,
                                            this->tangent, form);
    return std::make_tuple(T2_const_map(this->stress.get_map()[0].data()),
                           T4_const_map(this->tangent.get_map()[0].data()));
  }

  template <Dim_t DimM>
  void MaterialEvaluator<DimM>::check_init() const {
    const auto & size{this->material->size()};
    if (size < 1) {
      throw std::runtime_error(
          "Not initialised! You have to call `add_pixel(...)` on your material "
          "exactly one time before you can evaluate it.");
    } else if (size > 1) {
      std::stringstream error{};
      error << "The material to be evaluated should have exactly one pixel "
               "added. You've added "
            << size << " pixels.";
      throw std::runtime_error(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM>
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

    for (Dim_t i{}; i < DimM * DimM; ++i) {
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
        throw std::runtime_error("Unknown FiniteDiff value");
        break;
      }
      }
      static_assert(Int(decltype(tangent.col(i))::SizeAtCompileTime) ==
                        Int(T2_t::SizeAtCompileTime),
                    "wrong column size");
    }
    return tangent;
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_EVALUATOR_HH_
