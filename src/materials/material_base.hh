/**
 * @file   material_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   25 Oct 2017
 *
 * @brief  Base class for materials (constitutive models)
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

#ifndef SRC_MATERIALS_MATERIAL_BASE_HH_
#define SRC_MATERIALS_MATERIAL_BASE_HH_

#include "common/muSpectre_common.hh"
#include "materials/materials_toolbox.hh"

#include <libmugrid/field.hh>
#include <libmugrid/field_collection.hh>

#include <string>
#include <tuple>
namespace muSpectre {

  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law) and
  /**
   * @a DimM is the material dimension (i.e., the dimension of constitutive
   * law; even for e.g. two-dimensional problems the constitutive law could
   * live in three-dimensional space for e.g. plane strain or stress problems)
   */
  template <Dim_t DimS, Dim_t DimM>
  class MaterialBase {
   public:
    //! typedefs for data handled by this interface
    //! global field collection for cell-wide fields, like stress, strain, etc
    using GFieldCollection_t = muGrid::GlobalFieldCollection<DimS>;
    //! field collection for internal variables, such as eigen-strains,
    //! plastic strains, damage variables, etc, but also for managing which
    //! pixels the material is responsible for
    using MFieldCollection_t = muGrid::LocalFieldCollection<DimS>;
    using iterator = typename MFieldCollection_t::iterator;  //!< pixel iterator
    //! polymorphic base class for fields only to be used for debugging
    using Field_t = muGrid::internal::FieldBase<GFieldCollection_t>;
    //! Full type for stress fields
    using StressField_t = muGrid::TensorField<GFieldCollection_t, Real,
                                              muGrid::secondOrder, DimM>;
    using MStressField_t = muGrid::TensorField<MFieldCollection_t, Real,
                                               muGrid::secondOrder, DimM>;
    //! Full type for strain fields
    using StrainField_t = StressField_t;
    using MStrainField_t = MStressField_t;
    //! Full type for tangent stiffness fields fields

    using TangentField_t = muGrid::TensorField<GFieldCollection_t, Real,
                                               muGrid::fourthOrder, DimM>;
    using MTangentField_t = muGrid::TensorField<MFieldCollection_t, Real,
                                                muGrid::fourthOrder, DimM>;
    using MScalarField_t =
        muGrid::ScalarField<muGrid::LocalFieldCollection<DimS>, Real>;
    using MScalarFieldMap_t =
        muGrid::ScalarFieldMap<muGrid::LocalFieldCollection<DimS>, Real, true>;
    using MVectorField_t =
        muGrid::TensorField<muGrid::LocalFieldCollection<DimS>, Real,
                            muGrid::firstOrder, DimM>;

    using Ccoord = Ccoord_t<DimS>;  //!< cell coordinates type
    using Rcoord = Rcoord_t<DimS>;  //!< real coordinates type
    //! Default constructor
    MaterialBase() = delete;

    //! Construct by name
    explicit MaterialBase(std::string name);

    //! Copy constructor
    MaterialBase(const MaterialBase & other) = delete;

    //! Move constructor
    MaterialBase(MaterialBase && other) = delete;

    //! Destructor
    virtual ~MaterialBase() = default;

    //! Copy assignment operator
    MaterialBase & operator=(const MaterialBase & other) = delete;

    //! Move assignment operator
    MaterialBase & operator=(MaterialBase && other) = delete;

    /**
     *  take responsibility for a pixel identified by its cell coordinates
     *  WARNING: this won't work for materials with additional info per pixel
     *  (as, e.g. for eigenstrain), we need to pass more parameters. Materials
     *  of this type need to overload add_pixel
     */
    virtual void add_pixel(const Ccoord & ccooord);

    virtual void add_pixel_split(const Ccoord & local_ccoord, Real ratio);

    // this function is responsible for allocating fields in case cells are
    // split or laminate
    void allocate_optional_fields(SplitCell is_cell_split = SplitCell::no);

    //! allocate memory, etc, but also: wipe history variables!
    virtual void initialise();

    /**
     * for materials with state variables, these typically need to be
     * saved/updated an the end of each load increment, the virtual
     * base implementation does nothing, but materials with history
     * variables need to implement this
     */
    virtual void save_history_variables() {}

    //! return the material's name
    const std::string & get_name() const;

    //! spatial dimension for static inheritance
    constexpr static Dim_t sdim() { return DimS; }
    //! material dimension for static inheritance
    constexpr static Dim_t mdim() { return DimM; }
    //! computes stress
    virtual void compute_stresses(const StrainField_t & F, StressField_t & P,
                                  Formulation form,
                                  SplitCell is_cell_split = SplitCell::no) = 0;
    /**
     * Convenience function to compute stresses, mostly for debugging and
     * testing. Has runtime-cost associated with compatibility-checking and
     * conversion of the Field_t arguments that can be avoided by using the
     * version with strongly typed field references
     */
    void compute_stresses(const Field_t & F, Field_t & P, Formulation form,
                          SplitCell is_cell_split = SplitCell::no);
    //! computes stress and tangent moduli
    virtual void
    compute_stresses_tangent(const StrainField_t & F, StressField_t & P,
                             TangentField_t & K, Formulation form,
                             SplitCell is_cell_split = SplitCell::no) = 0;
    /**
     * Convenience function to compute stresses and tangent moduli, mostly for
     * debugging and testing. Has runtime-cost associated with
     * compatibility-checking and conversion of the Field_t arguments that can
     * be avoided by using the version with strongly typed field references
     */
    void compute_stresses_tangent(const Field_t & F, Field_t & P, Field_t & K,
                                  Formulation form,
                                  SplitCell is_cell_split = SplitCell::no);

    // this function return the ratio of which the
    // input pixel is consisted of this material
    Real get_assigned_ratio(Ccoord pixel);

    void get_assigned_ratios(std::vector<Real> & pixel_assigned_ratios,
                             Ccoord subdomain_resolutions,
                             Ccoord subdomain_locations);
    // This function returns the local field containng assigned ratios of this
    // material
    auto get_assigned_ratio_field() -> MScalarField_t &;

    //! iterator to first pixel handled by this material
    inline iterator begin() { return this->internal_fields.begin(); }
    //! iterator past the last pixel handled by this material
    inline iterator end() { return this->internal_fields.end(); }
    //! number of pixels assigned to this material
    inline size_t size() const { return this->internal_fields.size(); }

    //! type to return real-valued fields in
    using EigenMap =
        Eigen::Map<Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic>>;
    /**
     * return an internal field identified by its name as an Eigen Array
     */
    EigenMap get_real_field(std::string field_name);

    /**
     * list the names of all internal fields
     */
    std::vector<std::string> list_fields() const;

    //! gives access to internal fields
    inline MFieldCollection_t & get_collection() {
      return this->internal_fields;
    }

    using Stiffness_t = muGrid::T4Mat<Real, DimM>;
    using Strain_t = Eigen::Matrix<Real, DimM, DimM>;
    using Stress_t = Strain_t;
    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     */

    inline auto evaluate_stress_base(const Eigen::Ref<const Strain_t> & E,
                                     const size_t & pixel_index,
                                     Formulation form) -> Stress_t;

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor)
     */

    inline auto
    evaluate_stress_tangent_base(const Eigen::Ref<const Strain_t> & E,
                                 const size_t & pixel_index, Formulation form)
        -> std::tuple<Stress_t, Stiffness_t>;

    //
    virtual auto
    constitutive_law_small_strain(const Eigen::Ref<const Strain_t> & strain,
                                  const size_t & pixel_index) -> Stress_t = 0;
    virtual auto
    constitutive_law_finite_strain(const Eigen::Ref<const Strain_t> & strain,
                                   const size_t & pixel_index) -> Stress_t = 0;

    virtual auto constitutive_law_tangent_small_strain(
        const Eigen::Ref<const Strain_t> & strain, const size_t & pixel_index)
        -> std::tuple<Stress_t, Stiffness_t> = 0;
    virtual auto constitutive_law_tangent_finite_strain(
        const Eigen::Ref<const Strain_t> & strain, const size_t & pixel_index)
        -> std::tuple<Stress_t, Stiffness_t> = 0;

   protected:
    const std::string name;  //!< material's name (for output and debugging)
    MFieldCollection_t internal_fields{};  //!< storage for internal variables
    //!< field holding the assigning ratio of the material
    optional<std::reference_wrapper<MScalarField_t>> assigned_ratio{};
    //!< field holding the normal vetor of the interface in laminate materials
    bool is_initialised{false};  //!< to handle double initialisation right

    template <class Strains_t, class Stresses_t,
              SplitCell is_cell_split = SplitCell::no>
    class iterable_proxy;
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto MaterialBase<DimS, DimM>::evaluate_stress_base(
      const Eigen::Ref<const Strain_t> & strain, const size_t & pixel_index,
      Formulation form) -> Stress_t {
    Stress_t stress;
    Eigen::Map<Stress_t> P((stress.data()));
    switch (form) {
    case Formulation::small_strain: {
      P = constitutive_law_small_strain(strain, pixel_index);
      break;
    }
    case Formulation::finite_strain: {
      P = constitutive_law_finite_strain(strain, pixel_index);
      break;
    }
    default:
      throw std::runtime_error("Unknown formulation");
      break;
    }
    return P;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto MaterialBase<DimS, DimM>::evaluate_stress_tangent_base(
      const Eigen::Ref<const Strain_t> & strain, const size_t & pixel_index,
      Formulation form) -> std::tuple<Stress_t, Stiffness_t> {
    Stress_t stress;
    Stiffness_t stiffness;
    Eigen::Map<Stress_t> P((stress.data()));
    Eigen::Map<Stiffness_t> K((stiffness.data()));
    std::tuple<Eigen::Map<Stress_t>, Eigen::Map<Stiffness_t>> PK(P, K);
    switch (form) {
    case Formulation::small_strain: {
      PK = constitutive_law_tangent_small_strain(strain, pixel_index);
      break;
    }
    case Formulation::finite_strain: {
      PK = constitutive_law_tangent_finite_strain(strain, pixel_index);
      break;
    }
    default:
      throw std::runtime_error("Unknown formulation");
      break;
    }

    return std::tuple<Stress_t, Stiffness_t>(stress, stiffness);
  }

  namespace internal {

    template <class Dummy>
    struct StressesTComputer {};

    template <class StressMap_t>
    struct StressesTComputer<std::tuple<StressMap_t>> {
      using type = std::tuple<typename StressMap_t::reference>;
    };

    template <class StressMap_t, class TangentMap_t>
    struct StressesTComputer<std::tuple<StressMap_t, TangentMap_t>> {
      using type = std::tuple<typename StressMap_t::reference,
                              typename TangentMap_t::reference>;
    };

    template <class Dummy>
    struct StrainsTComputer {};

    template <class StrainMap_t>
    struct StrainsTComputer<std::tuple<StrainMap_t>> {
      using type = std::tuple<typename StrainMap_t::reference>;
    };

    template <class StrainMap_t>
    struct StrainsTComputer<std::tuple<StrainMap_t, StrainMap_t>> {
      using type = std::tuple<typename StrainMap_t::reference,
                              typename StrainMap_t::reference>;
    };

    template <class OutType>
    struct TupleBuilder {
      template <class... InTypes, size_t... I>
      static OutType helper(std::tuple<InTypes...> const & arg,
                            std::index_sequence<I...>) {
        return OutType(
            typename std::tuple_element<I, OutType>::type(std::get<I>(arg))...);
      }

      template <class... InTypes>
      static OutType build(std::tuple<InTypes...> const & arg) {
        return helper(arg, std::index_sequence_for<InTypes...>{});
      }
    };

  }  // namespace internal
  /* ---------------------------------------------------------------------- */
  //! this iterator class is a default for simple laws that just take a strain
  template <Dim_t DimS, Dim_t DimM>
  template <class Strains_t, class Stresses_t, SplitCell is_cell_split>
  class MaterialBase<DimS, DimM>::iterable_proxy {
   public:
    //! Default constructor
    iterable_proxy() = delete;

    //! expected type for strain values
    using Strain_t = typename internal::StrainsTComputer<Strains_t>::type;
    //! expected type for stress values
    using Stress_t = typename internal::StressesTComputer<Stresses_t>::type;

    //! tuple containing a strain and possibly a strain-rate field
    using StrainFieldTup = std::conditional_t<
        (std::tuple_size<Strains_t>::value == 2),
        std::tuple<const StrainField_t &, const StrainField_t &>,
        std::tuple<const StrainField_t &>>;

    //! tuple containing a stress and possibly a tangent stiffness field
    using StressFieldTup =
        std::conditional_t<(std::tuple_size<Stresses_t>::value == 2),
                           std::tuple<StressField_t &, TangentField_t &>,
                           std::tuple<StressField_t &>>;

    /** Iterator uses the material's internal variables field
        collection to iterate selectively over the global fields
        (such as the transformation gradient F and first
        Piola-Kirchhoff stress P.
    **/
    // ! Constructors
    // with tangent and with strain rate
    template <bool DoNeedTgt = std::tuple_size<Stresses_t>::value == 2,
              bool DoNeedRate = std::tuple_size<Strain_t>::value == 2>
    iterable_proxy(MaterialBase & mat, const StrainField_t & F,
                   std::enable_if_t<DoNeedRate, const StrainField_t> & F_rate,
                   StressField_t & P,
                   std::enable_if_t<DoNeedTgt, TangentField_t> & K)
        : material{mat}, strain_field{std::cref(F), std::cref(F_rate)},
          stress_tup{P, K} {};

    // without tangent and with strain rate
    template <bool DontNeedTgt = std::tuple_size<Stresses_t>::value == 1,
              bool DoNeedRate = std::tuple_size<Strain_t>::value == 2>
    iterable_proxy(MaterialBase & mat, const StrainField_t & F,
                   std::enable_if_t<DoNeedRate, const StrainField_t> & F_rate,
                   std::enable_if_t<DontNeedTgt, StressField_t> & P)
        : material{mat}, strain_field{std::cref(F), std::cref(F_rate)},
          stress_tup{P} {};

    // with tangent and without strain rate
    template <bool DoNeedTgt = std::tuple_size<Stresses_t>::value == 2,
              bool DontNeedRate = std::tuple_size<Strain_t>::value == 1>
    iterable_proxy(MaterialBase & mat,
                   std::enable_if_t<DontNeedRate, const StrainField_t> & F,
                   StressField_t & P,
                   std::enable_if_t<DoNeedTgt, TangentField_t> & K)
        : material{mat}, strain_field{std::cref(F)}, stress_tup{P, K} {};

    // without tangent and without strain rate
    template <bool DontNeedTgt = std::tuple_size<Stresses_t>::value == 1,
              bool DontNeedRate = std::tuple_size<Strain_t>::value == 1>
    iterable_proxy(MaterialBase & mat,
                   std::enable_if_t<DontNeedRate, const StrainField_t> & F,
                   std::enable_if_t<DontNeedTgt, StressField_t> & P)
        : material{mat}, strain_field{std::cref(F)}, stress_tup{P} {};

    //! Copy constructor
    iterable_proxy(const iterable_proxy & other) = default;

    //! Move constructor
    iterable_proxy(iterable_proxy && other) = default;

    //! Destructor
    virtual ~iterable_proxy() = default;

    //! Copy assignment operator
    iterable_proxy & operator=(const iterable_proxy & other) = default;

    //! Move assignment operator
    iterable_proxy & operator=(iterable_proxy && other) = default;

    /**
     * dereferences into a tuple containing strains, and internal
     * variables, as well as maps to the stress and potentially
     * stiffness maps where to write the response of a pixel
     */
    class iterator {
     public:
      //! return type contains a tuple of strain and possibly strain rate,
      //! stress and possibly stiffness, and a refererence to the pixel index
      using value_type = std::tuple<Strain_t, Stress_t, const size_t &, Real>;
      using iterator_category = std::forward_iterator_tag;  //!< stl conformance

      //! Default constructor
      iterator() = delete;

      /** Iterator uses the material's internal variables field
          collection to iterate selectively over the global fields
          (such as the transformation gradient F and first
          Piola-Kirchhoff stress P.
      **/
      explicit iterator(const iterable_proxy & it, bool begin = true)
          : it{it}, strain_map{internal::TupleBuilder<Strains_t>::build(
                        std::remove_cv_t<StrainFieldTup>(it.strain_field))},
            stress_map{internal::TupleBuilder<Stresses_t>::build(
                std::remove_cv_t<StressFieldTup>(it.stress_tup))},
            index{begin ? 0 : it.material.size()} {}

      //! Copy constructor
      iterator(const iterator & other) = default;

      //! Move constructor
      iterator(iterator && other) = default;

      //! Destructor
      virtual ~iterator() = default;

      //! Copy assignment operator
      iterator & operator=(const iterator & other) = default;

      //! Move assignment operator
      iterator & operator=(iterator && other) = default;

      //! pre-increment
      inline iterator & operator++();
      //! dereference
      inline value_type operator*();
      //! inequality
      inline bool operator!=(const iterator & other) const;

     protected:
      const iterable_proxy & it;  //!< ref to the proxy
      Strains_t strain_map;       //!< map onto the global strain field
      //! map onto the global stress field and possibly tangent stiffness
      Stresses_t stress_map;
      size_t index;  //!< index or pixel currently referred to
    };

    //! returns iterator to first pixel if this material
    iterator begin() { return std::move(iterator(*this)); }
    //! returns iterator past the last pixel in this material
    iterator end() { return std::move(iterator(*this, false)); }

   protected:
    MaterialBase & material;      //!< reference to the proxied material
    StrainFieldTup strain_field;  //!< cell's global strain field
    //! references to the global stress field and perhaps tangent
    StressFieldTup stress_tup;
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class Strains_t, class Stresses_t, SplitCell is_cell_split>
  bool MaterialBase<DimS, DimM>::iterable_proxy<Strains_t, Stresses_t,
                                                is_cell_split>::iterator::
  operator!=(const iterator & other) const {
    return (this->index != other.index);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class Strains_t, class Stresses_t, SplitCell is_cell_split>
  typename MaterialBase<DimS, DimM>::template iterable_proxy<
      Strains_t, Stresses_t, is_cell_split>::iterator &
  MaterialBase<DimS, DimM>::iterable_proxy<Strains_t, Stresses_t,
                                           is_cell_split>::iterator::
  operator++() {
    this->index++;
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class Strains_t, class Stresses_t, SplitCell is_cell_split>
  typename MaterialBase<DimS, DimM>::template iterable_proxy<
      Strains_t, Stresses_t, is_cell_split>::iterator::value_type
      MaterialBase<DimS, DimM>::iterable_proxy<Strains_t, Stresses_t,
                                               is_cell_split>::iterator::
      operator*() {
    const Ccoord_t<DimS> pixel{
        this->it.material.get_collection().get_ccoord(this->index)};

    auto && strains = apply(
        [&pixel](auto &&... strain_and_rate) {
          return std::make_tuple(strain_and_rate[pixel]...);
        },
        this->strain_map);

    auto && ratio = 1.0;
    if (is_cell_split != SplitCell::no) {
      ratio = this->it.material.get_assigned_ratio(pixel);
    }

    auto && stresses = apply(
        [&pixel](auto &&... stress_tgt) {
          return std::make_tuple(stress_tgt[pixel]...);
        },
        this->stress_map);
    return value_type(std::move(strains), std::move(stresses), this->index,
                      ratio);
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_BASE_HH_
