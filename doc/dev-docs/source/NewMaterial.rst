Writing a New Constitutive Law
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The abstraction for a constitutive law in *µ*\Spectre** is the ``Material``, and
new such materials must inherit from the class
:cpp:class:`muSpectre::MaterialBase`. Most often, however, it will be most
convenient to inherit from the derived class
:cpp:class:`muSpectre::MaterialMuSpectre`, as it implements a lot of the
machinery that is commonly used by constitutive laws. This section describes how
to implement a new constitutive law with internal variables (sometimes also
called state variables). The example material implemented here is
:cpp:class:`MaterialTutorial`, an objective linear elastic law with a
distribution of eigenstrains as internal variables. The constitutive law is
defined by the relationship between material (or second Piola-Kirchhoff) stress
:math:`\mathbf{S}` and Green-Lagrange strain :math:`\mathbf{E}`

.. math::
   :nowrap:

   \begin{align}
      \mathbf{S} &= \mathbb C:\left(\mathbf{E}-\mathbf{e}\right), \\
      S_{ij} &= C_{ijkl}\left(E_{kl}-e_{kl}\right), \\
   \end{align}

where :math:`\mathbb C` is the elastic stiffness tensor and :math:`\mathbf e` is
the local eigenstrain. Note that the implementation sketched out here is the
most convenient to quickly get started with using *µ*\Spectre**, but not the
most efficient one. For a most efficient implementation, refer to the
implementation of :cpp:class:`muSpectre::MaterialLinearElastic2`.

The :cpp:class:`muSpectre::MaterialMuSpectre` class
***************************************************

The class :cpp:class:`muSpectre::MaterialMuSpectre` is defined in
``material_muSpectre_base.hh`` and takes three template parameters;

#. ``class Material`` is a `CRTP
   <https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern>`_ and
   names the material inheriting from it. The reason for this construction is
   that we want to avoid virtual method calls from
   :cpp:class:`muSpectre::MaterialMuSpectre` to its derived classes. Rather, we
   want :cpp:class:`muSpectre::MaterialMuSpectre` to be able to call methods of
   the inheriting class directly without runtime overhead.
#. ``Dim_t DimS`` defines the number of spatial dimensions of the problem, i.e.,
   whether we are dealing with a two- or three-dimensional grid of
   pixels/voxels.
#. ``Dim_t DimM`` defines the number of dimensions of our material
   description. This value will typically be the same as ``DimS``, but in cases
   like generalised plane strain, we can for instance have a three
   three-dimensional material response in a two-dimensional pixel grid.

The main job of :cpp:class:`muSpectre::MaterialMuSpectre` is to

#. loop over all pixels to which this material has been assigned, transform the
   global gradient :math:`\mathbf{F}` (or small strain tensor
   :math:`\boldsymbol\varepsilon`) into the new material's required strain
   measure (e.g., the Green-Lagrange strain tensor :math:`\mathbf{E}`),
#. for each pixel evaluate the constitutive law by calling its
   ``evaluate_stress`` (computes the stress response) or
   ``evaluate_stress_tangent`` (computes both stress and consistent tangent)
   method with the local strain and internal variables, and finally
#. transform the stress (and possibly tangent) response from the material's
   stress measure into first Piola-Kirchhoff stress :math:`\mathbf{P}` (or
   Cauchy stress :math:`\boldsymbol\sigma` in small strain).

In order to use these facilities, the new material needs to inherit from
:cpp:class:`muSpectre::MaterialMuSpectre` (where we calculation of the response)
and specialise the type :cpp:class:`muSpectre::MaterialMuSpectre_traits` (where
we tell :cpp:class:`muSpectre::MaterialMuSpectre` how to use the new
material). These two steps are described here for our example material.

Specialising the :cpp:class:`muSpectre::MaterialMuSpectre_traits` structure
*************************************************************************** This
structure is templated by the new material (in this case
:cpp:class:`MaterialTutorial`) and needs to specify

#. the types used to communicate per-pixel strains, stresses and stiffness
   tensors to the material (i.e., whether you want to get maps to `Eigen`
   matrices or raw pointers, or ...). Here we will use the convenient
   :cpp:type:`muSpectre::MatrixFieldMap` for strains and stresses, and
   :cpp:type:`muSpectre::T4MatrixFieldMap` for the stiffness. Look through the
   classes deriving from :cpp:type:`muSpectre::FieldMap` for all available
   options.
#. the strain measure that is expected (e.g., gradient, Green-Lagrange strain,
   left Cauchy-Green strain, etc.). Here we will use Green-Lagrange strain. The
   options are defined by the enum :cpp:enum:`muSpectre::StrainMeasure`.
#. the stress measure that is computed by the law (e.g., Cauchy, first
   Piola-Kirchhoff, etc,). Here, it will be first Piola-Kirchhoff stress. The
   available options are defined by the enum
   :cpp:enum:`muSpectre::StressMeasure`.

Our traits look like this (assuming we are in the namespace ``muSpectre``::

  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialTutorial<DimS, DimM>>
  {
    //! global field collection
    using GFieldCollection_t = typename
      GlobalFieldCollection<DimS, DimM>;

    //! expected map type for strain fields
    using StrainMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM, true>;
    //! expected map type for stress fields
    using StressMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t = T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};

    //! local field_collections used for internals
    using LFieldColl_t = LocalFieldCollection<DimS, DimM>;
    //! local strain type
    using LStrainMap_t = MatrixFieldMap<LFieldColl_t, Real, DimM, DimM, true>;
    //! elasticity with eigenstrain
    using InternalVariables = std::tuple<LStrainMap_t>;

  };

Implementing the new material
*****************************

The new law needs to implement the methods ``add_pixel``, ``get_internals``,
``evaluate_stress``, and ``evaluate_stress_tangent``. Below is a commented
example header::

  template <Dim_t DimS, Dim_t DimM>
  class MaterialTutorial:
    public MaterialMuSpectre<MaterialTutorial<DimS, DimM>, DimS, DimM>
  {
  public:
    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialTutorial>;

    //! Type of container used for storing eigenstrain
    using InternalVariables = typename traits::InternalVariables;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialTutorial(std::string name, Real young, Real poisson);

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     */
    template <class s_t, class eigen_s_t>
    inline decltype(auto) evaluate_stress(s_t && E, eigen_s_t && E_eig);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor)
     */
    template <class s_t, class eigen_s_t>
    inline decltype(auto)
    evaluate_stress_tangent(s_t &&  E, eigen_s_t && E_eig);

    /**
     * return the internals tuple (needed by `muSpectre::MaterialMuSpectre`)
     */
    InternalVariables & get_internals() {
      return this->internal_variables;};

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const Ccoord_t<DimS> & pixel,
                   const Eigen::Matrix<Real, DimM, DimM> & E_eig);

  protected:
    //! stiffness tensor
    T4Mat<Real, DimM> C;
    //! storage for eigenstrain
    using Field_t =
      TensorField<LocalFieldCollection<DimS,DimM>, Real, secondOrder, DimM>;
    Field_t & eigen_field; //!< field of eigenstrains
    //! tuple for iterable eigen_field
    InternalVariables internal_variables;
  private:
  };

A possible implementation for the constructor would be::

  template <Dim_t DimS, Dim_t DimM>
  MaterialTutorial<DimS, DimM>::MaterialTutorial(std::string name,
                                                 Real young,
                                                 Real poisson)
    :MaterialMuSpectre<MaterialTutorial, DimS, DimM>(name) {

    // Lamé parameters
    Real lambda{young*poisson/((1+poisson)*(1-2*poisson))};
    Real mu{young/(2*(1+poisson))};

    // Kronecker delta
    Eigen::Matrix<Real, DimM, DimM> del{Eigen::Matrix<Real, DimM, DimM>::Identity()};


    // fill the stiffness tensor
    this->C.setZero();
    for (Dim_t i = 0; i < DimM; ++i) {
      for (Dim_t j = 0; j < DimM; ++j) {
        for (Dim_t k = 0; k < DimM; ++k) {
          for (Dim_t l = 0; l < DimM; ++l) {
            get(this->C, i, j, k, l) += (lambda * del(i,j)*del(k,l) +
                                         mu * (del(i,k)*del(j,l) + del(i,l)*del(j,k)));
          }
        }
      }
    }
  }

as an exercise, you could check how
:cpp:class:`muSpectre::MaterialLinearElastic1` uses *µ*\Spectre**'s materials
toolbox (in namespace ``MatTB``) to compute :math:`\mathbb C` in a much more
convenient fashion. The evaluation of the stress could be (here, we make use of
the ``Matrices`` namespace that defines common tensor algebra operations)::

  template <Dim_t DimS, Dim_t DimM>
  template <class s_t, class eigen_s_t>
  decltype(auto)
  MaterialTutorial<DimS, DimM>::
  evaluate_stress(s_t && E, eigen_s_t && E_eig) {
    return Matrices::tens_mult(this->C, E-E_eig);
  }



The remaining two methods are straight-forward::

  template <Dim_t DimS, Dim_t DimM>
  template <class s_t, class eigen_s_t>
  decltype(auto)
  MaterialTutorial<DimS, DimM>::
  evaluate_stress_tangent(s_t && E, eigen_s_t && E_eig) {
    return return std::make_tuple
          (evaluate_stress(E, E_eig),
           this->C);
  }

  template <Dim_t DimS, Dim_t DimM>
  InternalVariables &
  MaterialTutorial<DimS, DimM>::get_internals() {
    return this->internal_variables;
  }


Note that the methods ``evaluate_stress`` and ``evaluate_stress_tangent`` need
to be in the header, as both their input parameter types and output type depend
on the compile-time context.
