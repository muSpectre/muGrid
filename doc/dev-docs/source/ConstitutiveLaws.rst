
.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. _constitutive_laws:


Constitutive Laws
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   MaterialLinearElasticGeneric
   CellSplit
   MaterialLaminate

Testing Constitutive Laws
~~~~~~~~~~~~~~~~~~~~~~~~~

When writing new constitutive laws, the ability to evaluate the stress-strain
behaviour and the tangent moduli is convenient, but *µ*\Spectre's material model
makes it cumbersome to isolate and execute the :cpp:func:`evaluate_stress` and
:cpp:func:`evaluate_stress_tangent` methods than any daughter class of
:cpp:class:`MaterialMuSpectre<muSpectre::MaterialMuSpectre>` must implement
(e.g., :cpp:func:`evaluate_stress()
<muSpectre::MaterialLinearElastic1::evaluate_stress>`). As a helper object,
*µ*\Spectre offers the class
:cpp:class:`MaterialEvaluator<muSpectre::MaterialEvaluator>` to facilitate
precisely this:

A :cpp:class:`MaterialEvaluator<muSpectre::MaterialEvaluator>` object can be
constructed with a shared pointer to a
:cpp:class:`MaterialBase<muSpectre::MaterialBase>` and exposes functions to
evaluate just the stress, both the stress and tangent moduli, or a numerical
approximation to the tangent moduli. For materials with internal history
variables, :cpp:class:`MaterialEvaluator<muSpectre::MaterialEvaluator>` also
exposes the
:cpp:func:`MaterialBase::save_history_variables()<muSpectre::MaterialBase::save_history_variables>`
method. As a convenience function, all daughter classes of
:cpp:class:`MaterialMuSpectre<muSpectre::MaterialMuSpectre>` have the static
factory function :cpp:func:`make_evaluator()
<muSpectre::MaterialMuSpectre::make_evaluator>` to create a material and its
evaluator at once.  See the :ref:`reference` for the full class description.

Python Usage Example
````````````````````

.. code-block:: python

   import numpy as np
   from muSpectre import material
   from muSpectre import Formulation

   # MaterialLinearElastic1 is standard linear elasticity while
   # MaterialLinearElastic2 has a per pixel eigenstrain which needs to be set
   LinMat1, LinMat2 = (material.MaterialLinearElastic1_2d,
                       material.MaterialLinearElastic2_2d)

   young, poisson = 210e9, .33

   # the factory returns a material and it's corresponding evaluator
   material1, evaluator1 = LinMat1.make_evaluator(young, poisson)

   # the material is empty (i.e., does not have any pixel/voxel), so a pixel
   # needs to be added. The coordinates are irrelevant, there just needs to
   # be one pixel.
   material1.add_pixel([0,0])

   # the stress and tangent can be evaluated for finite strain
   F = np.array([[1., .01],[0, 1.0]])
   P, K = evaluator1.evaluate_stress_tangent(F, Formulation.finite_strain)
   # or small strain
   eps = .5 * ((F-np.eye(2)) + (F-np.eye(2)).T)
   sigma, C = evaluator1.evaluate_stress_tangent(eps, Formulation.small_strain)

   # and the tangent can be checked against a numerical approximation
   Delta_x = 1e-6
   num_C = evaluator1.estimate_tangent(eps, Formulation.small_strain, Delta_x)


   # Materials with per-pixel data behave similarly: the factory returns a
   # material and it's corresponding evaluator like before
   material2, evaluator2 = LinMat2.make_evaluator(young, poisson)

   # when adding the pixel, we now need to specify also the per-pixel data:
   eigenstrain = np.array([[.01, .002], [.002, 0.]])
   material2.add_pixel([0,0], eigenstrain)



C++ Usage Example
`````````````````

.. code-block:: c++

   #include"materials/material_linear_elastic2.hh"
   #include "materials/material_evaluator.hh"
   #include <libmugrid/T4_map_proxy.hh>

   #include "Eigen/Dense"

   using Mat_t = MaterialLinearElastic2<twoD, twoD>;

   constexpr Real Young{210e9};
   constexpr Real Poisson{.33};

   auto mat_eval{Mat_t::make_evaluator(Young, Poisson)};
   auto & mat{*std::get<0>(mat_eval)};
   auto & evaluator{std::get<1>(mat_eval)};


   using T2_t = Eigen::Matrix<Real, twoD, twoD>;
   using T4_t = T4Mat<Real, twoD>;
   const T2_t F{(T2_t::Random() - (T2_t::Ones() * .5)) * 1e-4 +
                T2_t::Identity()};

   T2_t eigen_strain{[](auto x) {
     return 1e-4 * (x + x.transpose());
   }(T2_t::Random() - T2_t::Ones() * .5)};

   mat.add_pixel({}, eigen_strain);

   T2_t P{};
   T4_t K{};

   std::tie(P, K) =
     evaluator.evaluate_stress_tangent(F, Formulation::finite_strain);
