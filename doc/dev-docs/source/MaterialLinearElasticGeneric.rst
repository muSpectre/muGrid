Generic Linear Elastic Material
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generic linear elastic material is implemented in the class :cpp:class:`muSpectre::MaterialLinearElasticGeneric`, and is defined solely by the elastic stiffness tensor :math:`\mathbb{C}`, which has to be specified in `Voigt notation <https://en.wikipedia.org/wiki/Voigt_notation>`_. The constitutive relation between the Cauchy stress :math:`\boldsymbol{\sigma}` and the small strain tensor :math:`\boldsymbol{\varepsilon}` is given by

.. math::
   :nowrap:

   \begin{align}
   \boldsymbol{\sigma} &= \mathbb{C}:\boldsymbol{\varepsilon}\\
   \sigma_{ij} &= C_{ijkl}\,\varepsilon_{kl}
   \end{align}

This implementation is convenient, as it covers all possible linear elastic behaviours, but it is by far not as efficient as :cpp:class:`muSpectre::MaterialLinearElastic1` for isotropic linear elasticity.

This law can be used in both small strain and finite strain calculations.

The following snippet shows how to use this law in python to implement isotropic linear elasticity:

.. code-block:: python

   C = np.array([[2 * mu + lam,          lam,          lam,  0,  0,  0],
                 [         lam, 2 * mu + lam,          lam,  0,  0,  0],
                 [         lam,          lam, 2 * mu + lam,  0,  0,  0],
                 [           0,            0,            0, mu,  0,  0],
                 [           0,            0,            0,  0, mu,  0],
                 [           0,            0,            0,  0,  0, mu]])

   mat = muSpectre.material.MaterialLinearElasticGeneric_3d.make(
       cell, "material", C)

