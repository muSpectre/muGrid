Generic Linear Elastic Material
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generic linear elastic material is implemented in the classes
:cpp:class:`MaterialLinearElasticGeneric1<muSpectre::MaterialLinearElasticGeneric1>` and
:cpp:class:`MaterialLinearElasticGeneric2<muSpectre::MaterialLinearElasticGeneric2>`, and is defined solely by
the elastic stiffness tensor :math:`\mathbb{C}`, which has to be specified in
`Voigt notation <https://en.wikipedia.org/wiki/Voigt_notation>`_. In the case of
:cpp:class:`MaterialLinearElasticGeneric2<muSpectre::MaterialLinearElasticGeneric2>`, additionally, a per-pixel
eigenstrain :math:`\bar{\boldsymbol{\varepsilon}}` can be supplied. The
constitutive relation between the Cauchy stress :math:`\boldsymbol{\sigma}` and
the small strain tensor :math:`\boldsymbol{\varepsilon}` is given by

.. math::
   :nowrap:

   \begin{align}
   \boldsymbol{\sigma} &= \mathbb{C}:\boldsymbol{\varepsilon}\\
   \sigma_{ij} &= C_{ijkl}\,\varepsilon_{kl}, \quad\mbox{for the simple version, and}\\
   \boldsymbol{\sigma} &= \mathbb{C}:\left(\boldsymbol{\varepsilon}-\bar{\boldsymbol{\varepsilon}}\right)\\
   \sigma_{ij} &= C_{ijkl}\,\left(\varepsilon_{kl} - \bar\varepsilon_{kl}\right) \quad\mbox{ for the version with eigenstrain}
   \end{align}

This implementation is convenient, as it covers all possible linear elastic
behaviours, but it is by far not as efficient as
:cpp:class:`MaterialLinearElastic1<muSpectre::MaterialLinearElastic1>` for isotropic linear elasticity.

This law can be used in both small strain and finite strain calculations.

The following snippet shows how to use this law in python to implement isotropic
linear elasticity:

Python Usage Example
````````````````````
.. code-block:: python

   C = np.array([[2 * mu + lam,          lam,          lam,  0,  0,  0],
                 [         lam, 2 * mu + lam,          lam,  0,  0,  0],
                 [         lam,          lam, 2 * mu + lam,  0,  0,  0],
                 [           0,            0,            0, mu,  0,  0],
                 [           0,            0,            0,  0, mu,  0],
                 [           0,            0,            0,  0,  0, mu]])

   eigenstrain = np.array([[  0, .01],
                           [.01,   0]])

   mat1 = muSpectre.material.MaterialLinearElasticGeneric1_3d.make(
       cell, "material", C)
   mat1.add_pixel(pixel)
   mat2 = muSpectre.material.MaterialLinearElasticGeneric2_3d.make(
       cell, "material", C)
   mat2.add_pixel(pixel, eigenstrain)

