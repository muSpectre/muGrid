Laminate Material
~~~~~~~~~~~~~~~~~

The generic laminate material is implemented in :cpp:class:`MaterialLaminate<muSpectre::MaterialLaminate>` inheriting directly from :cpp:class:`MaterialBase<muSpectre::MaterialBase>`, and it contains two arbitrary underlting materials(with either linear or nonlinear constitutive laws). In order to compile this class one should set the cmake variable ``SPLIT_CELL`` **ON** in the configuration.

The underlying materials behave as they consist a laminate. The resultant constitutive behavior depends on both of their constitutive laws, the volume fraction of phases (:math:`(\alpha^l)`, :math:`(\alpha^r)`), and the normal vector of their interface. The formulation governing the stress and the deformation gradient of the underlying phases is given by:

.. math::
   :nowrap:

       \begin{align}
       \boldsymbol{F} &= \alpha^l\boldsymbol{F}^l + \alpha^r\boldsymbol{F}^r \tag{1}\\
       \boldsymbol{P} &= \alpha^l\boldsymbol{P}^l + \alpha^r\boldsymbol{P}^r \tag{2}\\
       \boldsymbol{P}^l\ \cdot \boldsymbol{n} &= \boldsymbol{P}^r\ \cdot \boldsymbol{n} \tag{3}\\
       \boldsymbol{F}^l\ \cdot \big( \mathbb{I} - \boldsymbol{n} \otimes \boldsymbol{n} \big) &= \boldsymbol{F}^r\ \cdot \big( \mathbb{I} - \boldsymbol{n} \otimes \boldsymbol{n} \big)\tag{4}
       \end{align}

where, The superscripts :math:`(l)` and :math:`(r)` show the two constituent materials of the pixel and :math:`(\boldsymbol{P})`, :math:`(\boldsymbol{F})` are, respectively, first Piola-Kirchhoff and deforamtion gradient tensors. :math:`(\alpha)`s are the volume fraction of the phases. :math:`(\boldsymbol{n})` is the normal vector of phases' interface and :math:`(\mathbb{I})` is the fourth order identity matrix. Equations :math:`(3)` and  :math:`(4)` are the equilibrium and the compatibility equations on the pahses' interface in the laminate structure. By having deformations as the input (µSpectre) and from equation :math:`(4)` some components of deforamtion of both phases are easily derived. For calculating the remaining components, it is necessary to solve equation :math:`(3)`, which in the most general case is a nonlinear equation depending on both materials' constiturive laws. Accordingly this material's :cpp:func:`evaluate_stress()<muSpectre::MaterialLaminate::evaluate_stress()>` calls an internal solver implemented in :cpp:func:`laminate_solver()<muSpectre::LamHomogen::laminate_solver>` where equation :math:`(3)` is solved, per-pixel, employing both underlying materials' constitutive laws. Accordingly, this material is not expected to be as efficient as materials inheriting from :cpp:class:`MaterialMuSpectre<muSpectre::MaterialMuSpectre>`.

:cpp:class:`MaterialLaminate<muSpectre::MaterialLaminate>` at creation only needs a name. However, its :cpp:func:`add_pixel()<muSpectre::MaterialLaminate::add_pixel>` takes a pixel and pointers two the underlying materials for each pixel as well as volume fraction and interface normal vector for each pixel. For convinience, function :cpp:func:`make_pixels_precipitate_for_laminate_material()<muSpectre::CellBase::make_automatic_precipitate_split_pixels>` has been added to :cpp:class:`CellBase<muSpectre::CellBase>` using which user can add pixels to a :cpp:class:`MaterialLaminate<muSpectre::MaterialLaminate>` object by introducing the shape of a precipitate, it's material an the base material of the matrix media in which the precipitate lies. In addition, :cpp:func:`complete_material_assignment_simple()<muSpectre::CellBase::complete_material_assignment_simple>` enables to assign the remaining of the pixles (unassigned) pixels to a material(the base material of the matrix media).The following snippet shows how one can use the machinery to employ this specific Material in µSpectre.

Python Usage Example
````````````````````
 .. code-block:: python

                 rve = msp.Cell(res,
                                lengths,
                                formulation)

                 mat1_laminate = msp.material.MaterialLinearElastic1_2d.make_free(
                    "mat1_free", E1, noo)

                 mat2_laminate = msp.material.MaterialLinearElastic1_2d.make_free(
                    "mat2_free", E2, noo)

                 mat1 = msp.material.MaterialLinearElastic1_2d.make(
                     rve, "mat1", E1, noo)

                 mat2 = msp.material.MaterialLinearElastic1_2d.make(
                     rve, "mat2",  E2, noo)

                 mat_lam = msp.material.MaterialLaminate_2d.make(rve, "laminate")

                 points = np.ndarray(shape=(num, 2))
                 for j, tetha in enumerate(np.linspace(0, 2*np.pi, num, endpoint=false)):
                     points[j, 0] = center[0] + radius*np.cos(tetha)
                     points[j, 1] = center[1] + radius*np.sin(tetha)

                 points_list = [points.tolist()]

                 rve.make_precipitate_laminate(mat_lam, mat1,
                                               mat1_laminate,
                                               mat2_laminate,
                                               points_list)

                 rve.complete_material_assignemnt_simple(mat2)
