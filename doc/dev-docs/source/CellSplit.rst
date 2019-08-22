CellSplit
~~~~~~~~~~

The implementation of split cell class is :cpp:class:`CellSplit<muSpectre::CellSplit>`. In order to compile this class one should set the cmake variable ``SPLIT_CELL`` **ON** in the configuration.

Simulating multi-phase structures with muSpectre involves pixels which share material as they may lie in the interface of phases. Different homogenisation schemes can be used for substituting pixels if the effective media consists of two or more phases. One of these approximations assuming iso-strain pixels is the Voigt method. In the :cpp:class:`CellSplit<muSpectre::CellSplit>` pixels' effective constitutive behavior is approximated as the weighted average of the constituent materials w.r.t their volume fractions :math:`(\alpha)`

.. math::
   :nowrap:

       \begin{align}
       \boldsymbol{P}^l &= f\big(\boldsymbol{F}^l\big) &,
       \boldsymbol{P}^r &= f\big(\boldsymbol{F}^r\big)\tag{1}\\
       \boldsymbol{F} &= \boldsymbol{F}^r &,
       \boldsymbol{F} &= \boldsymbol{F}^l\tag{2}\\
       \overline{\boldsymbol{P}} &= \langle\boldsymbol{P}\rangle\tag{3}\\
       \end{align}

where

.. math::
   :nowrap:

       \begin{align}
       \langle\boldsymbol{P}\rangle &= \alpha^l \boldsymbol{P}^l + \alpha^r \boldsymbol{P}^r\tag{4}.
       \end{align}

The superscripts :math:`(l)` and :math:`(r)` show the two constituent materials of the pixel and :math:`(\boldsymbol{P})`, :math:`(\boldsymbol{F})` are, respectively, first Piola-Kirchhoff and deforamtion gradient tensors. :math:`(\alpha)` is the volume fraction of the phases.The :cpp:class:`CellSplit<muSpectre::CellSplit>` inherits from :cpp:class:`CellBase<muSpectre::CellBase>` and can be used in its stead. Currently, all materials inheriting from :cpp:class:`MaterialMuSpectre<muSpectre::MaterialMuSpectre>` can be added to an instance of :cpp:class:`CellSplit<muSpectre::CellSplit>`. However, it should be noted that for adding pixel to the materials contained in this type of cell, :cpp:func:`add_pixel_split()<muSpectre::MaterialMuSpectre::add_pixel_split>` sould be employed instead of plain :cpp:func:`add_pixel_split()<muSpectre::MaterialMuSpectre::add_pixel>`. This function takes the ratio of the materials in the pixel that is being assigned to it as an input parameter. It is notable that the summation of ratio of materials should add up to unity for all the pixels in the cell.

Specialised function :cpp:func:`make_automatic_precipitate_split_pixels()<muSpectre::CellSplit::make_automatic_precipitate_split_pixels>` exists in :cpp:class:`CellSplit<muSpectre::CellSplit>` which enables user to assign materials based on the material and geometry of precipitates (as a set of coordinates composing a polyheron/polygon in 3D/2D). Moreover, one can use the function  :cpp:func:`complete_material_assignment()<muSpectre::CellSplit::complete_material_assignment>` in order to assign the pixels whose assignments are not completed to a specific material. The following snippet shows how one can use the machinery to employ this specific kind of Cell in µSpectre.

Python Usage Example
````````````````````
.. code-block:: python

                rve = msp.Cell(res, lengths,
                               formulation, None, 'fftw', None,
                               msp.SplitCell.split)

                mat1 = msp.material.MaterialLinearElastic1_2d.make(
                   rve, "mat1", E1, .noo1)

                mat2 = msp.material.MaterialLinearElastic1_2d.make(
                   rve, "mat2",  E2, .noo2)

                points = np.ndarray(shape=(num, 2))
                for j, tetha in enumerate(np.linspace(0, 2*np.pi, num, endpoint=false)):
                    points[j, 0] = center[0] + radius*np.cos(tetha)
                    points[j, 1] = center[1] + radius*np.sin(tetha)

                points_list = [points.tolist()]

                rve.make_precipitate(mat1, points_list)
                rve.complete_material_assignemnt_simple(mat2)
