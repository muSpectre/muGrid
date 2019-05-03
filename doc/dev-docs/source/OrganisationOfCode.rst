
.. _code_organisation:


Organisation of the Code
########################

*µ*\Spectre's code base is split in three components which might be separated into different projects in the future, and this logical separation is already apparent in the directory structure. The three components are
1. *µ*\Grid
2. *µ*\FFT
3. *µ*\Spectre proper

At the lowest level, the header-only library *µ*\Grid, contains a set of tools to define and interact with mathematical fields discretised on a regular spatial grid as used by the `FFT <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`_. It is discussed in more detail in its :ref:`own section <mugrid>`.

On top of *µ*\Grid the library *µ*\FFT provides an uniform interface for multiple FFT implementations.

And finally *µ*\Spectre itself makes use of the two lower-level libraries and defines all the abstractions and classes to define material behaviours and to solve mechanics problems.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   muGrid
