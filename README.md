# µGrid

µGrid is a library for discrete representation of fields on structured grids.
A *field* is a physical quantity that varies in space. µGrid makes it easy to
implement algorithms that operate on fields, such as solving partial
differential equations. It supports parallelization using domain decomposition
implemented using the Message Passing Interface (MPI).

µGrid is written in C++ and currently has language bindings for
[Python](https://www.python.org/).

This README contains only a small quick start guide. Please refer to the
[full documentation](https://muspectre.github.io/muGrid/) for more help.

## Quick start

To install µGrid, run

    pip install muGrid

Note that on most platforms this will install a binary wheel, that was
compiled with a minimal configuration. To compile for your specific platform
use

    pip install -v --no-binary muGrid muGrid

which will compile the code. µGrid will autodetect
[MPI](https://www.mpi-forum.org/).
For I/O, it will try to use
[Unidata NetCDF](https://www.unidata.ucar.edu/software/netcdf/)
for serial builds and
[PnetCDF](https://parallel-netcdf.github.io/) for MPI-parallel builds.
Monitor output to see which of these options were automatically detected.

## Funding

This development has received funding from the
[Swiss National Science Foundation](https://www.snf.ch/en)
within an Ambizione Project and by the
[European Research Council](https://erc.europa.eu) within
[Starting Grant 757343](https://cordis.europa.eu/project/id/757343).
