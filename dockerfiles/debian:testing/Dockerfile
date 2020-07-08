FROM debian:testing
MAINTAINER Till Junge <till.junge@epfl.ch>

ARG HDF5_VERSION=1.10.5
ARG NETCDF4_VERSION=4.7.0
ARG NETCDF4_PYTHON_VERSION=1.5.1.2

RUN apt-get -qq update && apt-get -y -qq install \
    clang \
    cmake \
    curl \
    doxygen \
    g++ \
    git \
    pybind11-dev \
    libboost-test-dev \
    libcurl4-openssl-dev \
    libeigen3-dev \
    libfftw3-dev \
    libfftw3-mpi-dev \
    libgmp-dev \
    libopenmpi-dev \
    libpnetcdf-dev \
    libnetcdf-dev \
    m4 \
    python3-breathe \
    python3-dev \
    python3-mpi4py \
    python3-numpy \
    python3-pip \
    python3-pytest \
    python3-scipy \
    python3-setuptools \
    python3-sphinx \
    python3-sphinx-rtd-theme \
    && rm -rf /var/lib/apt/lists/*

# apt-get on one line due to https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run

# We need to compile HDF5 ourselves because the MPI-parallel debian package is
# broken.

RUN curl https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.gz | tar -xzC /tmp \
  && cd /tmp/hdf5-${HDF5_VERSION} \
  && CC=mpicc CXX=mpicxx ./configure --enable-parallel --prefix=/usr/local \
  && make \
  && make install

# We need to compile NetCDF ourselves because there is no package that has
# parallel PnetCDF and HDF5 enabled.

RUN curl https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-c-${NETCDF4_VERSION}.tar.gz | tar -xzC /tmp \
  && mkdir /tmp/netcdf-c-build \
  && cd /tmp/netcdf-c-build \
  && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DUSE_PARALLEL=ON -DENABLE_PARALLEL4=ON -DENABLE_PNETCDF=ON /tmp/netcdf-c-${NETCDF4_VERSION} \
  && make \
  && make install

# Install netcdf4-python and make sure that it is compiled (no-binary),
# otherwise it will not have parallel support.

RUN CC=mpicc python3 -m pip install --no-binary netCDF4 netCDF4==${NETCDF4_PYTHON_VERSION}
RUN python3 -m pip install uvw
