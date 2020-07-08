FROM debian:stable
MAINTAINER Till Junge <till.junge@epfl.ch>

ARG NETCDF4_PYTHON_VERSION=1.5.1.2

RUN apt-get -qq update && apt-get -y -qq install \
    clang \
    cmake \
    curl \
    doxygen \
    graphviz \
    g++ \
    git \
    libboost-test-dev \
    libcurl4-openssl-dev \
    libeigen3-dev \
    libfftw3-dev \
    libfftw3-mpi-dev \
    libgmp-dev \
    libhdf5-mpi-dev \
    libpnetcdf-dev \
    libnetcdf-dev \
    locales \
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

# generating utf8 locales is required by the tests for debian stable
RUN  sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen &&     dpkg-reconfigure --frontend=noninteractive locales &&  update-locale LANG=en_US.UTF-8
ENV LANG C.UTF-8

RUN python3 -m pip install netCDF4==${NETCDF4_PYTHON_VERSION}
RUN python3 -m pip install uvw
