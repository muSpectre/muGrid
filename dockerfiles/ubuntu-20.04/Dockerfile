FROM ubuntu:20.04
MAINTAINER Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

## for apt to be noninteractive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

RUN apt-get -qq update && apt-get -y -qq install \
    clang \
    cmake \
    curl \
    doxygen \
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
    meson \
    m4 \
    ninja-build \
    python3-pip \
    python3-breathe \
    python3-dev \
    python3-netcdf4 \
    python3-meshio \
    python3-mpi4py \
    python3-numpy \
    python3-pytest \
    python3-scipy \
    python3-setuptools \
    python3-sphinx \
    python3-sphinx-rtd-theme \
    python3-venv \
    gcovr \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# apt-get on one line due to https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run

RUN python3 -m pip install uvw build twine cibuildwheel
