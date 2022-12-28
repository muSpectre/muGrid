FROM debian:testing
MAINTAINER Till Junge <till.junge@epfl.ch>

RUN apt-get -qq update && apt-get -y -qq install \
    clang \
    cmake \
    curl \
    doxygen \
    g++ \
    git \
    libboost-test-dev \
    libcurl4-openssl-dev \
    libfftw3-dev \
    libfftw3-mpi-dev \
    libgmp-dev \
    libopenmpi-dev \
    libopenblas-dev \
    libpnetcdf-dev \
    libnetcdf-dev \
    meson \
    m4 \
    ninja-build \
    pybind11-dev \
    python3-breathe \
    python3-dev \
    python3-mpi4py \
    python3-numpy \
    python3-pip \
    python3-pytest \
    python3-scipy \
    python3-sphinx \
    python3-sphinx-rtd-theme \
    python3-venv \
    valgrind \
    gcovr \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# apt-get on one line due to https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run

RUN python3 -m pip install --no-binary numpy --no-binary scipy uvw meshio[all]>=4.3.10 build twine cibuildwheel
