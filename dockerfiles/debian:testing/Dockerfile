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
    libeigen3-dev \
    libfftw3-dev \
    libfftw3-mpi-dev \
    libgmp-dev \
    m4 \
    python3-breathe \
    python3-dev \
    python3-mpi4py \
    python3-numpy \
    python3-pytest \
    python3-scipy \
    python3-setuptools \
    python3-sphinx \
    python3-sphinx-rtd-theme \
    && rm -rf /var/lib/apt/lists/*

# apt-get on one line due to https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run


