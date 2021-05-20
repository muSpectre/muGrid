FROM debian:testing
MAINTAINER Till Junge <till.junge@epfl.ch>

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
    python3-meshio \
    python3-mpi4py \
    python3-numpy \
    python3-pip \
    python3-pytest \
    python3-scipy \
    python3-setuptools \
    python3-sphinx \
    python3-sphinx-rtd-theme \
    valgrind \
    && rm -rf /var/lib/apt/lists/*

# apt-get on one line due to https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run

RUN python3 -m pip install uvw

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

RUN cd /tmp \
    && git clone https://github.com/mpip/pfft.git \
    && cd pfft \
    && ./bootstrap.sh \
    && CFLAGS="-O2 -march=sandybridge -mno-avx512f" ./configure --disable-fortran --enable-shared --prefix=/usr/local \
    && make -j 4 \
    && make install