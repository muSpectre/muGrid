FROM alpine:3.16

ENV PYTHONUNBUFFERED 1

#
# System dependencies
#
RUN apk add \
    libcap \
    bash \
    alpine-sdk \
    valgrind \
    libexecinfo libexecinfo-dev \
    cmake \
    meson \
    ninja \
    python3 \
    py3-pip \
    py3-numpy \
    py3-scipy \
    py3-pybind11-dev \
    eigen-dev \
    netcdf-dev \
    hdf5-dev

RUN python3 -m pip install meshio[all]>=4.3.10 build uvw
