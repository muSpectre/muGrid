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
    ccache \
    meson \
    ninja \
    python3 python3-dev \
    py3-pip \
    py3-numpy \
    py3-scipy \
    netcdf-dev \
    hdf5-dev \
    jq

RUN python3 -m pip install meshio[all]>=4.3.10 build uvw
