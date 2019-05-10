FROM ubuntu:18.04
MAINTAINER Till Junge <till.junge@epfl.ch>

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

## for apt to be noninteractive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

RUN apt-get -qq update && apt-get -y -qq install libfftw3-dev libfftw3-mpi-dev libeigen3-dev libboost-test-dev python3-dev curl python3-numpy python3-scipy python3-breathe python3-sphinx python3-sphinx-rtd-theme python3-pytest python3-setuptools git doxygen g++ clang cmake && rm -rf /var/lib/apt/lists/*
# apt-get on one line due to https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run


