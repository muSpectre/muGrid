FROM debian:stretch
MAINTAINER Till Junge <till.junge@epfl.ch>

RUN apt-get -qq update && apt-get -y -qq install libfftw3-dev locales libfftw3-mpi-dev libeigen3-dev libboost-test-dev python3-dev curl python3-numpy python3-scipy python3-breathe python3-sphinx python3-sphinx-rtd-theme python3-pytest git doxygen python3-mpi4py python3-setuptools g++ clang cmake && rm -rf /var/lib/apt/lists/*
# apt-get on one line due to https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run

# generating utf8 locales is required by the tests for debian stable
RUN  sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen &&     dpkg-reconfigure --frontend=noninteractive locales &&  update-locale LANG=en_US.UTF-8
ENV LANG C.UTF-8


