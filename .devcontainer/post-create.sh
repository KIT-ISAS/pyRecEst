#!/usr/bin/env bash
set -euxo pipefail

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  gfortran \
  libcfitsio-dev \
  libcurl4-openssl-dev \
  libfftw3-dev \
  libgeos-dev \
  libhealpix-cxx-dev \
  liblapack-dev \
  libopenblas-dev \
  ninja-build \
  pkg-config \
  zlib1g-dev

python -m pip install --user --upgrade pip wheel "poetry>=2.0,<3.0" "poetry-plugin-export>=1.8,<2.0"
python -m poetry env use python
python -m poetry install --with dev --extras "healpy_support" --extras "pytorch_support"
