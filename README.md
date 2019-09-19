# Gaussian Focused Monte Carlo - GFMC

A Method for Coupling Monte Carlo Simulations of Photon Transport to Gaussian Beam Propagation

From the abstract of Clark and Cook (2019): In this manuscript we present a method for modeling focused Gaussian beams in turbid media. Compared to previous methods based on ray tracing, our method is simple and fast. Rather than sampling photon launch positions from the surface of the media, we directly sample launch positions from inside the sample volume using the probability distribution for the position that a ballistic photon will first interact with the tissue. The examples presented illustrate the importance of accounting for Gaussian beam optics when simulating a focused beam. Not doing so will give an artificially large irradiance at the beam waist position. Thermal models used to predict laser damage to tissue will predict temperature rises that are too large for samples containing the beam focus.

### Contents

1. [Introduction](#intro)
2. [What are all these files for?](#files)
3. [Installation](#install)
4. [User Guide](#begin)

## Introduction <a name="intro">

This repository serves to show example code of Gaussian Focused Monte Carlo (GFMC) for simulating Gaussian beam propagation with Monte Carlo photon transport. This code accompanies "A Method for Coupling Monte Carlo Simulations of Photon Transport to Gaussian Beam Propagation" by Clark and Cook (2019). This code was written on/for Ubuntu 18.04, 18.10, and 19.04.

## What are all these files for? <a name="files">

This repository hosts two implementations of the GFMC method: a Python 3 implementation and a CUDA implementation. The Python implementation is `GFMC.py` and the CUDA implementation is present in the `CUDA/` directory. Code to plot the results of both implementations is present in `plot-GFMC-all.py`.

## Installation <a name="install">

The Python implementation of GFMC requires the `scipy`, `numpy`, `matplotlib`, and `h5py` packages which can be installed with `pip3 install scipy h5py numpy matplotlib`.

The CUDA implementation requires CUDA 8+ which can be installed from the standard Ubuntu apt repositories. The HDF5 library is also required, this can either be compiled from the source code present on The HDF Group website, or install from the standard Ubuntu apt repositories.

You may choose to use the precompiled source (`GFMLMC.o`) or compile the source code yourself. The first line of `GFMLMC.cu` is the compilation command if the HDF5 library was installed with `apt`, otherwise your linking paths and include paths will be different.

## User Guide <a name="begin">

Both the Python and CUDA implementations will produce similar output: a single file (MAT-files from Python and HDF5 files from CUDA) which contain the spacial distribution of the absorbed energy in the simulation.

#### How to use the Python implementation

In the top of the file, the number of photons, sample optical properties, beam properties, and bin configuration can be specified. To run the simulation, simply run `python3 GFMC.py`. The script will output a MAT-file with the spacial distribution of the absorbed energy in the sample for both the Gaussian focusing and traditional focusing cases.

#### How to use the CUDA implementation

The CUDA implementation is based on a multilayer Monte Carlo photon transport method, but can only be used with single layer samples at the moment. Otherwise, it is nearly identical to the Python implementation, other than the options are specified as arguments to the executible. To see the available options, use `./GFMLMC.o -h`. To run an example simulation and return aggregate results (instead of the spacial distribution of absorbed energy) use `./GFMLMC.o --example`.

#### How to plot the results

To plot the results from a Python or CUDA simulation, the script `plot-GFMC-all.py` is used. Simply run `python3 plot-GFMC-all.py <path/to/GFMC_output>` where `<path/to/GFMC_output>` should be the full path to either a MAT-file from the Python implementation or a HDF5 file from the CUDA implementation. This script will produce four graphs, saved both as `.png` and `.ps`. Two of these graphs are the spacial distribution of absorbed energy (one for Gaussian focusing and one for traditional focusing). The third graph is the power absorbed along the z-axis, and the final graph is the beam's second moment width as a function of z.

