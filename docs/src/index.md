# NFFT.jl

*Julia package for the Non-equidistant Fast Fourier Transform*

## Introduction

This package provides a Julia implementation of the Non-equidistant Fast Fourier Transform (NFFT).
For a detailed introduction into the NFFT and its application please have a look at [www.nfft.org](http://www.nfft.org).

The NFFT is a fast implementation of the Non-equidistant Discrete Fourier Transform (NDFT) that is
basically a DFT with non-equidistant sampling nodes in either Fourier or time/space domain.
In contrast to the FFT, the NFFT is an approximative algorithm whereas the accuracy can be controlled
by two parameters: the window width `m` and the oversampling factor `σ`.

## Installation

Start julia and open the package mode by entering `]`. Then enter
```julia
add NFFT
```
This will install the packages `NFFT.jl` and all its dependencies. 
If you need support for `CUDA` you also need to install the package `CuNFFT.jl`

## Guide

* The documentation starts with the [Mathematical Background](@ref) that properly defines the NDFT, the NFFT and its directional variants. You might want to skip this part if you are familiar with the notation and concepts of the NFFT. 
* Then, an [Overview](@ref) about the usage of the NFFT functions is given in a tutorial style manner.  
* In the section about the [Abstract Interface for NFFTs](@ref) we outline how the package is divided into an interface package and two implementation packages. This part is useful if you plan to use different NFFT implementations, e.g. one for the CPU and another for the GPU and would like to switch.
* The section about [Tools](@ref) introduced some high-level functions that build upon the NFFT. For instance NFFT inversion is discussed in that section.
* Then an overview about [Accuracy and Performance](@ref) is given.
* [Implementation](@ref) outlines some implementation details.
* Finally, the documentation contains an [API](@ref) index.



## License / Terms of Usage

The source code of this project is licensed under the MIT license. This implies that
you are free to use, share, and adapt it. However, please give appropriate credit
by citing the project.

## Contact

If you have problems using the software, find bugs or have ideas for improvements please use
the [issue tracker](https://github.com/tknopp/NFFT.jl/issues). For general questions please use
the [discussions](https://github.com/tknopp/NFFT.jl/discussions) section on Github.

## Contributors

* [Tobias Knopp](https://www.tuhh.de/ibi/people/tobias-knopp-head-of-institute.html)
* [Robert Dahl Jacobsen](https://github.com/robertdj)
* [Mirco Grosser](https://github.com/migrosser)
* [Jakob Assländer](https://med.nyu.edu/faculty/jakob-asslaender)
* [Mosè Giordano](https://github.com/giordano)

A complete list of contributors can be found on the [Github page](https://github.com/tknopp/NFFT.jl/graphs/contributors).