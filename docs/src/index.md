# NFFT.jl

*Julia package for the Non-equidistant Fast Fourier Transform*

## Introduction

This package provides a Julia implementation of the Non-equidistant Fast Fourier Transform (NFFT).
For a detailed introduction into the NFFT and its application please have a look at the [software paper](https://arxiv.org/pdf/2208.00049.pdf) on the `NFFT.jl`. Further resources are [nfft.org](http://www.nfft.org) and [finufft.readthedocs.io](https://finufft.readthedocs.io). You 

The NFFT is a fast implementation of the Non-equidistant Discrete Fourier Transform (NDFT) that is
basically a Discrete Fourier Transform (DFT) with non-equidistant sampling nodes in either Fourier or time/space domain.
In contrast to the Fast Fourier Transform (FFT), the NFFT is an approximative algorithm whereas the accuracy can be controlled
by two parameters: the window width parameter `m` and the oversampling factor `σ`.

The `NFFT.jl` project serves two different purposes:
* Provide a package `AbstractNFFTs.jl` that allows to use any NFFT Julia package such as [NFFT3.jl](https://github.com/NFFT/NFFT3.jl) or [FINUFFT.jl](https://github.com/ludvigak/FINUFFT.jl) using the same interface
* Provide a high-performance, multi-threaded reference implementation in pure Julia. This is realized in the Julia package `NFFT.jl`.
The term `NFFT.jl` thus may either mean the entire Github project consisting of several packages or the concrete reference implementation.

## Installation

Start julia and open the package mode by entering `]`. Then enter
```julia
add NFFT
```
This will install the packages `NFFT.jl` and all its dependencies. Most importantly it installs the abstract interface package `AbstractNFFTs.jl`, which `NFFT.jl` implements.

Additional NFFT related tools can be obtained by adding the package `NFFTTools.jl`. If you need support for `CUDA` you also need to install the package `CuNFFT.jl`.

In case you want to use an alternative NFFT implementation such as [NFFT3.jl](https://github.com/NFFT/NFFT3.jl) or [FINUFFT.jl](https://github.com/ludvigak/FINUFFT.jl) we provide wrapper types allowing to use them as `AbstractNFFTs` implementations. They can be used like this:

```julia
julia> using AbstractNFFTs
julia> include(joinpath(dirname(pathof(AbstractNFFTs)), "..", "..", "Wrappers", "FINUFFT.jl"))
julia> include(joinpath(dirname(pathof(AbstractNFFTs)), "..", "..", "Wrappers", "NFFT3.jl"))
```
This requires that you first `add` the package you want to use. 

## Guide

* The documentation starts with the [Mathematical Background](@ref) that properly defines the NDFT, the NFFT and its directional variants. You might want to skip this part if you are familiar with the notation and concepts of the NFFT. 
* Then, an [Overview](@ref) about the usage of the NFFT functions is given in a tutorial style manner.  
* Then, an overview about [Accuracy and Performance](@ref) is given.
* The section about [Tools](@ref) introduced some high-level functions that build upon the NFFT. For instance NFFT inversion is discussed in that section.
* In the section about the [Abstract Interface for NFFTs](@ref) we outline how the package is divided into an interface package and implementation packages. This part is useful if you plan to use different NFFT implementations, e.g. one for the CPU and another for the GPU and would like to switch.
* Finally, the documentation contains an [API](@ref) index.

## License / Terms of Usage

The source code of this project is licensed under the MIT license. This implies that
you are free to use, share, and adapt it. However, please give appropriate credit
by citing the project. You can do so by citing the publication

T. Knopp, M. Boberg and M. Grosser, NFFT.jl: Generic and Fast Julia Implementation of the Nonequidistant
 Fast Fourier Transform, 2022 [*arXiv:2208.00049*](http://arxiv.org/abs/2208.00049)

A BibTeX file `NFFT.bib` can be found in the root folder of the Github repository.

## Contact

If you have problems using the software, find bugs or have ideas for improvements please use
the [issue tracker](https://github.com/JuliaMath/NFFT.jl/issues). For general questions please use
the [discussions](https://github.com/JuliaMath/NFFT.jl/discussions) section on Github.

## Contributors

* [Tobias Knopp](https://www.tuhh.de/ibi/people/tobias-knopp-head-of-institute.html)
* [Robert Dahl Jacobsen](https://github.com/robertdj)
* [Mirco Grosser](https://github.com/migrosser)
* [Jakob Assländer](https://med.nyu.edu/faculty/jakob-asslaender)
* [Mosè Giordano](https://github.com/giordano)

A complete list of contributors can be found on the [Github page](https://github.com/JuliaMath/NFFT.jl/graphs/contributors).