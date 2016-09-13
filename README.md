NFFT.jl
=======

[![Build Status](https://travis-ci.org/tknopp/NFFT.jl.svg?branch=master)](https://travis-ci.org/tknopp/NFFT.jl)

This package provides a Julia implementation of the Non-equidistant Fast Fourier Transform (NFFT).
This algorithm is also referred as Gridding in the literature (e.g. in MRI literature) 
For a detailed introduction into the NFFT and its application please have a look at www.nfft.org.
The NFFT is a fast implementation of the Non-equidistant Discrete Fourier Transform (NDFT) that is
basically a DFT with non-equidistant sampling nodes in either Fourier or time/space domain. In contrast
to the FFT, the NFFT is an approximative algorithm whereas the accuracy can be controlled by two parameters:
the window width `m` and the oversampling factor `sigma`.

## Installation

In Julia, run


```julia
Pkg.add("NFFT")
```


## Basic usage

Basic usage of NFFT.jl is shown in the following example for 1D:

```julia
using NFFT 

M, N = 1024, 512
x = linspace(-0.4, 0.4, M)      # nodes at which the NFFT is evaluated
fHat = randn(M) + randn(M)*im   # data to be transformed
p = NFFTPlan(x, N)              # create plan. m and sigma are optional parameters
f = nfft_adjoint(p, fHat)       # calculate adjoint NFFT
g = nfft(p, f)                  # calculate forward NFFT
```

In 2D:

```julia
M, N = 1024, 16
x = rand(2, M) - 0.5
fHat = randn(M) + randn(M)*im
p = NFFTPlan(x, (N,N))
f = nfft_adjoint(p, fHat)
g = nfft(p, f)
```


### Directional NFFT

There are special methods for computing 1D NFFT's for each 1D slice along a particular dimension of a higher dimensional array.

```julia
M = 11
y = rand(M) - 0.5
N = (16,20)
P1 = NFFTPlan(y, 1, N)
f = randn(N) + randn(N)*im
fHat = nfft(P1, f)
```

Here `size(f) = (16,20)` and `size(fHat) = (11,20)` since we compute and NFFT along the first dimension.
To compute the NFFT along the second dimension 

```julia
P2 = NFFTPlan(y, 2, N)
fHat = nfft(P2, f)
```

Now `size(fHat) = (16,11)`.

