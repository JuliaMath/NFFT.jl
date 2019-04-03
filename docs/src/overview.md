# Overview

Basic usage of NFFT.jl is shown in the following example for 1D:

```julia
using NFFT

M, N = 1024, 512
x = range(-0.4, stop=0.4, length=M)  # nodes at which the NFFT is evaluated
fHat = randn(ComplexF64,M)           # data to be transformed
p = NFFTPlan(x, N)                   # create plan. m and sigma are optional parameters
f = nfft_adjoint(p, fHat)            # calculate adjoint NFFT
g = nfft(p, f)                       # calculate forward NFFT
```

In 2D:

```julia
M, N = 1024, 16
x = rand(2, M) .- 0.5
fHat = randn(ComplexF64,M)
p = NFFTPlan(x, (N,N))
f = nfft_adjoint(p, fHat)
g = nfft(p, f)
```
