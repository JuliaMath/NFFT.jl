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

Currently, the eltype of the arguments `f` and `fHat`
must be compatible that of the variable `x` used in the `NFFTPlan` call.
For example, if one wants to use `Float32` types to save memory,
then one can make the plan using something like this:

```
x = Float32.(LinRange(-0.5,0.5,64))
p = NFFTPlan(x, N)
```

The plan will then internally use `Float32` types.
Then the arguments `f` and `fHat` above should have eltype `Complex{Float32}`
or equivalently `ComplexF32`, otherwise there will be error messages.
