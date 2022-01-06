# Overview

## Standard

Basic usage of NFFT.jl is shown in the following example for 1D:

```julia
using NFFT

M, N = 1024, 512
x = range(-0.4, stop=0.4, length=M)  # nodes at which the NFFT is evaluated
fHat = randn(ComplexF64,M)           # data to be transformed
p = plan_nfft(x, N)                  # create plan. m and σ are optional parameters
f = nfft_adjoint(p, fHat)            # calculate adjoint NFFT
g = nfft(p, f)                       # calculate forward NFFT
```

In 2D:

```julia
M, N = 1024, 16
x = rand(2, M) .- 0.5
fHat = randn(ComplexF64,M)
p = plan_nfft(x, (N,N))
f = nfft_adjoint(p, fHat)
g = nfft(p, f)
```

Currently, the eltype of the arguments `f` and `fHat`
must be compatible that of the variable `x` used in the `plan_nfft` call.
For example, if one wants to use `Float32` types to save memory,
then one can make the plan using something like this:

```
x = Float32.(LinRange(-0.5,0.5,64))
p = plan_nfft(x, N)
```

The plan will then internally use `Float32` types.
Then the arguments `f` and `fHat` above should have eltype `Complex{Float32}`
or equivalently `ComplexF32`, otherwise there will be error messages.

One can also perform NFFT computations directly without first creating a plan:
```
g = nfft(x, f)
f = nfft_adjoint(x, N, fHat)
```
These forms are more forgiving about the types of the input arguments.
The versions based on a plan are more optimized for repeated use with the same `x`


## Directional

There are special methods for computing 1D NFFT's for each 1D slice along a particular dimension of a higher dimensional array.

```julia
M = 11
y = rand(M) .- 0.5
N = (16,20)
P1 = plan_nfft(y, 1, N)
f = randn(ComplexF64,N)
fHat = nfft(P1, f)
```

Here `size(f) = (16,20)` and `size(fHat) = (11,20)` since we compute an NFFT along the first dimension.
To compute the NFFT along the second dimension

```julia
P2 = plan_nfft(y, 2, N)
fHat = nfft(P2, f)
```

Now `size(fHat) = (16,11)`.
