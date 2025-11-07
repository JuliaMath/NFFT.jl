# Abstract Interface for NFFTs

The package `AbstractNFFTs` provides the abstract interface for NFFT implementations. Defining an abstract interface has the advantage that different implementations can be used and exchanging requires 
only small effort.

An overview about the current packages and their dependencies is shown in the following package tree:

![NFFT.jl package family structure](./assets/NFFTPackages.svg)

!!! note
    If you are not an expert user, you likely do not require different NFFT implementations and we therefore recommend to just use `NFFT.jl` and not worry about the abstract interface. 

## Implementations

Currently, there are four implementations of the `AbstractNFFTs` interface:
1. **NFFT.jl**: This is the reference implementation running on the CPU and with configurations on the GPU.
2. **NFFT3.jl**: In the `Wrapper` directory of `NFFT.jl` there is a wrapper around the `NFFT3.jl` package following the  `AbstractNFFTs` interface. `NFFT3.jl` is itself a wrapper around the high performance C library [NFFT3](http://www.nfft.org).
3. **FINUFFT.jl**: In the `Wrapper` directory of `NFFT.jl` there is a wrapper around the `FINUFFT.jl` package. `FINUFFT.jl` is itself a wrapper around the high performance C++ library [FINUFFT](https://finufft.readthedocs.io).
4. **NonuniformFFTs.jl**: Pure Julia package written with generic and fast GPU kernels written with KernelAbstractions.jl.

!!! note
    Right now one needs to install `NFFT.jl` and manually include the wrapper files. In the future we hope to integrate the wrappers in `NFFT3.jl` and `FINUFFT.jl` directly such that it is much more convenient to switch libraries.

It's possible to change between different implementation backends. Each backend has to implement a backend type, which by convention can be accessed via for example `NFFT.backend()`. There are several ways to activate a backend:
```julia
# Actively setting a backend:
AbstractNFFTs.set_active_backend!(NFFT.backend())
# Activating a backend:
NFFT.activate!()
# and creating a new dynamic scope which uses a different backend:
with(nfft_backend => NonuniformFFTs.backend()) do
    # Uses NonuniformFFTs as implementation backend
end
# It's also possible to directly pass backends to functions:
nfft(NonuniformFFTs.backend(), ...)
```

## Interface

An NFFT implementation needs to define a new type that is a subtype of `AbstractNFFTPlan{T,D,R}`.
Here
* `T` is the real-valued element type of the nodes, i.e. a transform operating on `Complex{Float64}` values and `Float64` nodes uses the type `T=Float64`.
* `D` is the size of the input vector
* `R` is the size of the output vector. Usually this will be `R=1` unless a directional NFFT is implemented.

For instance the `NFFTPlan` is defined like this
```julia
mutable struct NFFTPlan{T,D,R} <: AbstractNFFTPlan{T,D,R} 
  ...
end
```

Furthermore, a package needs to implement its own backend type to dispatch on
```julia
struct MyBackend <: AbstractNFFTBackend
```
and it should allow a user to activate the package, which by convention can be done with (unexported) functions:
```julia
activate!() = AbstractNFFTs.set_active_backend!(MyBackend())
backend() = MyBackend()
```

In addition to the plan and backend, the following functions need to be implemented: 
```julia
size_out(p)
size_out(p)
mul!(fHat, p, f) -> fHat
mul!(f, p::Adjoint{Complex{T},<:AbstractNFFTPlan{T}}, fHat) -> f
nodes!(p, k) -> p
```
All these functions are exported from `AbstractNFFTs` and we recommend to implement them using the explicit `AbstractNFFTs.` prefix:

```
function AbstractNFFTs.size_out(p:MyNFFTPlan)
 ...
end
```

We next outline all of the aforementioned functions and describe their behavior:

```julia
    size_in(p)
```
Size of the input array for an NFFT operation. The returned tuple has `D` entries. 
Note that this will be the output array for an adjoint NFFT.

```julia
    size_out(p)
```
Size of the output array for an NFFT operation. The returned tuple has `R` entries. 
Note that this will be the input array for an adjoint NFFT.

```julia
    mul!(fHat, p, f) -> fHat
```

Inplace NFFT transforming the `D` dimensional array `f` to the `R` dimensional array `fHat`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
Both `f` and `fHat` must be complex arrays of element type `Complex{T}`.

```julia
    mul!(f, p::Adjoint{Complex{T},<:AbstractNFFTPlan{T}}, fHat) -> f
```
Inplace adjoint NFFT transforming the `R` dimensional array `fHat` to the `D` dimensional array `f`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
Both `f` and `fHat` must be complex arrays of element type `Complex{T}`.

```julia
    nodes!(p, k)
```
Exchange the nodes `k` in the plan `p` and return the plan. The implementation of this function is optional.

## Plan Interface

The constructor for a plan also has a defined interface. It should be implemented in this way:
```
function MyNFFTPlan(k::Matrix{T}, N::NTuple{D,Int}; kwargs...) where {T,D}
  ...
end
```
All parameters are put into keyword arguments that have to match as well. We describe the keyword arguments in more detail in the overview page. Using the same plan interface allows to load several NFFT libraries simultaneously and exchange the constructor dynamically by storing the constructor in a function object. This is how the unit tests of `NFFT.jl` run.

Additionally, to the type-specific constructor one can provide the factory
```
plan_nfft(b::MyBackend, Q::Type, k::Matrix{T}, N::NTuple{D,Int}; kargs...) where {D}
```
where `Q` is the Array type, e.g. `Array`. The reason to require the array type is, that this allows for GPU implementations, which would use for instance `CuArray` here.

The package `AbstractNFFTs` provides a convenient constructor
```
plan_nfft(b::MyBackend, k::Matrix{T}, N::NTuple{D,Int}; kargs...) where {D}
```
defaulting to the `Array` type.

## Derived Interface

Based on the core low-level interface that an `AbstractNFFTPlan` needs to provide, the package
`AbstractNFFT.jl` also provides high-level functions like `*`, `nfft`, and `nfft_adjoint`, which internally use the low-level interface. Thus, the implementation of high-level function is shared
among all `AbstractNFFT.jl` implementations.

