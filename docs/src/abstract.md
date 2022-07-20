# Abstract Interface for NFFTs

The package `AbstractNFFTs` provides the abstract interface for NFFT implementations. Defining an abstract interface has the advantage that different implementations can be used and exchanging requires 
only small effort.

An overview about the current packages and their dependencies is shown in the following package tree:

![NFFT.jl package family structure](./assets/NFFTPackages.svg)

!!! note
    If you are not an expert user, you likely do not require different NFFT implementations and we therefore recommend to just use `NFFT.jl` and not worry about the abstract interface. 

## Implementations

Currently, there are four implementations of the `AbstractNFFTs` interface:
1. **NFFT.jl**: This is the reference implementation running und the CPU.
2. **CuNFFT.jl**: An implementation running on graphics hardware of Nvidia exploiting CUDA.jl
3. **NFFT3.jl**: In the `Wrapper` directory of `NFFT.jl` there is a wrapper around the `NFFT3.jl` package following the  `AbstractNFFTs` interface. `NFFT3.jl` is itself a wrapper around the high performance C library [NFFT3](http://www.nfft.org).
4. **FINUFFT.jl**: In the `Wrapper` directory of `NFFT.jl` there is a wrapper around the `FINUFFT.jl` package. `FINUFFT.jl` is itself a wrapper around the high performance C++ library [FINUFFT](https://finufft.readthedocs.io).

!!! note
    Right now one needs to install `NFFT.jl` and manually include the wrapper files. In the future we hope to integrate the wrappers in `NFFT3.jl` and `FINUFFT.jl` directly such that it is much more convenient to switch libraries.


## Interface

An NFFT implementation needs to define a new type that is a subtype of `AbstractNFFTPlan{T,D,R}`.
Here
* `T` is the real-valued element type of the nodes, i.e. a transform operating on `Complex{Float64}` values and `Float64` nodes uses the type `T=Float64`.
* `D` is the size of the input vector
* `R` is the size of the output vector. Usually this will be `R=1` unless a directional NFFT is implemented.

For instance the `CuNFFTPlan` is defined like this
```julia
mutable struct CuNFFTPlan{T,D} <: AbstractNFFTPlan{T,D,1} 
  ...
end
```

In addition to the plan, the following functions need to be implemented: 
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
plan_nfft(Q::Type, k::Matrix{T}, N::NTuple{D,Int}; kargs...) where {D}
```
where `Q` is the Array type, e.g. `Array`. The reason to require the array type is, that this allows for GPU implementations, which would use for instance `CuArray` here.

The package `AbstractNFFTs` provides a convenient constructor
```
plan_nfft(k::Matrix{T}, N::NTuple{D,Int}; kargs...) where {D}
```
defaulting to the `Array` type.

!!! note
    Different packages implementing `plan_nfft` will conflict if the same `Q` is implemented. In case of `NFFT.jl` and `CuNFFT.jl` there is no conflict since the array type is different.


## Derived Interface

Based on the core low-level interface that an `AbstractNFFTPlan` needs to provide, the package
`AbstractNFFT.jl` also provides high-level functions like `*`, `nfft`, and `nfft_adjoint`, which internally use the low-level interface. Thus, the implementation of high-level function is shared
among all `AbstractNFFT.jl` implementations.

