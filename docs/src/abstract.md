# Abstract Interface for NFFTs

The package `AbstractNFFTs` provides the abstract interface for NFFT implementations. Defining an abstract interface has the advantage that different implementation can exist and be exchanged with 
close to zero effort on the user side.

## Implementations

Right now there are three implementations:
1. **NFFT.jl**: This is the reference implementation running und the cpu.
2. **CuNFFT.jl**: An implementation running on graphics hardware of Nvidia exploiting CUDA.jl
3. **NFFT3.jl**: In the test directory of `NFFT.jl` there is a wrapper around the NFFT3.jl package following the  `AbstractNFFTs` interface. `NFFT3.jl` is itself a wrapper around the high performance C library [nfft3](http://www.nfft.org).

## Interface

An NFFT implementation needs to define a new type that is a subtype of `AbstractNFFTPlan{T,D,R}`.
Here
* `T` is the real-valued element type of the nodes, i.e. a transform the operates on `Complex{Float64}` values and has `Float64` nodes uses the type `T=Float64` here.
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
nodes!(p, x) -> p
```
All these functions are exported from `AbstractNFFTs` and we recommend to implement them by using the explicit `AbstractNFFTs.` prefix:

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
    nodes!(p, x)
```
Change nodes `x` in the plan `p` operation and return the plan. The implementation of this function is optional.

## Plan Interface

The constructor for an plan also has a defined interface. It should be implemented in this way:
```
function MyNFFTPlan(x::Matrix{T}, N::NTuple{D,Int}; kwargs...) where {T,D}
  ...
end
```
All parameters are put into keyword arguments that have to match as well. We describe the keyword arguments in more detail in the overview page.

Additionally, to the constructor an `AbstractNFFTPlan` implementation can provide the factory
```
plan_nfft(Q::Type, x::Matrix{T}, N::NTuple{D,Int}; kwargs...) where {D}
```
where `Q` is the Array type, e.g. `Array`. The reason to require the array type is, that this allows for GPU implementations, which would use for instance `CuArray` here.

The package `AbstractNFFTs` provides a convenient constructor
```
plan_nfft(x::Matrix{T}, N::NTuple{D,Int}; kwargs...) where {D}
```
defaulting to the `Array` type.

!!! note
    Different packages implementing `plan_nfft` will conflict if the same `Q` is implemented. In case of `NFFT.jl` and `CuNFFT.jl` there is no conflict since the array type is different.


## Derived Interface

The following derived functions are provided for all plans that inherit from `AbstractNFFTPlan`:

#### Non-preallocated NFFT

The following two functions allocate a fresh output vector an operate out of place
```julia
*(p, f) -> fHat
*(adjoint(p), fHat) -> f
```


...The NFFT can also be considered as a matrix vector multiplication. Julia provides the interface
```julia
  *(A, x) -> b
  mul!(b, A, x) -> b
```
for this. Both operations are implemented for any `AbstractNFFTPlan`. To obtain the adjoint on
needs to apply `adjoint(p)` to the plan `p` before multiplication.


#### Non-preallocated Plan

The following two functions perform an NFFT without a preallocated plan:
```julia
nfft(x, f) -> fHat
nfft_adjoint(x, N, fHat) -> f
```
Note that `N` needs only be specified for the adjoint. The direct NFFT can derive it from `f`.


