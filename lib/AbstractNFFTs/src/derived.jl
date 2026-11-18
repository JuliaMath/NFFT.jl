
##########################
# plan_* constructors
##########################

# Implementations define one method per family, e.g.
#   AbstractNFFTs.plan_nfft(b::MyBackend, k::AbstractMatrix, N::Dims; kwargs...)
# and resolve the generic keywords with `generic_options(k, N; kwargs...)`. The methods below
# select the active backend and bring nodes and sizes into that form.

for op in [:nfft, :nfct, :nfst, :rnfft]
planfunc = Symbol("plan_"*"$op")
@eval begin

$(planfunc)(k::AbstractVecOrMat, N::Union{Integer,Dims}; kwargs...) =
    $(planfunc)(active_backend(), k, N; kwargs...)

$(planfunc)(b::AbstractNFFTBackend, k::AbstractVector, N::Union{Integer,Dims}; kwargs...) =
    $(planfunc)(b, reshape(k, 1, :), N; kwargs...)

$(planfunc)(b::AbstractNFFTBackend, k::AbstractMatrix, N::Integer; kwargs...) =
    $(planfunc)(b, k, (Int(N),); kwargs...)

$(planfunc)(::Missing, args...; kwargs...) = no_backend_error()
end
end

"""
    plan_nfft([backend,] k, N; reltol=1e-8, dims=:, eltype, storage, nthreads, kwargs...)

Plan an NFFT for the nodes `k` (a d × J matrix, or a vector for d = 1, entries in [-1/2, 1/2))
and an equidistant array of size `N` (an integer or a tuple; with `dims`, the size of the whole
array including batch dimensions). Without `backend` the active backend is used.

Generic options, accepted by every implementation:

* `reltol`: requested relative error (ℓ² norm); a value the implementation cannot reach throws an
  `ArgumentError`
* `dims`: dimensions to transform (`:`, an integer or a contiguous range); the other dimensions
  are batch dimensions and pass through unchanged
* `eltype`: real element type of the data, default `float(eltype(k))`
* `storage`: array type of data and workspace, default the array type of `k`
* `nthreads`: number of CPU threads, default `Threads.nthreads()`

Further keyword arguments are specific to the implementation; an implementation throws an
`ArgumentError` for options it does not know.
"""
plan_nfft

"""
    plan_nfct([backend,] k, N; kwargs...)

Plan a non-equidistant fast cosine transform (NFCT) for nodes `k` in [0, 1/2] and a real array of
size `N`. Keyword arguments as for [`plan_nfft`](@ref).
"""
plan_nfct

"""
    plan_nfst([backend,] k, N; kwargs...)

Plan a non-equidistant fast sine transform (NFST) for nodes `k` in [0, 1/2] and a real array of
size `N`. Keyword arguments as for [`plan_nfft`](@ref).
"""
plan_nfst

"""
    plan_rnfft([backend,] k, N; kwargs...)

Plan an NFFT with real values at the nodes `k` (in [-1/2, 1/2)) for an array of logical size `N`
(the NFFT counterpart of FFTW's `brfft`/`rfft`). The input of the direct transform is the half
`h` of the coefficients with `n₁ = 0 … N₁÷2`, an array of size `(N₁÷2+1, N₂, …)` whose other
dimensions are indexed as in [`plan_nfft`](@ref); the direct transform is

    f̂_j = Re Σ_n h_n exp(-2πi n·k_j),

the adjoint maps real values `y` at the nodes to `h_n = Σ_j y_j exp(2πi n·k_j)`. The two are
adjoint with respect to the real inner product. With `dims`, the first of the transformed
dimensions is the halved one. Keyword arguments as for [`plan_nfft`](@ref).
"""
plan_rnfft

"""
    plan_nnfft([backend,] k, y[, N]; kwargs...)

Plan an NNFFT with non-equidistant nodes `k` (`D × J`) in the frequency domain and `y` (`D × M`)
in the spatial domain. The input is a vector of length `M`, or an array of size `N` whose
dimension `dims` (an integer) holds the `M` sources; the output has the `J` targets in that
dimension, and the other dimensions are batch dimensions. Keyword arguments as for
[`plan_nfft`](@ref).
"""
plan_nnfft

## NNFFT constructor
plan_nnfft(k::AbstractVecOrMat, y::AbstractVecOrMat, N::Dims...; kwargs...) = plan_nnfft(active_backend(), k, y, N...; kwargs...)
plan_nnfft(b::AbstractNFFTBackend, k::AbstractVector, y::AbstractVector, N::Dims...; kwargs...) =
    plan_nnfft(b, reshape(k, 1, :), reshape(y, 1, :), N...; kwargs...)
plan_nnfft(::Missing, args...; kwargs...) = no_backend_error()


###############################################
# Allocating trafo functions with plan creation
###############################################

"""
    nfft(k, f; kwargs...)
    nfft(backend, k, f; kwargs...)

NFFT of the array `f` at the nodes `k` (`D × J`, a vector for `D = 1`), with a plan made for this
call: `plan_nfft(backend, k, size(f); kwargs...) * f`. For repeated transforms at the same nodes,
make the plan once.

Without `backend` the active backend is used. A package activates itself with
`BackendModule.activate!()`, and a scoped value overrides the active backend within a scope:

```julia
julia> NFFT.activate!()

julia> nfft(k, f)                                   # uses NFFT

julia> with(nfft_backend => NonuniformFFTs.backend()) do
          nfft(k, f)                                # uses NonuniformFFTs
       end
```
"""
nfft

"""
    nfft_adjoint(k, N, fHat; kwargs...)
    nfft_adjoint(backend, k, N, fHat; kwargs...)

Adjoint NFFT of the values `fHat` at the nodes `k`, an array of size `N`:
`adjoint(plan_nfft(backend, k, N; kwargs...)) * fHat`. Backend as for [`nfft`](@ref).
"""
nfft_adjoint

"""
    nfct(k, f; kwargs...)
    nfct(backend, k, f; kwargs...)

NFCT of the real array `f` at the nodes `k` in ``[0, 1/2]^D``: `plan_nfct(backend, k, size(f); kwargs...) * f`.
Backend as for [`nfft`](@ref).
"""
nfct

"""
    nfct_transpose(k, N, fHat; kwargs...)
    nfct_transpose(backend, k, N, fHat; kwargs...)

Transposed NFCT of the values `fHat` at the nodes `k`, an array of size `N`:
`transpose(plan_nfct(backend, k, N; kwargs...)) * fHat`. Backend as for [`nfft`](@ref).
"""
nfct_transpose

"""
    nfst(k, f; kwargs...)
    nfst(backend, k, f; kwargs...)

NFST of the real array `f` at the nodes `k` in ``[0, 1/2]^D``: `plan_nfst(backend, k, size(f); kwargs...) * f`.
Backend as for [`nfft`](@ref).
"""
nfst

"""
    nfst_transpose(k, N, fHat; kwargs...)
    nfst_transpose(backend, k, N, fHat; kwargs...)

Transposed NFST of the values `fHat` at the nodes `k`, an array of size `N`:
`transpose(plan_nfst(backend, k, N; kwargs...)) * fHat`. Backend as for [`nfft`](@ref).
"""
nfst_transpose

for (op,trans) in zip([:nfft, :nfct, :nfst],
                      [:adjoint, :transpose, :transpose])
planfunc = Symbol("plan_$(op)")
tfunc = Symbol("$(op)_$(trans)")
@eval begin 

$(op)(k, f::AbstractArray; kargs...) = $(op)(active_backend(), k, f::AbstractArray; kargs...) 
function $(op)(b::AbstractNFFTBackend, k, f::AbstractArray; kargs...) 
  p = $(planfunc)(b, k, size(f); kargs...)
  return p * f
end
$(op)(::Missing, k, f::AbstractArray; kargs...) = no_backend_error()


$(tfunc)(k, N, fHat;  kargs...) = $(tfunc)(active_backend(), k, N, fHat;  kargs...)
function $(tfunc)(b::AbstractNFFTBackend, k, N, fHat;  kargs...) 
  p = $(planfunc)(b, k, N; kargs...)
  return $(trans)(p) * fHat
end
$(tfunc)(::Missing, k, N, fHat;  kargs...) = no_backend_error()


end
end

############################
# Allocating trafo functions
############################

"""
    *(p, f) -> fHat
    *(adjoint(p), fHat) -> f
    *(transpose(p), fHat) -> f

Apply a plan, its adjoint (NFFT, NNFFT, RNFFT) or its transpose (NFCT, NFST) and return a new
array. `f` has size `size_in(p)` and `fHat` size `size_out(p)`; batch dimensions (`dims`) are part
of both. `mul!` writes into a given array instead.
"""
function Base.:*(p::AbstractComplexFTPlan{T}, f::AbstractArray{Complex{U},D}) where {T,U,D}
  fHat = similar(f, Complex{T}, size_out(p))
  mul!(fHat, p, f)
  return fHat
end

function Base.:*(p::AbstractRNFFTPlan{T}, h::AbstractArray) where {T}
  fHat = similar(h, T, size_out(p))
  mul!(fHat, p, h)
  return fHat
end
function Base.:*(p::Adjoint{Complex{T},<:AbstractRNFFTPlan{T}}, fHat::AbstractArray) where {T}
  h = similar(fHat, Complex{T}, size_out(p))
  mul!(h, p, fHat)
  return h
end
function Base.:*(p::Adjoint{Complex{T},<:AbstractRNFFTPlan{T}}, fHat::AbstractVector) where {T}
  h = similar(fHat, Complex{T}, size_out(p))
  mul!(h, p, fHat)
  return h
end
function Base.:*(p::Adjoint{Complex{T},<:AbstractRNFFTPlan{T}}, fHat::AbstractMatrix) where {T}
  h = similar(fHat, Complex{T}, size_out(p))
  mul!(h, p, fHat)
  return h
end

function Base.:*(p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T}}, fHat::AbstractArray{Complex{U},D}) where {T,U,D}
  f = similar(fHat, Complex{T}, size_out(p))
  mul!(f, p, fHat)
  return f
end

# The following two methods are redundant but need to be defined because of a method ambiguity with Julia Base
function Base.:*(p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T}}, fHat::AbstractVector{Complex{U}}) where {T,U}
  f = similar(fHat, Complex{T}, size_out(p))
  mul!(f, p, fHat)
  return f
end
function Base.:*(p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T}}, fHat::AbstractArray{Complex{U},2}) where {T,U}
  f = similar(fHat, Complex{T}, size_out(p))
  mul!(f, p, fHat)
  return f
end

function Base.:*(p::AbstractRealFTPlan{T}, f::AbstractArray{U,D}) where {T,U,D}
  fHat = similar(f, T, size_out(p))
  mul!(fHat, p, f)
  return fHat
end

function Base.:*(p::Transpose{T,<:AbstractRealFTPlan{T}}, fHat::AbstractArray{U,D}) where {T,U,D}
  f = similar(fHat, T, size_out(p))
  mul!(f, p, fHat)
  return f
end

# The following two methods are redundant but need to be defined because of a method ambiguity with Julia Base
function Base.:*(p::Transpose{T,<:AbstractRealFTPlan{T}}, fHat::AbstractVector{U}) where {T,U}
  f = similar(fHat, T, size_out(p))
  mul!(f, p, fHat)
  return f
end
function Base.:*(p::Transpose{T,<:AbstractRealFTPlan{T}}, fHat::AbstractArray{U,2}) where {T,U}
  f = similar(fHat, T, size_out(p))
  mul!(f, p, fHat)
  return f
end
