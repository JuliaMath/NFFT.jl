
##########################
# plan_* constructors
##########################


for op in [:nfft, :nfct, :nfst]
planfunc = Symbol("plan_"*"$op")
@eval begin 

# The following automatically call the plan_* version for type Array

$(planfunc)(b::AbstractNFFTBackend, k::AbstractArray, N::Union{Integer,NTuple{D,Int}}, args...; kargs...) where {D} =
    $(planfunc)(b, Array, k, N, args...; kargs...)

$(planfunc)(b::AbstractNFFTBackend, k::AbstractArray, y::AbstractArray, args...; kargs...) =
    $(planfunc)(b, Array, k, y, args...; kargs...)

$(planfunc)(k::AbstractArray, args...; kargs...) = $(planfunc)(active_backend(), k, args...; kargs...)

# The follow convert 1D parameters into the format required by the plan

$(planfunc)(b::AbstractNFFTBackend, Q::Type, k::AbstractVector, N::Integer, rest...; kwargs...)  =
    $(planfunc)(b, Q, collect(reshape(k,1,length(k))), (N,), rest...; kwargs...)

$(planfunc)(b::AbstractNFFTBackend, Q::Type, k::AbstractVector, N::NTuple{D,Int}, rest...; kwargs...) where {D} =
    $(planfunc)(b, Q, collect(reshape(k,1,length(k))), N, rest...; kwargs...) 

$(planfunc)(b::AbstractNFFTBackend, Q::Type, k::AbstractMatrix, N::NTuple{D,Int}, rest...; kwargs...) where {D}  =
    $(planfunc)(b, Q, collect(k), N, rest...; kwargs...)

$(planfunc)(Q::Type, args...; kwargs...) = $(planfunc)(active_backend(), Q, args...; kwargs...)

$(planfunc)(::Missing, args...; kwargs...) = no_backend_error()
end
end

## NNFFT constructor
plan_nnfft(Q::Type, args...; kwargs...) = plan_nnfft(active_backend(), Q, args...; kwargs...)
plan_nnfft(b::AbstractNFFTBackend, Q::Type, k::AbstractVector, y::AbstractVector, rest...; kwargs...)  =
    plan_nnfft(b, Q, collect(reshape(k,1,length(k))), collect(reshape(y,1,length(k))), rest...; kwargs...)
plan_nnfft(::Missing, args...; kwargs...) = no_backend_error()


###############################################
# Allocating trafo functions with plan creation
###############################################

"""
    nfft(k, f, rest...; kwargs...)
    nfft(backend, k, f, rest...; kwargs...)

calculates the nfft of the array `f` for the nodes contained in the matrix `k`
The output is a vector of length M=`size(nodes,2)`.

Uses the active AbstractNFFTs `backend` if no `backend` argument is provided. Backends can be activated with `BackendModule.activate!()`.
Backends can also be set with a scoped value overriding the current active backend within a scope:

```julia
julia> NFFT.activate!()

julia> nfft(k, f, rest...; kwargs...) # uses NFFT

julia> with(nfft_backend => NonuniformFFTs.backend()) do
          nfft(k, f, rest...; kwargs...) # uses NonuniformFFTs
       end
```
"""
nfft
"""
    nfft_adjoint(k, N, fHat, rest...; kwargs...)
    nfft_adjoint(backend, k, N, fHat, rest...; kwargs...)

calculates the adjoint nfft of the vector `fHat` for the nodes contained in the matrix `k`.
The output is an array of size `N`.

Uses the active AbstractNFFTs `backend` if no `backend` argument is provided. Backends can be activated with `BackendModule.activate!()`.
Backends can also be set with a scoped value overriding the current active backend within a scope:

```julia
julia> NFFT.activate!()

julia> nfft_adjoint(k, N, fHat, rest...; kwargs...) # uses NFFT

julia> with(nfft_backend => NonuniformFFTs.backend()) do
          nfft_adjoint(k, N, fHat, rest...; kwargs...) # uses NonuniformFFTs
       end
```
"""
nfft_adjoint
"""
    nfft_transpose(k, N, fHat, rest...; kwargs...)
    nfft_transpose(backend, k, N, fHat, rest...; kwargs...)

calculates the transpose nfft of the vector `fHat` for the nodes contained in the matrix `k`.
The output is an array of size `N`.

Uses the active AbstractNFFTs `backend` if no `backend` argument is provided. Backends can be activated with `BackendModule.activate!()`.
Backends can also be set with a scoped value overriding the current active backend within a scope:

```julia
julia> NFFT.activate!()

julia> nfft_transpose(k, N, fHat, rest...; kwargs...) # uses NFFT

julia> with(nfft_backend => NonuniformFFTs.backend()) do
          nfft_transpose(k, N, fHat, rest...; kwargs...) # uses NonuniformFFTs
       end
```
"""
nfft_transpose

"""
    nfct(k, f, rest...; kwargs...)
    nfct(backend, k, f, rest...; kwargs...)

calculates the nfct of the array `f` for the nodes contained in the matrix `k`
The output is a vector of length M=`size(nodes,2)`.

Uses the active AbstractNFFTs `backend` if no `backend` argument is provided. Backends can be activated with `BackendModule.activate!()`.
"""
nfct
"""
    nfct_adjoint(k, N, fHat, rest...; kwargs...)
    nfct_adjoint(backend, k, N, fHat, rest...; kwargs...)

calculates the adjoint nfct of the vector `fHat` for the nodes contained in the matrix `k`.
The output is an array of size `N`.

Uses the active AbstractNFFTs `backend` if no `backend` argument is provided. Backends can be activated with `BackendModule.activate!()`.
"""
nfct_adjoint
"""
    nfct_transpose(k, N, fHat, rest...; kwargs...)
    nfct_transpose(backend, k, N, fHat, rest...; kwargs...)

calculates the transpose nfct of the vector `fHat` for the nodes contained in the matrix `k`.
The output is an array of size `N`.

Uses the active AbstractNFFTs `backend` if no `backend` argument is provided. Backends can be activated with `BackendModule.activate!()`.
"""
nfct_transpose

"""
    nfst(k, f, rest...; kwargs...)
    nfst(backend, k, f, rest...; kwargs...)

calculates the nfst of the array `f` for the nodes contained in the matrix `k`
The output is a vector of length M=`size(nodes,2)`.

Uses the active AbstractNFFTs `backend` if no `backend` argument is provided. Backends can be activated with `BackendModule.activate!()`.
"""
nfst
"""
    nfst_adjoint(k, N, fHat, rest...; kwargs...)
    nfst_adjoint(backend, k, N, fHat, rest...; kwargs...)

calculates the adjoint nfst of the vector `fHat` for the nodes contained in the matrix `k`.
The output is an array of size `N`.

Uses the active AbstractNFFTs `backend` if no `backend` argument is provided. Backends can be activated with `BackendModule.activate!()`.
"""
nfst_adjoint
"""
    nfst_transpose(k, N, fHat, rest...; kwargs...)
    nfst_transpose(backend, k, N, fHat, rest...; kwargs...)

calculates the transpose nfst of the vector `fHat` for the nodes contained in the matrix `k`.
The output is an array of size `N`.

Uses the active AbstractNFFTs `backend` if no `backend` argument is provided. Backends can be activated with `BackendModule.activate!()`.
"""
nfst_transpose

for (op,trans) in zip([:nfft, :nfct, :nfst],
                      [:adjoint, :transpose, :transpose])
planfunc = Symbol("plan_$(op)")
tfunc = Symbol("$(op)_$(trans)")
@eval begin 

$(op)(k, f::AbstractArray; kargs...) = $(op)(active_backend(), k, f::AbstractArray; kargs...) 
function $(op)(b::AbstractNFFTBackend, k, f::AbstractArray; kargs...) 
  p = $(planfunc)(k, size(f); kargs... )
  return p * f
end
$(op)(::Missing, k, f::AbstractArray; kargs...) = no_backend_error()


$(tfunc)(k, N, fHat;  kargs...) = $(tfunc)(active_backend(), k, N, fHat;  kargs...)
function $(tfunc)(b::AbstractNFFTBackend, k, N, fHat;  kargs...) 
  p = $(planfunc)(k, N;  kargs...)
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

For a **non**-directional `D` dimensional plan `p` this calculates the NFFT/NNFFT of a `D` dimensional array `f` of size `N`.
`fHat` is a vector of length `M`.
(`M` and `N` are defined in the plan creation)

For a **directional** `D` dimensional plan `p` both `f` and `fHat` are `D`
dimensional arrays, and the dimension specified in the plan creation is
affected.
"""
function Base.:*(p::AbstractComplexFTPlan{T}, f::AbstractArray{Complex{U},D}; kargs...) where {T,U,D}
  fHat = similar(f, Complex{T}, size_out(p))
  mul!(fHat, p, f; kargs...)
  return fHat
end

"""
        *(p::Adjoint{T,<:AbstractFTPlan{T}}, fHat) -> f

For a **non**-directional `D` dimensional plan `p` this calculates the adjoint NFFT/NNFFT of a length `M` vector `fHat`
`f` is a `D` dimensional array of size `N`.
(`M` and `N` are defined in the plan creation)

For a **directional** `D` dimensional plan `p` both `f` and `fHat` are `D`
dimensional arrays, and the dimension specified in the plan creation is
affected.
"""

function Base.:*(p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T}}, fHat::AbstractArray{Complex{U},D}; kargs...) where {T,U,D}
  f = similar(fHat, Complex{T}, size_out(p))
  mul!(f, p, fHat; kargs...)
  return f
end

# The following two methods are redundant but need to be defined because of a method ambiguity with Julia Base
function Base.:*(p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T}}, fHat::AbstractVector{Complex{U}}; kargs...) where {T,U}
  f = similar(fHat, Complex{T}, size_out(p))
  mul!(f, p, fHat; kargs...)
  return f
end
function Base.:*(p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T}}, fHat::AbstractArray{Complex{U},2}; kargs...) where {T,U}
  f = similar(fHat, Complex{T}, size_out(p))
  mul!(f, p, fHat; kargs...)
  return f
end



"""
        *(p, f) -> fHat

For a **non**-directional `D` dimensional plan `p` this calculates the NFCT/NFST of a `D` dimensional array `f` of size `N`.
`fHat` is a vector of length `M`.
(`M` and `N` are defined in the plan creation)

For a **directional** `D` dimensional plan `p` both `f` and `fHat` are `D`
dimensional arrays, and the dimension specified in the plan creation is
affected.
"""
function Base.:*(p::AbstractRealFTPlan{T}, f::AbstractArray{U,D}; kargs...) where {T,U,D}
  fHat = similar(f, T, size_out(p))
  mul!(fHat, p, f; kargs...)
  return fHat
end

"""
        *(p::Transpose{T,AbstractRealFTPlan{T}}, fHat) -> f

For a **non**-directional `D` dimensional plan `p` this calculates the adjoint NFCT/NFST of a length `M` vector `fHat`
`f` is a `D` dimensional array of size `N`.
(`M` and `N` are defined in the plan creation)

For a **directional** `D` dimensional plan `p` both `f` and `fHat` are `D`
dimensional arrays, and the dimension specified in the plan creation is
affected.
"""

function Base.:*(p::Transpose{T,<:AbstractRealFTPlan{T}}, fHat::AbstractArray{U,D}; kargs...) where {T,U,D}
  f = similar(fHat, T, size_out(p))
  mul!(f, p, fHat; kargs...)
  return f
end

# The following two methods are redundant but need to be defined because of a method ambiguity with Julia Base
function Base.:*(p::Transpose{T,<:AbstractRealFTPlan{T}}, fHat::AbstractVector{U}; kargs...) where {T,U}
  f = similar(fHat, T, size_out(p))
  mul!(f, p, fHat; kargs...)
  return f
end
function Base.:*(p::Transpose{T,<:AbstractRealFTPlan{T}}, fHat::AbstractArray{U,2}; kargs...) where {T,U}
  f = similar(fHat, T, size_out(p))
  mul!(f, p, fHat; kargs...)
  return f
end
