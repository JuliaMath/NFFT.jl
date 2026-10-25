# Macro to define interface function. Taking from Graphics.jl
macro mustimplement(sig)
  fname = sig.args[1]
  arg1 = sig.args[2]
  if isa(arg1,Expr)
      arg1 = arg1.args[1]
  end
  :($(esc(sig)) = error(typeof($(esc(arg1))),
                        " must implement ", $(Expr(:quote,fname))))
end

"""
    DEFAULT_RELTOL

Relative error requested from a plan when `reltol` is not given.
"""
const DEFAULT_RELTOL = 1e-8

"""
    GenericOptions

The options every implementation of the interface accepts, resolved from the keyword arguments
of `plan_nfft` and friends by [`generic_options`](@ref):

* `reltol`: requested relative error (ℓ² norm over random data)
* `dims`: dimensions of the transform (a contiguous range); all other dimensions are batch
  dimensions that pass through unchanged
* `eltype`: real element type of the data (`Float32`, `Float64`, …)
* `storage`: array type of the data and the workspace (`Array`, a GPU array type, …)
* `nthreads`: number of CPU threads
"""
struct GenericOptions{T}
  reltol::Float64
  dims::UnitRange{Int}
  eltype::Type{T}
  storage::Type
  nthreads::Int
end

"""
    generic_options(k, N; reltol, dims, eltype, storage, nthreads) -> (GenericOptions, rest)

Resolve the generic keyword arguments of a plan constructor for nodes `k` and array size `N`,
and return the remaining (implementation-specific) keyword arguments untouched.
"""
function generic_options(k::AbstractMatrix, N::Dims{D}; reltol::Real=DEFAULT_RELTOL,
                         dims::Union{Colon,Integer,AbstractUnitRange}=Colon(),
                         eltype::Union{Nothing,Type{<:Real}}=nothing, storage::Union{Nothing,Type}=nothing,
                         nthreads::Integer=Threads.nthreads(), kwargs...) where {D}
  d = dims isa Colon ? (1:D) : dims isa Integer ? (dims:dims) : UnitRange{Int}(dims)
  (first(d) >= 1 && last(d) <= D && !isempty(d)) ||
    throw(ArgumentError("dims = $dims is not a contiguous range within 1:$D"))
  size(k, 1) == length(d) ||
    throw(ArgumentError("nodes have $(size(k, 1)) dimensions, the transform has $(length(d))"))
  reltol > 0 || throw(ArgumentError("reltol must be positive"))
  nthreads >= 1 || throw(ArgumentError("nthreads must be at least 1"))
  T = eltype === nothing ? float(Base.eltype(k)) : eltype
  S = storage === nothing ? storage_type(k) : storage
  GenericOptions{T}(Float64(reltol), d, T, S, Int(nthreads)), kwargs
end

# Array type at the root of views and wrappers (SubArray, Adjoint, ReshapedArray, …).
storage_type(x::AbstractArray) = parent(x) === x ? Base.typename(typeof(x)).wrapper : storage_type(parent(x))

"""
    check_options(backend, kwargs, supported)

Throw an `ArgumentError` if `kwargs` contains a keyword that is not in `supported`. Every
implementation must reject options it does not understand instead of ignoring them.
"""
function check_options(backend, kwargs, supported)
  unknown = setdiff(keys(kwargs), supported)
  isempty(unknown) || throw(ArgumentError(
    "$(nameof(typeof(backend))) does not support the option(s) $(join(unknown, ", ")). " *
    "Generic options: reltol, dims, eltype, storage, nthreads. " *
    "Options of this implementation: $(join(supported, ", "))."))
  nothing
end
