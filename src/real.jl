# NFCT and NFST with the definitions of NFFT3 (Fenn and Potts):
#   NFCT: f̂_j = Σ_{n=0}^{N-1} f_n Π_d cos(2π n_d k_dj),  NFST: f̂_j = Σ_{n=1}^{N} f_n Π_d sin(2π n_d k_dj),
# nodes in [0, 1/2]^D. Both are NFFTs of the even (cosine) or odd (sine) extension of f on a
# virtual periodic grid of 2Ñ points per dimension, whose values are even or odd about 0 and Ñ.
# Only the points l = 0…Ñ are stored; the halo holds their mirror images, so the resampling loops
# of the NFFT run unchanged on a real grid. The FFT stage is a DCT-I on l = 0…Ñ or a DST-I on
# l = 1…Ñ−1 (the sine grid vanishes at 0 and Ñ).

# DCT-I / DST-I along every dimension of a contiguous array F (Ñ+1 resp. Ñ−1 points per dimension),
# one real FFT of length 2Ñ per line of the even (odd) extension (FFTW's REDFT00/RODFT00 run in
# O(n²) for most sizes when planned with ESTIMATE, and MEASURE planning takes seconds to minutes).
# Like the complex FFT stage, the passes skip lines that are zero (direct transform: modes outside
# 1:N along the dimensions not yet transformed) or not needed (transposed transform).
struct R2RPass{T}
  d::Int
  Ñ::Int
  A::Int                                  # F viewed as (A, size(F, d), C)
  C::Int
  items::Vector{NTuple{3,Int}}            # (first line along A or C, number of lines, c)
  bufs::Vector{Matrix{T}}                 # per task: (2Ñ, Lb) extended lines
  outs::Vector{Matrix{Complex{T}}}        # per task: (Ñ + 1, Lb) spectra
  plans::Vector{Any}
end

const R2R_LINES = 16

function R2RPass(::Type{T}, F, d, Ñ, N, kept_below, flags, nthreads) where {T}
  M = size(F); D = ndims(F)
  A = prod(M[1:d-1]; init=1); C = prod(M[d+1:end]; init=1)
  items = NTuple{3,Int}[]
  if d == 1
    Lb = D == 1 ? 1 : R2R_LINES
    for c0 in 1:Lb:C
      push!(items, (c0, min(Lb, C - c0 + 1), 0))
    end
  else
    Lb = R2R_LINES
    # dimensions below d: all points, or only 1:N where the data are zero elsewhere
    R = ntuple(e -> kept_below ? (1:N[e]) : (1:M[e]), d - 1)
    LA = LinearIndices(M[1:d-1])
    for c in 1:C, m in CartesianIndices(Base.tail(R)), a0 in first(R[1]):Lb:last(R[1])
      push!(items, (LA[a0, Tuple(m)...], min(Lb, last(R[1]) - a0 + 1), c))
    end
  end
  ntasks = D == 1 ? 1 : min(4nthreads, length(items))
  FFTW.set_num_threads(D == 1 ? nthreads : 1)
  bufs = [zeros(T, 2Ñ, Lb) for _ in 1:ntasks]
  outs = [zeros(Complex{T}, Ñ + 1, Lb) for _ in 1:ntasks]
  plans = Any[FFTW.plan_rfft(b, 1; flags) for b in bufs]
  R2RPass{T}(d, Ñ, A, C, items, bufs, outs, plans)
end

function apply_r2r!(P::R2RPass, F, ::Val{S}) where {S}
  F3 = reshape(F, P.A, size(F, P.d), P.C)
  rs = chunks(eachindex(P.items), length(P.bufs))
  tasks(length(rs)) do t
    buf, out, plan = P.bufs[t], P.outs[t], P.plans[t]
    for i in rs[t]
      x0, nb, c = P.items[i]
      r2r_lines!(F3, buf, out, plan, P.Ñ, x0, nb, c, Val(S))
    end
  end
  F
end

# line b of an item: F3[1, :, x0+b−1] (d = 1, c = 0) or F3[x0+b−1, :, c] (d > 1)
@inline line(F3, x0, b, c) = c == 0 ? (1, x0 + b - 1) : (x0 + b - 1, c)

function r2r_lines!(F3, buf, out, plan, Ñ, x0, nb, c, ::Val{S}) where {S}
  T = eltype(buf)
  @inbounds for b in 1:nb
    a, cc = line(F3, x0, b, c)
    if S === :cos                         # x_l at F index l + 1, l = 0…Ñ
      for l in 0:Ñ
        v = F3[a, l+1, cc]
        buf[l+1, b] = v
        0 < l < Ñ && (buf[2Ñ-l+1, b] = v)
      end
    else                                  # x_l at F index l, l = 1…Ñ−1
      buf[1, b] = zero(T); buf[Ñ+1, b] = zero(T)
      for l in 1:Ñ-1
        v = F3[a, l, cc]
        buf[l+1, b] = v
        buf[2Ñ-l+1, b] = -v
      end
    end
  end
  mul!(out, plan, buf)
  @inbounds for b in 1:nb
    a, cc = line(F3, x0, b, c)
    if S === :cos                         # Y_k = Σ_l x_l (2 − δ_l0 − δ_lÑ) cos(π k l / Ñ)
      for l in 0:Ñ
        F3[a, l+1, cc] = real(out[l+1, b])
      end
    else                                  # Y_k = 2 Σ_l x_l sin(π k l / Ñ)
      for l in 1:Ñ-1
        F3[a, l, cc] = -imag(out[l+1, b])
      end
    end
  end
end

# passes of the direct transform (dimensions in decreasing order, lines with zeros skipped) and of
# the transposed transform (increasing order, only lines needed later)
function r2r_passes(::Type{T}, F, Ñ, N, flags, nthreads) where {T}
  D = ndims(F)
  fwd = [R2RPass(T, F, d, Ñ[d], N, true, flags, nthreads) for d in D:-1:1]
  adj = [R2RPass(T, F, d, Ñ[d], N, true, flags, nthreads) for d in 1:D]
  fwd, adj
end

struct RealCore{S,D,K<:SymKernel,T,Tk}   # S = :cos or :sin
  N::NTuple{D,Int}
  Ñ::NTuple{D,Int}                       # stored points l = 0…Ñ; virtual periodic grid 2Ñ
  W::Int
  B::Int
  kernel::K
  cfwd::NTuple{D,Vector{T}}              # resampling correction factors per mode, direct
  cadj::NTuple{D,Vector{T}}              # and transposed transform
  binned::Binned{D,Tk}
  gp::Array{T,D}                         # padded grid, (Ñ + 1) .+ W
  F::Array{T,D}                          # contiguous copy of the stored points for the FFT stage
  fwd::Vector{R2RPass{T}}
  adj::Vector{R2RPass{T}}
  locks::TileLocks{D}
  bufs::Vector{Array{T,D}}
  flags::UInt32
  nthreads::Int
end

# Modes of the virtual NFFT: n = 0…N−1 (cosine) or 1…N (sine) need an NFFT of size 2N or 2N + 2.
virtual_size(S, N) = S === :cos ? 2 .* N : 2 .* (N .+ 1)

function RealCore{S}(k::AbstractMatrix, N::NTuple{D,Int}, Ñv::NTuple{D,Int}, W, φ, ::Type{T}, fft, B, nthreads) where {S,D,T}
  K = SymKernel(PolyKernel(φ, W, W + 3; T))
  Ñ = Ñv .÷ 2
  # φ̂ of the virtual grid at the modes n ≥ 0; the factors 1/2 come from the DCT-I/DST-I
  # normalisation (Y_l = X_0 + 2 Σ X_n cos(π n l / Ñ), resp. 2 Σ X_n sin(π n l / Ñ))
  phihat = ntuple(D) do d
    if S === :cos
      kernel_ft(φ, W, 2N[d], Ñv[d]; nthreads)[N[d]+1:2N[d]]
    else
      kernel_ft(φ, W, 2N[d] + 2, Ñv[d]; nthreads)[N[d]+3:2N[d]+2]
    end
  end
  cadj = map(p -> T.(1 ./ (2 .* p)), phihat)
  cfwd = S === :cos ? map(p -> T.([1 / p[1]; 1 ./ (2 .* p[2:end])]), phihat) : cadj
  B = B === nothing ? default_tile(D, W) : B
  h = halo(W)
  gp = zeros(T, (Ñ .+ 1) .+ W)
  F = zeros(T, size(stored(gp, h, Ñ, Val(S))))
  fwd, adj = r2r_passes(T, F, Ñ, N, fft.flags, nthreads)
  RealCore{S,D,typeof(K),T,eltype(k)}(N, Ñ, W, B, K, cfwd, cadj, Binned(k, Ñv, B, nthreads), gp, F, fwd, adj,
    TileLocks(size(gp), B), task_buffers(T, ntuple(_ -> B + W + 1, D), nthreads), fft.flags, nthreads)
end

# Padded index of grid point l is l + h[1] + 1. The DCT-I acts on l = 0…Ñ, the DST-I on l = 1…Ñ−1.
stored(gp, h, Ñ, ::Val{:cos}) = view(gp, map(n -> h[1]+1:h[1]+n+1, Ñ)...)
stored(gp, h, Ñ, ::Val{:sin}) = view(gp, map(n -> h[1]+2:h[1]+n, Ñ)...)

# Point l of the virtual grid as a stored point r ∈ 0…Ñ and a sign (period 2Ñ, mirror at 0 and Ñ).
function mirror(l, Ñ, s)
  r = mod(l, 2Ñ)
  r <= Ñ ? (r, one(s)) : (2Ñ - r, s)
end

# Fill the halo from the stored points (before the resampling), or fold it back onto them (after
# the adjoint resampling): one dimension after the other over whole slices, which covers corners.
function fill_mirror!(gp, h, Ñ, s)
  for d in 1:ndims(gp)
    for l in [-h[1]:-1; Ñ[d]+1:Ñ[d]+h[2]]
      r, σ = mirror(l, Ñ[d], s)
      selectdim(gp, d, l + h[1] + 1) .= σ .* selectdim(gp, d, r + h[1] + 1)
    end
  end
  gp
end
function fold_mirror!(gp, h, Ñ, s)
  for d in ndims(gp):-1:1
    for l in [-h[1]:-1; Ñ[d]+1:Ñ[d]+h[2]]
      r, σ = mirror(l, Ñ[d], s)
      selectdim(gp, d, r + h[1] + 1) .+= σ .* selectdim(gp, d, l + h[1] + 1)
    end
  end
  gp
end

# Correction: mode n (index i = 1…N) sits at stored DCT position i, DST position i (point l = i − 1
# resp. l = i); `c[d]` holds the factors per mode.
function real_correction!(g, f, c, N::NTuple{D,Int}, nthreads) where {D}
  @maybe_threads nthreads for r in chunks(1:N[D], nthreads)
    @inbounds for il in r
      cl = c[D][il]
      for I in CartesianIndices(Base.front(N))
        a = cl
        for d in 1:D-1
          a *= c[d][I[d]]
        end
        g[I, il] = a * f[I, il]
      end
    end
  end
  g
end
function real_correction_adjoint!(f, g, c, N::NTuple{D,Int}, nthreads) where {D}
  @maybe_threads nthreads for r in chunks(1:N[D], nthreads)
    @inbounds for il in r
      cl = c[D][il]
      for I in CartesianIndices(Base.front(N))
        a = cl
        for d in 1:D-1
          a *= c[d][I[d]]
        end
        f[I, il] = a * g[I, il]
      end
    end
  end
  f
end

function trafo_core!(out, c::RealCore{S}, f) where {S}
  h = halo(c.W); g = stored(c.gp, h, c.Ñ, Val(S))
  threaded_fill!(c.F, 0, c.nthreads)
  real_correction!(c.F, f, c.cfwd, c.N, c.nthreads)
  for P in c.fwd
    apply_r2r!(P, c.F, Val(S))
  end
  if S === :sin                          # the grid vanishes at l = 0 and l = Ñ
    for d in 1:ndims(c.gp)
      selectdim(c.gp, d, h[1] + 1) .= 0
      selectdim(c.gp, d, h[1] + 1 + c.Ñ[d]) .= 0
    end
  end
  threaded_copy!(g, c.F, c.nthreads)
  fill_mirror!(c.gp, h, c.Ñ, S === :cos ? 1 : -1)
  resample!(out, c.gp, c.kernel, c.binned, 2 .* c.Ñ, c.bufs)
  out
end

function adjoint_core!(f, c::RealCore{S}, vals) where {S}
  h = halo(c.W)
  threaded_fill!(c.gp, 0, c.nthreads)
  resample_adjoint!(c.gp, c.kernel, c.binned, vals, 2 .* c.Ñ, c.locks, c.bufs)
  fold_mirror!(c.gp, h, c.Ñ, S === :cos ? 1 : -1)
  threaded_copy!(c.F, stored(c.gp, h, c.Ñ, Val(S)), c.nthreads)
  if S === :cos
    # the transposed DCT-I is the DCT-I with the end points l = 0 and l = Ñ doubled
    for d in 1:ndims(c.F)
      selectdim(c.F, d, 1) .*= 2
      selectdim(c.F, d, size(c.F, d)) .*= 2
    end
  end
  for P in c.adj
    apply_r2r!(P, c.F, Val(S))
  end
  real_correction_adjoint!(f, c.F, c.cadj, c.N, c.nthreads)
  f
end

Base.copy(c::RealCore) = real_copy(c, c.nthreads)
single_core(c::RealCore) = real_copy(c, 1)

function real_copy(c::RealCore{S,D,K,T,Tk}, nthreads) where {S,D,K,T,Tk}
  F = zero(c.F)
  fwd, adj = r2r_passes(T, F, c.Ñ, c.N, c.flags, nthreads)
  RealCore{S,D,K,T,Tk}(c.N, c.Ñ, c.W, c.B, c.kernel, c.cfwd, c.cadj, c.binned, zero(c.gp), F, fwd, adj,
                       TileLocks(size(c.gp), c.B), task_buffers(T, size(c.bufs[1]), nthreads), c.flags, nthreads)
end

slice_cores(c::RealCore, Nd, J, L) = use_slices(L, J + prod(Nd), c.nthreads) ? single_cores(c, L) : Any[]

"""
    NFCTPlan{T,D,R}
    NFSTPlan{T,D,R}

NFCT and NFST plans of NFFT.jl, created by `plan_nfct(k, N; kwargs...)` and
`plan_nfst(k, N; kwargs...)`, with the definitions of NFFT3: for nodes `k` in [0, 1/2]^d and a real
array `f` of size `N`,

    NFCT: f̂_j = Σ_{n=0}^{N-1} f_n Π_d cos(2π n_d k_dj),   NFST: f̂_j = Σ_{n=1}^{N} f_n Π_d sin(2π n_d k_dj)

(the array index of mode n is n + 1 for the NFCT and n for the NFST). `transpose(p)` is the
transposed transform. Options as for [`NFFTPlan`](@ref) (without `tilesize` for batch dimensions;
batch dimensions loop over the slices).
"""
mutable struct NFCTPlan{T,D,R} <: AbstractNFCTPlan{T,D,R}
  N::NTuple{D,Int}
  NOut::NTuple{R,Int}
  dims::UnitRange{Int}
  k::AbstractMatrix
  kernel::KernelEntry
  accuracy::Float64
  options::GenericOptions{T}
  fft::AbstractFFTBackend
  tilesize::Union{Nothing,Int}
  core::Any
  slicecores::Vector{Any}              # single-threaded cores for many slices (`use_slices`)
end

@doc (@doc NFCTPlan)
mutable struct NFSTPlan{T,D,R} <: AbstractNFSTPlan{T,D,R}
  N::NTuple{D,Int}
  NOut::NTuple{R,Int}
  dims::UnitRange{Int}
  k::AbstractMatrix
  kernel::KernelEntry
  accuracy::Float64
  options::GenericOptions{T}
  fft::AbstractFFTBackend
  tilesize::Union{Nothing,Int}
  core::Any
  slicecores::Vector{Any}              # single-threaded cores for many slices (`use_slices`)
end

const RealPlan{T,D,R} = Union{NFCTPlan{T,D,R},NFSTPlan{T,D,R}}
kind(::NFCTPlan) = :cos
kind(::NFSTPlan) = :sin
kind(::Type{<:NFCTPlan}) = :cos
kind(::Type{<:NFSTPlan}) = :sin

function AbstractNFFTs.plan_nfct(b::NFFTBackend, k::AbstractMatrix, N::Dims{D}; kwargs...) where {D}
  g, rest = generic_options(k, N; kwargs...)
  check_options(b, rest, NFFT_OPTIONS)
  RealPlan(NFCTPlan, k, N, g; rest...)
end
function AbstractNFFTs.plan_nfst(b::NFFTBackend, k::AbstractMatrix, N::Dims{D}; kwargs...) where {D}
  g, rest = generic_options(k, N; kwargs...)
  check_options(b, rest, NFFT_OPTIONS)
  RealPlan(NFSTPlan, k, N, g; rest...)
end

NFCTPlan(k::AbstractVecOrMat, N; kwargs...) = plan_nfct(NFFTBackend(), k, N; kwargs...)
NFSTPlan(k::AbstractVecOrMat, N; kwargs...) = plan_nfst(NFFTBackend(), k, N; kwargs...)

function RealPlan(P::Type, k::AbstractMatrix, N::NTuple{D,Int}, g::GenericOptions{T}; window=:auto,
                  W=nothing, σ=nothing, fft=FFTWBackend(), tilesize=nothing) where {D,T}
  dims = g.dims
  Nv = virtual_size(kind(P), N[dims])
  J = size(k, 2)
  if window isa CustomWindow
    (W === nothing || σ === nothing) && throw(ArgumentError("a CustomWindow needs W and σ"))
    kernel, acc = KernelEntry(window, W, σ, NaN, NaN, NaN, NaN), NaN
  elseif window === :auto || window isa AbstractWindow
    kernel, acc = choose_kernel(g.reltol, Nv, J, T; window, W, σ, Ncost=N[dims])
  else
    throw(ArgumentError("window must be :auto or an AbstractWindow, got $window"))
  end
  NOut = (N[1:first(dims)-1]..., J, N[last(dims)+1:end]...)
  core = real_core(kind(P), k, N[dims], kernel, T, fft, tilesize, g.nthreads, g.storage, batch_sizes(N, dims)...)
  P{T,D,length(NOut)}(N, NOut, dims, to_storage(g.storage, k, T), kernel, acc, g, fft, tilesize, core,
                      slice_cores(core, N[dims], J, prod(batch_sizes(N, dims))))
end

function real_core(S, k, Nd, kernel::KernelEntry, ::Type{T}, fft, tilesize, nthreads, storage=Array, L1=1, L3=1) where {T}
  k = k isa Array ? k : Array(k)
  for x in k
    (isfinite(x) && 0 <= x <= 0.5) ||
      throw(ArgumentError("nodes of an NFCT/NFST must be in [0, 1/2], found $x"))
  end
  Ñv = grid_size(kernel.σ, virtual_size(S, Nd), kernel.W)
  φ = window_function(kernel.window, kernel.W, kernel.β, kernel.e0)
  storage <: Array || return device_real_core(Val(S), storage, k, Nd, Ñv, kernel.W, φ, T, nthreads, L1, L3)
  RealCore{S}(k, Nd, Ñv, kernel.W, φ, T, fft, tilesize, nthreads)
end

# Core for data on a device (GPU array types; the GPU extension), handling all batch dimensions.
device_real_core(S, storage, args...) = device_core(storage)

AbstractNFFTs.size_in(p::RealPlan) = p.N
AbstractNFFTs.size_out(p::RealPlan) = p.NOut
AbstractNFFTs.nodes(p::RealPlan) = p.k
AbstractNFFTs.accuracy(p::RealPlan) = p.accuracy
parameters(p::RealPlan) = (window=p.kernel.window, W=p.kernel.W, σ=p.kernel.σ)

function Base.copy(p::P) where {T,D,R,P<:RealPlan{T,D,R}}
  core = copy(p.core)
  P(p.N, p.NOut, p.dims, p.k, p.kernel, p.accuracy, p.options, p.fft, p.tilesize, core,
    slice_cores(core, p.N[p.dims], size(p.k, 2), prod(batch_sizes(p.N, p.dims))))
end

function AbstractNFFTs.nodes!(p::RealPlan{T}, k::AbstractMatrix) where {T}
  size(k, 1) == length(p.dims) || throw(DimensionMismatch("nodes must have $(length(p.dims)) rows"))
  p.core = real_core(kind(p), k, p.N[p.dims], p.kernel, T, p.fft, p.tilesize, p.options.nthreads,
                     p.options.storage, batch_sizes(p.N, p.dims)...)
  p.slicecores = slice_cores(p.core, p.N[p.dims], size(k, 2), prod(batch_sizes(p.N, p.dims)))
  p.k = to_storage(p.options.storage, k, T)
  p.NOut = (p.N[1:first(p.dims)-1]..., size(k, 2), p.N[last(p.dims)+1:end]...)
  p
end

function Base.show(io::IO, p::RealPlan{T}) where {T}
  e = p.kernel
  print(io, "$(nameof(typeof(p))){$T} for an input of size $(p.N) along dims $(p.dims) at $(size(p.k, 2)) nodes: ",
        "$(nameof(typeof(e.window))) window, W = $(e.W), σ = $(e.σ), predicted error $(p.accuracy)")
end

function LinearAlgebra.mul!(fHat::AbstractArray, p::RealPlan, f::AbstractArray)
  check_sizes(p, f, fHat)
  if isempty(p.slicecores)
    real_transform!(p, p.core, fHat, f)
  else
    fr, gr, L1, L3, Dd = batch_views(p, f, fHat)
    slices_parallel(p.slicecores, L1, L3) do c, a, b
      trafo_core!(view(gr, a, :, b), c, view(fr, a, ntuple(_ -> Colon(), Dd)..., b))
    end
  end
  fHat
end

function LinearAlgebra.mul!(f::AbstractArray, pl::Transpose{T,<:RealPlan{T}}, fHat::AbstractArray) where {T}
  p = pl.parent
  check_sizes(p, f, fHat)
  if isempty(p.slicecores)
    real_transform_adjoint!(p, p.core, f, fHat)
  else
    fr, gr, L1, L3, Dd = batch_views(p, f, fHat)
    slices_parallel(p.slicecores, L1, L3) do c, a, b
      adjoint_core!(view(fr, a, ntuple(_ -> Colon(), Dd)..., b), c, view(gr, a, :, b))
    end
  end
  f
end

# The core argument is the function barrier into code specialised on the kernel; batch dimensions
# loop over the slices.
function real_transform!(p, c, fHat, f)
  fr, gr, L1, L3, Dd = batch_views(p, f, fHat)
  for b in 1:L3, a in 1:L1
    trafo_core!(view(gr, a, :, b), c, view(fr, a, ntuple(_ -> Colon(), Dd)..., b))
  end
end
function real_transform_adjoint!(p, c, f, fHat)
  fr, gr, L1, L3, Dd = batch_views(p, f, fHat)
  for b in 1:L3, a in 1:L1
    adjoint_core!(view(fr, a, ntuple(_ -> Colon(), Dd)..., b), c, view(gr, a, :, b))
  end
end
