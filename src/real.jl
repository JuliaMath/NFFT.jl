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
# Like the complex FFT stage, pruned passes skip lines that are zero (direct transform: modes outside
# 1:N along the dimensions not yet transformed) or not needed (transposed transform); the stage
# interface uses unpruned passes.
struct R2RPass{T,P}
  d::Int
  Ñ::Int
  A::Int                                  # F viewed as (A, size(F, d), C)
  C::Int
  items::Vector{NTuple{3,Int}}            # (first line along A or C, number of lines, c)
  bufs::Vector{Matrix{T}}                 # per task: (2Ñ, Lb) extended lines
  outs::Vector{Matrix{Complex{T}}}        # per task: (Ñ + 1, Lb) spectra
  plans::Vector{P}
end

const R2R_LINES = 16

# lines along dimension e that a pruned pass visits
r2r_length(pruned, N, M, e) = pruned ? N[e] : M[e]

function R2RPass(::Type{T}, F, d, Ñ, N, flags, nthreads; pruned=true) where {T}
  M = size(F); D = ndims(F)
  A = 1; for e in 1:d-1; A *= M[e]; end
  C = 1; for e in d+1:D; C *= M[e]; end
  items = NTuple{3,Int}[]
  Lb = d == 1 && D == 1 ? 1 : R2R_LINES
  if d == 1
    for c0 in 1:Lb:C
      push!(items, (c0, min(Lb, C - c0 + 1), 0))
    end
  else
    # dimensions below d: only 1:N where the data are zero elsewhere; dimension 1 in chunks of Lb,
    # the others (2…d−1) one index at a time, in column-major order
    P = 1; for e in 2:d-1; P *= r2r_length(pruned, N, M, e); end
    for c in 1:C, q in 0:P-1
      lin = 1; r = q; stride = M[1]
      for e in 2:d-1
        n = r2r_length(pruned, N, M, e)
        lin += (r % n) * stride; r ÷= n; stride *= M[e]
      end
      n1 = r2r_length(pruned, N, M, 1)
      for a0 in 1:Lb:n1
        push!(items, (lin + a0 - 1, min(Lb, n1 - a0 + 1), c))
      end
    end
  end
  ntasks = D == 1 ? 1 : min(4nthreads, length(items))
  bufs = [zeros(T, 2Ñ, Lb) for _ in 1:ntasks]
  outs = [zeros(Complex{T}, Ñ + 1, Lb) for _ in 1:ntasks]
  plans = with_fftw_threads(() -> [FFTW.plan_rfft(b, 1; flags) for b in bufs], D == 1 ? nthreads : 1)
  R2RPass(d, Ñ, A, C, items, bufs, outs, plans)
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

# Passes of the direct transform (dimensions in decreasing order, lines with zeros skipped); the
# transposed transform runs the same passes in increasing order (only the lines needed later).
r2r_passes(::Type{T}, F, Ñ, N, flags, nthreads; pruned=true) where {T} =
  [R2RPass(T, F, d, Ñ[d], N, flags, nthreads; pruned) for d in ndims(F):-1:1]

struct RealCore{S,D,K<:SymKernel,T,Tk,P<:R2RPass}   # S = :cos or :sin
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
  passes::Vector{P}                      # direct: in this order, transposed: reversed
  locks::TileLocks{D}
  bufs::Vector{Array{T,D}}
  flags::UInt32
  nthreads::Int
end

# Modes of the virtual NFFT: n = 0…N−1 (cosine) or 1…N (sine) need an NFFT of size 2N or 2N + 2.
virtual_size(S, N) = S === :cos ? 2 .* N : 2 .* (N .+ 1)

function RealCore{S}(k::AbstractMatrix, N::NTuple{D,Int}, Ñv::NTuple{D,Int}, W, φ::Fφ, ::Type{T}, fft::FFTWBackend, B,
                     nthreads; kernel=make_kernel(φ, W, T)) where {S,D,Fφ,T}
  K = kernel
  Ñ = Ñv .÷ 2
  cfwd, cadj = real_factors(Val(S), φ, W, N, Ñv, T, nthreads)
  B = B === nothing ? default_tile(D, W) : B
  h = halo(W)
  gp = zeros(T, (Ñ .+ 1) .+ W)
  F = zeros(T, size(stored(gp, h, Ñ, Val(S))))
  passes = r2r_passes(T, F, Ñ, N, fft.flags, nthreads)
  RealCore{S,D,typeof(K),T,eltype(k),eltype(passes)}(N, Ñ, W, B, K, cfwd, cadj, Binned(k, Ñv, B, nthreads), gp, F,
    passes, TileLocks(size(gp), B), task_buffers(T, ntuple(_ -> B + W + 1, D), nthreads), fft.flags, nthreads)
end

# Correction factors of the direct and the transposed transform per mode: 1/φ̂ of the virtual grid
# at the modes n ≥ 0, with the factors 1/2 of the DCT-I/DST-I normalisation
# (Y_l = X_0 + 2 Σ X_n cos(π n l / Ñ), resp. 2 Σ X_n sin(π n l / Ñ)).
function real_factors(::Val{S}, φ::Fφ, W, N::NTuple{D,Int}, Ñv, ::Type{T}, nthreads) where {S,D,Fφ,T}
  phihat = ntuple(D) do d
    if S === :cos
      kernel_ft(φ, W, 2N[d], Ñv[d]; nthreads)[N[d]+1:2N[d]]
    else
      kernel_ft(φ, W, 2N[d] + 2, Ñv[d]; nthreads)[N[d]+3:2N[d]+2]
    end
  end
  cadj = map(p -> T.(1 ./ (2 .* p)), phihat)
  cfwd = S === :cos ? map(p -> T.([1 / p[1]; 1 ./ (2 .* p[2:end])]), phihat) : cadj
  cfwd, cadj
end

# Padded index of grid point l is l + h[1] + 1. The DCT-I acts on l = 0…Ñ, the DST-I on l = 1…Ñ−1.
stored(gp, h, Ñ, ::Val{:cos}) = view(gp, map(n -> h[1]+1:h[1]+n+1, Ñ)...)
stored(gp, h, Ñ, ::Val{:sin}) = view(gp, map(n -> h[1]+2:h[1]+n, Ñ)...)

# Point l of the virtual grid as a stored point r ∈ 0…Ñ and a sign (period 2Ñ, mirror at 0 and Ñ).
function mirror(l, Ñ, s)
  r = mod(l, 2Ñ)
  r <= Ñ ? (r, one(s)) : (2Ñ - r, s)
end

# The indices of A with index i along dimension d (selectdim with a run-time dimension, as indices).
hyperplane(A::AbstractArray{T,N}, d, i) where {T,N} =
  CartesianIndices(ntuple(e -> e == d ? (i:i) : (1:size(A, e)), Val(N)))

# Fill the halo from the stored points (before the resampling), or fold it back onto them (after
# the adjoint resampling): one dimension after the other over whole slices, which covers corners.
function fill_mirror!(gp, h, Ñ, s)
  for d in 1:ndims(gp)
    for l in [-h[1]:-1; Ñ[d]+1:Ñ[d]+h[2]]
      r, σ = mirror(l, Ñ[d], s)
      for I in hyperplane(gp, d, l + h[1] + 1)
        gp[I] = σ * gp[shift(I, d, r - l)]
      end
    end
  end
  gp
end
function fold_mirror!(gp, h, Ñ, s)
  for d in ndims(gp):-1:1
    for l in [-h[1]:-1; Ñ[d]+1:Ñ[d]+h[2]]
      r, σ = mirror(l, Ñ[d], s)
      for I in hyperplane(gp, d, l + h[1] + 1)
        gp[shift(I, d, r - l)] += σ * gp[I]
      end
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
  threaded_fill!(c.F, 0, c.nthreads)
  real_correction!(c.F, f, c.cfwd, c.N, c.nthreads)
  r2r!(c.F, c.passes, Val(S))
  real_resample!(out, c, c.F)
end

function adjoint_core!(f, c::RealCore{S}, vals) where {S}
  real_resample_adjoint!(c.F, c, vals)
  r2r_transposed!(c.F, c.passes, Val(S))
  real_correction_adjoint!(f, c.F, c.cadj, c.N, c.nthreads)
end

# The stages between the stored points F and the nodes, and the DCT-I/DST-I with its transpose.
function real_resample!(out, c::RealCore{S}, F) where {S}
  h = halo(c.W)
  if S === :sin                          # the grid vanishes at l = 0 and l = Ñ
    for d in 1:ndims(c.gp), i in (h[1] + 1, h[1] + 1 + c.Ñ[d]), I in hyperplane(c.gp, d, i)
      c.gp[I] = 0
    end
  end
  threaded_copy!(stored(c.gp, h, c.Ñ, Val(S)), F, c.nthreads)
  fill_mirror!(c.gp, h, c.Ñ, S === :cos ? 1 : -1)
  resample!(out, c.gp, c.kernel, c.binned, 2 .* c.Ñ, c.bufs)
  out
end

function real_resample_adjoint!(F, c::RealCore{S}, vals) where {S}
  h = halo(c.W)
  threaded_fill!(c.gp, 0, c.nthreads)
  resample_adjoint!(c.gp, c.kernel, c.binned, vals, 2 .* c.Ñ, c.locks, c.bufs)
  fold_mirror!(c.gp, h, c.Ñ, S === :cos ? 1 : -1)
  threaded_copy!(F, stored(c.gp, h, c.Ñ, Val(S)), c.nthreads)
end

function r2r!(F, passes, ::Val{S}) where {S}
  for P in passes
    apply_r2r!(P, F, Val(S))
  end
  F
end

function r2r_transposed!(F, passes, ::Val{S}) where {S}
  if S === :cos
    # the transposed DCT-I is the DCT-I with the end points l = 0 and l = Ñ doubled
    for d in 1:ndims(F), i in (1, size(F, d)), I in hyperplane(F, d, i)
      F[I] *= 2
    end
  end
  for P in Iterators.reverse(passes)
    apply_r2r!(P, F, Val(S))
  end
  F
end

function copy_core(c::RealCore{S,D,K,T,Tk,P}, nthreads) where {S,D,K,T,Tk,P}
  F = zero(c.F)
  RealCore{S,D,K,T,Tk,P}(c.N, c.Ñ, c.W, c.B, c.kernel, c.cfwd, c.cadj, c.binned, zero(c.gp), F,
                       r2r_passes(T, F, c.Ñ, c.N, c.flags, nthreads),
                       TileLocks(size(c.gp), c.B), task_buffers(T, size(c.bufs[1]), nthreads), c.flags, nthreads)
end

slice_size(c::RealCore) = c.N

"""
    NFCTPlan{T,D,R}
    NFSTPlan{T,D,R}

NFCT and NFST plans of NFFT.jl, created by `plan_nfct(k, N; kwargs...)` and
`plan_nfst(k, N; kwargs...)`, with the definitions of NFFT3: for nodes `k` in [0, 1/2]^d and a real
array `f` of size `N`,

    NFCT: f̂_j = Σ_{n=0}^{N-1} f_n Π_d cos(2π n_d k_dj),   NFST: f̂_j = Σ_{n=1}^{N} f_n Π_d sin(2π n_d k_dj)

(the array index of mode n is n + 1 for the NFCT and n for the NFST). `transpose(p)` is the
transposed transform. Options as for [`NFFTPlan`](@ref); batch dimensions run as slices, in
parallel or one after the other.
"""
mutable struct NFCTPlan{T,D,R} <: AbstractNFCTPlan{T,D,R}
  @plan_fields
end

@doc (@doc NFCTPlan)
mutable struct NFSTPlan{T,D,R} <: AbstractNFSTPlan{T,D,R}
  @plan_fields
end

const RealPlan{T,D,R} = Union{NFCTPlan{T,D,R},NFSTPlan{T,D,R}}
kind(::Type{<:NFCTPlan}) = :cos
kind(::Type{<:NFSTPlan}) = :sin

function AbstractNFFTs.plan_nfct(b::NFFTBackend, k::AbstractMatrix, N::Dims{D}; kwargs...) where {D}
  g, rest = generic_options(k, N; kwargs...)
  check_options(b, rest, NFFT_OPTIONS)
  fast_plan(NFCTPlan, k, N, g, out_ndims(N, kwargs); rest...)
end
function AbstractNFFTs.plan_nfst(b::NFFTBackend, k::AbstractMatrix, N::Dims{D}; kwargs...) where {D}
  g, rest = generic_options(k, N; kwargs...)
  check_options(b, rest, NFFT_OPTIONS)
  fast_plan(NFSTPlan, k, N, g, out_ndims(N, kwargs); rest...)
end

NFCTPlan(k::AbstractVecOrMat, N; kwargs...) = plan_nfct(NFFTBackend(), k, N; kwargs...)
NFSTPlan(k::AbstractVecOrMat, N; kwargs...) = plan_nfst(NFFTBackend(), k, N; kwargs...)

# The error model sees the virtual NFFT (twice the size), the cost model the stored half.
error_size(P::Type{<:RealPlan}, Nd) = virtual_size(kind(P), Nd)

check_nodes(::Type{<:RealPlan}, k) = check_range(k, 0.0, 0.5, "nodes of an NFCT/NFST must be in [0, 1/2]")

function make_core(P::Type{<:RealPlan}, k, Nd, kernel::KernelEntry, ::Type{T}, fft, tilesize, nthreads, storage,
                   L1, L3) where {T}
  S = kind(P)
  Ñv = grid_size(kernel.σ, virtual_size(S, Nd), kernel.W)
  φ = window_function(kernel)
  storage <: Array || return device_real_core(Val(S), storage, k, Nd, Ñv, kernel.W, φ, T, nthreads, L1, L3)
  RealCore{S}(k, Nd, Ñv, kernel.W, φ, T, fft, tilesize, nthreads)
end

# Core for data on a device (GPU array types; the GPU extension), handling all batch dimensions.
device_real_core(S, storage, args...) = device_core(storage)

# Stage interface (plans without batch dimensions, on the CPU): the oversampled array is the stored
# grid F, l = 0…Ñ (NFCT) or l = 1…Ñ−1 (NFST) per dimension; the FFT stages are the DCT-I/DST-I and
# its transpose, unpruned, so that the stages can be composed freely.
function real_stage_core(p::RealPlan)
  c = full_core(p)
  c isa RealCore || throw(ArgumentError("the stage interface of NFCT/NFST plans needs a plan on the CPU"))
  c
end

function check_real_stage(p, f, g, fHat)
  c = real_stage_core(p)
  f === nothing || check_size(f, p.N, "the equidistant array")
  g === nothing || check_size(g, size(c.F), "the oversampled array")
  fHat === nothing || check_size(fHat, p.NOut, "the array at the nodes")
  c
end

unpruned_passes(c::RealCore{S,D,K,T}, F) where {S,D,K,T} = r2r_passes(T, F, c.Ñ, c.N, c.flags, c.nthreads; pruned=false)

AbstractNFFTs.oversampled_size(p::RealPlan) = size(real_stage_core(p).F)

function AbstractNFFTs.resampling_correction!(p::RealPlan, f::AbstractArray, g::AbstractArray)
  c = check_real_stage(p, f, g, nothing)
  fill!(g, 0)
  real_correction!(g, f, c.cfwd, c.N, c.nthreads)
end
function AbstractNFFTs.oversampled_fft!(p::RealPlan, g::AbstractArray)
  c = check_real_stage(p, nothing, g, nothing)
  r2r!(g, unpruned_passes(c, g), Val(kind(typeof(p))))
end
AbstractNFFTs.resampling!(p::RealPlan, g::AbstractArray, fHat::AbstractArray) =
  (real_resample!(vec(fHat), check_real_stage(p, nothing, g, fHat), g); fHat)
AbstractNFFTs.resampling_adjoint!(p::RealPlan, fHat::AbstractArray, g::AbstractArray) =
  real_resample_adjoint!(g, check_real_stage(p, nothing, g, fHat), vec(fHat))
# The transforms compute 2·C instead of the transpose diag(1, 2, …, 2, 1)·C of the DCT-I and take the
# factor into `cadj`; the stages are the exact transposes of the direct stages.
function AbstractNFFTs.oversampled_bfft!(p::RealPlan, g::AbstractArray)
  c = check_real_stage(p, nothing, g, nothing)
  r2r_transposed!(g, unpruned_passes(c, g), Val(kind(typeof(p))))
  if kind(typeof(p)) === :cos
    for d in 1:ndims(g), i in (1, size(g, d)), I in hyperplane(g, d, i)
      g[I] /= 2
    end
  end
  g
end
function AbstractNFFTs.resampling_correction_adjoint!(p::RealPlan, g::AbstractArray, f::AbstractArray)
  c = check_real_stage(p, f, g, nothing)
  real_correction_adjoint!(f, g, c.cfwd, c.N, c.nthreads)
end
