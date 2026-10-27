# NFFT, RNFFT, NFCT and NFST on GPUs: the stages as KernelAbstractions kernels, the FFTs through
# AbstractFFTs (cuFFT, rocFFT, Metal's FFT). The plan is set up on the CPU (parameters, binning,
# window polynomial); the device holds, per node in binned order, the offset into the padded grid
# and the fractional coordinate s ∈ [−1, 1] of every dimension, so that the kernels evaluate the
# window in the working precision T (Metal has no Float64) while the grid position itself was
# computed in Float64.
#
# Batch dimensions: the data are viewed as (L1, N..., L3) and (L1, J, L3), the grids as (..., L)
# with L = L1·L3; every kernel runs over all L transforms at once and the FFTs are batched.
#
# Closures inside a @kernel that capture its local variables are boxed (KernelAbstractions wraps the
# body), and a boxed variable crashes the Metal compiler instead of raising an error: kernels call
# helper functions that take everything as arguments.
module NFFTGPUExt

using NFFT, NFFT.AbstractNFFTs
using NFFT: SymKernel, PolyKernel, halo, gridpos, weights, resample_node, kernel_ft, mirror
using NFFT.FFTW: plan_fft!, plan_bfft!, plan_rfft, plan_brfft
using LinearAlgebra: mul!
using GPUArrays: AbstractGPUArray
using KernelAbstractions
const KA = KernelAbstractions

const WORKGROUP = 128

# Adjoint resampling by tiles of the padded grid (`adjoint_tile_kernel!`): workgroup size, nodes per
# chunk, preferred tile edge, and the local memory a workgroup may use (32 KB: Metal's limit, below
# CUDA's 48 KB of static shared memory).
const TILE_WORKGROUP = 64
const TILE_CHUNK = 32
tile_edge(D) = D == 2 ? 32 : 8
const LOCAL_BYTES = 32 * 1024

## nodes and resampling (shared by all transforms)

# Nodes are sorted by bins of the start of their support in the padded grid; bin edges are the tile
# edges. `nt` tiles per dimension, linear tile index with dimension 1 fastest (`tstr` its strides),
# nodes of tile t at tileptr[t]:tileptr[t+1]-1.
struct Tiles{D,P}
  B::NTuple{D,Int}
  nt::NTuple{D,Int}
  tstr::NTuple{D,Int}
  bstr::NTuple{D,Int}
  tileptr::P
end

struct Nodes{K,O,S,P,TL}
  W::Int
  kernel::K
  offs::O                    # (D, J) Int32: index of support point 0 in the padded grid, minus 1
  ss::S                      # (D, J) fractional coordinates s = 2t − 1 in T
  perm::P                    # binned order → input index
  tiles::TL                  # Tiles: adjoint resampling by tiles; nothing: by atomic additions
end

col_strides(n) = (s = 1; map(x -> (r = s; s *= x; r), n))

# Tile edges for the adjoint resampling by tiles: at least W − 1, so that a tile needs the nodes of
# its own bin and of the previous one per dimension; nothing in 1D (one work-item per node with
# atomics is fast there) and when the local buffers do not fit.
function tile_edges(D, W, T)
  D in (2, 3) || return nothing
  B = ntuple(_ -> max(tile_edge(D), W - 1), D)
  bytes = 2sizeof(T) * prod(B) + TILE_CHUNK * D * (W * sizeof(T) + 4) + 2TILE_CHUNK * sizeof(T)
  bytes <= LOCAL_BYTES ? B : nothing
end

# Nodes k (CPU) on a periodic grid of Ñ points per dimension (`gridpos`), padded grid of size P.
function Nodes(storage, k::Matrix, Ñ::NTuple{D,Int}, P::NTuple{D,Int}, W, φ, ::Type{T}) where {D,T}
  K = SymKernel(PolyKernel(φ, W, W + 3; T))
  J = size(k, 2)
  offs = Matrix{Int32}(undef, D, J)
  ss = Matrix{T}(undef, D, J)
  Threads.@threads for j in 1:J
    for d in 1:D
      # as `support` in geometry.jl
      v = gridpos(k[d, j], Ñ[d]) - W / 2
      f = floor(v)
      offs[d, j] = unsafe_trunc(Int, f) + 1 + fld(W, 2)
      ss[d, j] = 2 * T(v - f) - one(T)
    end
  end
  # binned order: work-items with neighbouring indices touch neighbouring parts of the grid
  Bt = tile_edges(D, W, T)
  B = Bt === nothing ? ntuple(_ -> D == 1 ? 1024 : 16, D) : Bt
  nt = cld.(P, B)
  tstr = col_strides(nt)
  tile = [1 + sum(ntuple(d -> (offs[d, j] ÷ B[d]) * tstr[d], D)) for j in 1:J]
  perm = sortperm(tile; alg=Base.Sort.DEFAULT_STABLE)
  counts = zeros(Int, prod(nt))
  for t in tile
    counts[t] += 1
  end
  tileptr = cumsum([1; counts])
  tiles = Bt === nothing ? nothing : Tiles(B, nt, tstr, col_strides(B), storage(Int32.(tileptr)))
  Nodes(W, K, storage(offs[:, perm]), storage(ss[:, perm]), storage(Int32.(perm)), tiles)
end

@inline node_weights(K, ss, q, ::Val{D}) where {D} = ntuple(d -> weights(K, @inbounds(ss[d, q])), Val(D))
@inline node_offset(offs, q, ::Val{D}) where {D} = ntuple(d -> Int(@inbounds(offs[d, q])), Val(D))
# the first D indices, and grid l of an array of grids
@inline front(I, ::Val{D}) where {D} = ntuple(d -> I[d], Val(D))
@inline grid(g, l, ::Val{D}) where {D} = view(g, ntuple(_ -> Colon(), Val(D))..., l)
# batch index l = a + (b − 1)·L1
@inline batch(l, L1) = ((l - 1) % L1 + 1, (l - 1) ÷ L1 + 1)

# One work-item per node and transform; `out` is (L1, J, L3).
@kernel function resample_kernel!(out, @Const(gp), K, @Const(offs), @Const(ss), @Const(perm), L1, ::Val{D}) where {D}
  q, l = @index(Global, NTuple)
  ws = node_weights(K, ss, q, Val(D))
  off = node_offset(offs, q, Val(D))
  a, b = batch(l, L1)
  @inbounds out[a, perm[q], b] = resample_node(grid(gp, l, Val(D)), off, ws)
end

# `gr` is the padded grid as real numbers with a leading dimension (re, im) for complex grids, of
# length 1 for real grids: atomic additions exist for real numbers only.
@kernel function resample_adjoint_kernel!(gr, K, @Const(offs), @Const(ss), @Const(perm), @Const(vals), L1,
                                          ::Val{D}, ::Val{W}) where {D,W}
  q, l = @index(Global, NTuple)
  ws = node_weights(K, ss, q, Val(D))
  off = node_offset(offs, q, Val(D))
  a, b = batch(l, L1)
  v = @inbounds vals[a, perm[q], b]
  w1v = ws[1] .* v
  for It in CartesianIndices(ntuple(_ -> Base.OneTo(W), Val(D - 1)))
    wt = prod(ntuple(d -> ws[d+1][It[d]], Val(D - 1)); init=one(eltype(gr)))
    jt = Base.tail(off) .+ Tuple(It)
    for i in 1:W
      atomic_add!(gr, (off[1] + i, jt..., l), wt * w1v[i])
    end
  end
end

@inline atomic_add!(gr, I, c::Real) = (KA.@atomic gr[1, I...] += c; nothing)
@inline function atomic_add!(gr, I, c::Complex)
  KA.@atomic gr[1, I...] += real(c)
  KA.@atomic gr[2, I...] += imag(c)
  nothing
end

real_view(gp::AbstractArray{<:Complex}) = reinterpret(reshape, real(eltype(gp)), gp)
real_view(gp::AbstractArray{<:Real}) = reshape(gp, 1, size(gp)...)

# Adjoint resampling by tiles, without atomics and deterministic: one workgroup per tile of the
# padded grid and transform accumulates in local memory. It takes the nodes of the 2^D bins that can
# reach the tile in chunks (weights, offsets and values into local memory); for each node the
# work-items split the intersection of its support with the tile, and a barrier separates the nodes.
# The tile is then written to the padded grid, which it covers exactly once. D = 2, 3.
@kernel function adjoint_tile_kernel!(gp, K, @Const(offs), @Const(ss), @Const(perm),
                                      @Const(vals), @Const(tileptr), nt, B, tstr, bstr, L1,
                                      ::Val{D}, ::Val{W}, ::Val{S}, ::Val{C}, ::Type{T}) where {D,W,S,C,T}
  t, lb = batch(@index(Group, Linear), prod(nt))
  a0, b0 = batch(lb, L1)
  li = @index(Local, Linear)
  gs = @groupsize()[1]
  bre = @localmem T (S,)
  bim = @localmem T (S,)
  wb = @localmem T (C * D * W,)
  ob = @localmem Int32 (C * D,)
  vre = @localmem T (C,)
  vim = @localmem T (C,)
  for i in li:gs:S
    @inbounds bre[i] = zero(T); @inbounds bim[i] = zero(T)
  end
  tc = lin2cart(t, tstr, nt)
  o = map(*, tc, B)                  # the tile covers padded indices o .+ (1:B)
  for nb in 0:(1<<D)-1
    # bin tc − δ, δ ∈ {0, 1}^D
    tn = 1; ok = true
    for d in 1:D
      cd = tc[d] - ((nb >> (d - 1)) & 1)
      ok &= cd >= 0
      tn += cd * tstr[d]
    end
    ok || continue
    q0 = Int(@inbounds tileptr[tn]); q1 = Int(@inbounds tileptr[tn+1]) - 1
    for qc in q0:C:q1
      cnt = min(C, q1 - qc + 1)
      @synchronize
      for a in li:gs:cnt*D
        c = (a - 1) ÷ D; d = (a - 1) % D + 1
        q = qc + c
        ws = weights(K, @inbounds ss[d, q])
        for i in 1:W
          @inbounds wb[(c * D + d - 1) * W + i] = ws[i]
        end
        @inbounds ob[c * D + d] = offs[d, q]
      end
      for a in li:gs:cnt
        v = @inbounds vals[a0, perm[qc + a - 1], b0]
        @inbounds vre[a] = real(v); @inbounds vim[a] = imag(v)
      end
      @synchronize
      for c in 0:cnt-1
        # support point i ∈ 1:W lies at padded index off + i; its intersection with the tile is
        # lo .+ (0:len-1), the same for all work-items, so the barrier below is uniform
        lo1, l1 = intersection(ob[c*D+1], o[1], B[1], W)
        lo2, l2 = intersection(ob[c*D+2], o[2], B[2], W)
        lo3, l3 = D == 3 ? intersection(ob[c*D+3], o[3], B[3], W) : (Int32(1), Int32(1))
        n = l1 * l2 * l3
        if n > 0
          v1 = @inbounds vre[c+1]; v2 = @inbounds vim[c+1]
          for p in Int32(li - 1):Int32(gs):n-Int32(1)
            r = p ÷ l1
            i1 = p - r * l1 + lo1
            i2 = r % l2 + lo2
            i3 = r ÷ l2 + lo3
            w = @inbounds wb[c*D*W+i1] * wb[(c*D+1)*W+i2]
            l = (@inbounds(ob[c*D+1]) + i1 - Int32(o[1]) - Int32(1)) +
                (@inbounds(ob[c*D+2]) + i2 - Int32(o[2]) - Int32(1)) * Int32(bstr[2])
            if D == 3
              w *= @inbounds wb[(c*D+2)*W+i3]
              l += (@inbounds(ob[c*D+3]) + i3 - Int32(o[3]) - Int32(1)) * Int32(bstr[3])
            end
            @inbounds bre[l+1] += w * v1
            @inbounds bim[l+1] += w * v2
          end
          @synchronize
        end
      end
    end
  end
  @synchronize
  for i in li:gs:prod(B)
    gi = map((x, y) -> x + y + 1, o, lin2cart(i, bstr, B))
    if all(map(<=, gi, front(size(gp), Val(D))))
      @inbounds gp[gi..., lb] = grid_value(eltype(gp), bre[i], bim[i])
    end
  end
end

@inline lin2cart(i, str, n) = map((s, m) -> ((i - 1) ÷ s) % m, str, n)
@inline grid_value(::Type{<:Complex}, re, im) = complex(re, im)
@inline grid_value(::Type{<:Real}, re, im) = re

# Support points lo .+ (0:len-1) (of 1:W) of a node with offset `off` inside the tile o .+ (1:B).
@inline function intersection(off, o, B, W)
  lo = max(Int32(1), Int32(o) + Int32(1) - Int32(off))
  hi = min(Int32(W), Int32(o) + Int32(B) - Int32(off))
  lo, max(hi - lo + Int32(1), Int32(0))
end

launch(backend, k, args...; ndrange) = prod(ndrange) == 0 || k(backend, WORKGROUP)(args...; ndrange)

# values (L1, J, L3) at the nodes from the padded grids gp
resample!(backend, nd::Nodes, vals, gp, L1, ::Val{D}) where {D} =
  launch(backend, resample_kernel!, vals, gp, nd.kernel, nd.offs, nd.ss, nd.perm, L1, Val(D);
         ndrange=(length(nd.perm), size(gp, D + 1)))

# padded grids gp (all points written) from the values (L1, J, L3)
function resample_adjoint!(backend, nd::Nodes, gp, vals, L1, ::Val{D}) where {D}
  tl = nd.tiles
  if tl === nothing
    fill!(gp, 0)
    launch(backend, resample_adjoint_kernel!, real_view(gp), nd.kernel, nd.offs, nd.ss, nd.perm, vals, L1,
           Val(D), Val(nd.W); ndrange=(length(nd.perm), size(gp, D + 1)))
  else
    adjoint_tile_kernel!(backend, TILE_WORKGROUP)(gp, nd.kernel, nd.offs, nd.ss, nd.perm, vals, tl.tileptr,
      tl.nt, tl.B, tl.tstr, tl.bstr, L1, Val(D), Val(nd.W), Val(prod(tl.B)), Val(TILE_CHUNK), real(eltype(gp));
      ndrange=prod(tl.nt) * size(gp, D + 1) * TILE_WORKGROUP)
  end
  gp
end

## periodic padding (NFFT, RNFFT)

# Padded grids from the grids: every padded point reads its periodic partner (interior and halo).
@inline periodic(I, h, Ñ) = map((i, ñ) -> mod(i - 1 - h, ñ) + 1, I, Ñ)

@kernel function pad_kernel!(gp, @Const(gf), h::Int, ::Val{D}) where {D}
  I = Tuple(@index(Global, Cartesian))
  src = periodic(front(I, Val(D)), h, front(size(gf), Val(D)))
  @inbounds gp[I...] = gf[src..., I[D+1]]
end

# Grids from the padded grids: every point sums its periodic images (folds the halo without
# atomics). The grid has at least W points, so the images are p − Ñ, p, p + Ñ.
@kernel function unpad_kernel!(gf, @Const(gp), h::Int, ::Val{D}) where {D}
  I = Tuple(@index(Global, Cartesian))
  acc = zero(eltype(gf))
  for c in CartesianIndices(ntuple(_ -> 3, Val(D)))
    p = image(front(I, Val(D)), Tuple(c), h, front(size(gf), Val(D)))
    if all(map((x, s) -> 1 <= x <= s, p, front(size(gp), Val(D))))
      @inbounds acc += gp[p..., I[D+1]]
    end
  end
  @inbounds gf[I...] = acc
end
@inline image(i, c, h, Ñ) = map((x, s, n) -> x + h + (s - 2) * n, i, c, Ñ)

pad!(backend, gp, gf, W, ::Val{D}) where {D} = launch(backend, pad_kernel!, gp, gf, halo(W)[1], Val(D); ndrange=size(gp))
unpad!(backend, gf, gp, W, ::Val{D}) where {D} = launch(backend, unpad_kernel!, gf, gp, halo(W)[1], Val(D); ndrange=size(gf))

function check_device(backend, xs...)
  for x in xs
    KA.get_backend(x) == backend ||
      throw(ArgumentError("the data must be on the device of the plan, got $(typeof(x))"))
  end
end

nbatch(c) = c.L1 * c.L3
node_values(c, v) = reshape(v, c.L1, :, c.L3)

## NFFT

struct GPUCore{D,ND,V,G,PF,PB,B}
  N::NTuple{D,Int}
  Ñ::NTuple{D,Int}
  L1::Int                    # batch size before and after the transformed dimensions
  L3::Int
  W::Int
  nodes::ND
  invphi::NTuple{D,V}        # 1/φ̂ of the kept modes per dimension
  gp::G                      # padded grids (Ñ .+ W..., L) (halo as in geometry.jl)
  gf::G                      # grids (Ñ..., L) for the FFT
  pf::PF
  pb::PB
  backend::B
end

function NFFT.device_core(storage::Type{<:AbstractGPUArray}, k::Matrix, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, W, φ,
                          ::Type{T}, nthreads, L1, L3) where {D,T}
  invphi = ntuple(d -> T.(1 ./ kernel_ft(φ, W, N[d], Ñ[d]; nthreads)), D)
  nd = Nodes(storage, k, Ñ, Ñ .+ W, W, φ, T)
  gf = storage(zeros(Complex{T}, Ñ..., L1 * L3))
  backend = KA.get_backend(gf)
  gp = KA.zeros(backend, Complex{T}, (Ñ .+ W)..., L1 * L3)
  GPUCore(N, Ñ, L1, L3, W, nd, map(storage, invphi), gp, gf, plan_fft!(gf, 1:D), plan_bfft!(gf, 1:D), backend)
end

# A copy has its own grids and FFT plans and shares the rest.
function Base.copy(c::GPUCore{D}) where {D}
  gp = similar(c.gp); gf = similar(c.gf)
  GPUCore(c.N, c.Ñ, c.L1, c.L3, c.W, c.nodes, c.invphi, gp, gf, plan_fft!(gf, 1:D), plan_bfft!(gf, 1:D), c.backend)
end

# Resampling correction between the modes f, (L1, N..., L3), and the grids gf, (Ñ..., L): one
# work-item per mode and transform.
@kernel function correction_kernel!(gf, f, @Const(invphi), N, L1, ::Val{D}, adjoint::Bool) where {D}
  I = Tuple(@index(Global, Cartesian))
  m = front(I, Val(D)); l = I[D+1]
  a, b = batch(l, L1)
  c = one(real(eltype(gf)))
  for d in 1:D
    @inbounds c *= invphi[d][m[d]]
  end
  gi = map(NFFT.gridindex, m, N, front(size(gf), Val(D)))
  if adjoint
    @inbounds f[a, m..., b] = c * gf[gi..., l]
  else
    @inbounds gf[gi..., l] = c * f[a, m..., b]
  end
end

modes(c::GPUCore, f) = reshape(f, c.L1, c.N..., c.L3)

correction!(c::GPUCore{D}, g, f) where {D} =
  (fill!(g, 0); launch(c.backend, correction_kernel!, g, f, c.invphi, c.N, c.L1, Val(D), false; ndrange=(c.N..., nbatch(c))); g)
correction_adjoint!(c::GPUCore{D}, f, g) where {D} =
  (launch(c.backend, correction_kernel!, g, f, c.invphi, c.N, c.L1, Val(D), true; ndrange=(c.N..., nbatch(c))); f)
resampling!(c::GPUCore{D}, vals, g) where {D} =
  (pad!(c.backend, c.gp, g, c.W, Val(D)); resample!(c.backend, c.nodes, vals, c.gp, c.L1, Val(D)); vals)
resampling_adjoint!(c::GPUCore{D}, g, vals) where {D} =
  (resample_adjoint!(c.backend, c.nodes, c.gp, vals, c.L1, Val(D)); unpad!(c.backend, g, c.gp, c.W, Val(D)); g)

function NFFT.trafo_core!(out, c::GPUCore, f)
  check_device(c.backend, out, f)
  correction!(c, c.gf, modes(c, f))
  c.pf * c.gf
  resampling!(c, node_values(c, out), c.gf)
  out
end

function NFFT.adjoint_core!(f, c::GPUCore, vals)
  check_device(c.backend, f, vals)
  resampling_adjoint!(c, c.gf, node_values(c, vals))
  c.pb * c.gf
  correction_adjoint!(c, modes(c, f), c.gf)
  f
end

# Stage interface: plans without batch dimensions, g of size Ñ.
grids(g) = reshape(g, size(g)..., 1)
NFFT.correction_stage!(c::GPUCore, g, f) =
  (check_device(c.backend, g, f); correction!(c, grids(g), modes(c, f)); g)
NFFT.correction_adjoint_stage!(c::GPUCore, f, g) =
  (check_device(c.backend, f, g); correction_adjoint!(c, modes(c, f), grids(g)); f)
NFFT.resampling_stage!(c::GPUCore, fHat, g) =
  (check_device(c.backend, fHat, g); resampling!(c, node_values(c, fHat), grids(g)); fHat)
NFFT.resampling_adjoint_stage!(c::GPUCore, g, fHat) =
  (check_device(c.backend, g, fHat); resampling_adjoint!(c, grids(g), node_values(c, fHat)); g)

## RNFFT (see rnfft.jl): half spectra, backward FFTs along 2…D, c2r FFT along 1 into real grids

struct RGPUCore{D,ND,V,G,H,P1,P2,P3,P4,B}
  N::NTuple{D,Int}           # logical size; the input has Nh = N₁÷2 + 1 modes along dimension 1
  Nh::Int
  Ñ::NTuple{D,Int}
  L1::Int
  L3::Int
  W::Int
  nodes::ND
  invphi::NTuple{D,V}        # dimension 1: n₁ = 0…Nh−1; others as in the NFFT
  gp::G                      # padded real grids (Ñ .+ W..., L)
  gf::G                      # real grids (Ñ..., L)
  H::H                       # half spectra (Ñ₁÷2+1, Ñ₂, …, L)
  pb::P1                     # backward FFTs along 2…D (nothing in 1D)
  pf::P2                     # forward FFTs along 2…D
  c2r::P3
  r2c::P4
  backend::B
end

function NFFT.device_rnfft_core(storage::Type{<:AbstractGPUArray}, k::Matrix, N::NTuple{D,Int}, Ñ::NTuple{D,Int},
                                W, φ, ::Type{T}, nthreads, L1, L3) where {D,T}
  Nh = N[1] ÷ 2 + 1
  phi1 = kernel_ft(φ, W, 2Nh, Ñ[1]; nthreads)[Nh+1:2Nh]          # modes −Nh…Nh−1, keep 0…Nh−1
  invphi = ntuple(d -> d == 1 ? T.(1 ./ phi1) : T.(1 ./ kernel_ft(φ, W, N[d], Ñ[d]; nthreads)), D)
  nd = Nodes(storage, k, Ñ, Ñ .+ W, W, φ, T)
  gf = storage(zeros(T, Ñ..., L1 * L3))
  backend = KA.get_backend(gf)
  gp = KA.zeros(backend, T, (Ñ .+ W)..., L1 * L3)
  rnfft_core(N, Nh, Ñ, L1, L3, W, nd, map(storage, invphi), gp, gf, backend)
end

function rnfft_core(N::NTuple{D}, Nh, Ñ, L1, L3, W, nd, invphi, gp, gf, backend) where {D}
  H = KA.zeros(backend, complex(eltype(gf)), Ñ[1] ÷ 2 + 1, Base.tail(Ñ)..., L1 * L3)
  pb = D == 1 ? nothing : plan_bfft!(H, 2:D)
  pf = D == 1 ? nothing : plan_fft!(H, 2:D)
  RGPUCore(N, Nh, Ñ, L1, L3, W, nd, invphi, gp, gf, H, pb, pf, plan_brfft(H, Ñ[1], 1), plan_rfft(gf, 1), backend)
end

Base.copy(c::RGPUCore) =
  rnfft_core(c.N, c.Nh, c.Ñ, c.L1, c.L3, c.W, c.nodes, c.invphi, similar(c.gp), similar(c.gf), c.backend)

# the input of the plan is (L1, Nh, N₂…, L3)
modes(c::RGPUCore, h) = reshape(h, c.L1, c.Nh, Base.tail(c.N)..., c.L3)

# Correction between the half input h and the half spectra H: conj(h)/φ̂, halved for n₁ ≥ 1 in the
# direct transform (the c2r FFT doubles those modes); one work-item per mode and transform.
@kernel function rnfft_correction_kernel!(H, h, @Const(invphi), N, L1, ::Val{D}, adjoint::Bool) where {D}
  I = Tuple(@index(Global, Cartesian))
  m = front(I, Val(D)); l = I[D+1]
  a, b = batch(l, L1)
  c = one(real(eltype(H)))
  for d in 1:D
    @inbounds c *= invphi[d][m[d]]
  end
  gi = (m[1], map(NFFT.gridindex, Base.tail(m), Base.tail(N), Base.tail(front(size(H), Val(D))))...)
  if adjoint
    @inbounds h[a, m..., b] = c * conj(H[gi..., l])
  else
    @inbounds H[gi..., l] = (m[1] == 1 ? c : c / 2) * conj(h[a, m..., b])
  end
end

function NFFT.trafo_core!(out, c::RGPUCore{D}, h) where {D}
  check_device(c.backend, out, h)
  fill!(c.H, 0)
  launch(c.backend, rnfft_correction_kernel!, c.H, modes(c, h), c.invphi, c.N, c.L1, Val(D), false;
         ndrange=(c.Nh, Base.tail(c.N)..., nbatch(c)))
  D > 1 && c.pb * c.H
  mul!(c.gf, c.c2r, c.H)
  pad!(c.backend, c.gp, c.gf, c.W, Val(D))
  resample!(c.backend, c.nodes, node_values(c, out), c.gp, c.L1, Val(D))
  out
end

function NFFT.adjoint_core!(h, c::RGPUCore{D}, vals) where {D}
  check_device(c.backend, h, vals)
  resample_adjoint!(c.backend, c.nodes, c.gp, node_values(c, vals), c.L1, Val(D))
  unpad!(c.backend, c.gf, c.gp, c.W, Val(D))
  mul!(c.H, c.r2c, c.gf)
  D > 1 && c.pf * c.H
  launch(c.backend, rnfft_correction_kernel!, c.H, modes(c, h), c.invphi, c.N, c.L1, Val(D), true;
         ndrange=(c.Nh, Base.tail(c.N)..., nbatch(c)))
  h
end

# the core handles all batch dimensions at once
NFFT.slice_cores(c::RGPUCore, Nd, J, L) = Any[]
NFFT.rnfft_loop!(c::RGPUCore, gr, hr, L1, L3, Dd) = NFFT.trafo_core!(gr, c, hr)
NFFT.rnfft_adjoint_loop!(c::RGPUCore, hr, gr, L1, L3, Dd) = NFFT.adjoint_core!(hr, c, gr)

## NFCT, NFST (see real.jl): stored points F of the even/odd extension, DCT-I/DST-I per dimension

struct RealGPUCore{S,D,ND,V,G,E,P,B}
  N::NTuple{D,Int}
  Ñ::NTuple{D,Int}           # stored points l = 0…Ñ; virtual periodic grid 2Ñ
  L1::Int
  L3::Int
  W::Int
  nodes::ND
  cfwd::NTuple{D,V}          # correction factors per mode, direct and transposed transform
  cadj::NTuple{D,V}
  gp::G                      # padded real grids ((Ñ + 1) .+ W..., L)
  F::G                       # stored points (Ñ + 1 resp. Ñ − 1 per dimension, L)
  E::E                       # per dimension d: F with dimension d extended to 2Ñ[d], complex
  pE::P                      # FFTs of E[d] along d
  backend::B
end

nstored(S, Ñ) = S === :cos ? Ñ .+ 1 : Ñ .- 1

function NFFT.device_real_core(::Val{S}, storage::Type{<:AbstractGPUArray}, k::Matrix, N::NTuple{D,Int},
                               Ñv::NTuple{D,Int}, W, φ, ::Type{T}, nthreads, L1, L3) where {S,D,T}
  Ñ = Ñv .÷ 2
  # as in RealCore (real.jl)
  phihat = ntuple(D) do d
    if S === :cos
      kernel_ft(φ, W, 2N[d], Ñv[d]; nthreads)[N[d]+1:2N[d]]
    else
      kernel_ft(φ, W, 2N[d] + 2, Ñv[d]; nthreads)[N[d]+3:2N[d]+2]
    end
  end
  cadj = map(p -> T.(1 ./ (2 .* p)), phihat)
  cfwd = S === :cos ? map(p -> T.([1 / p[1]; 1 ./ (2 .* p[2:end])]), phihat) : cadj
  nd = Nodes(storage, k, Ñv, (Ñ .+ 1) .+ W, W, φ, T)
  F = storage(zeros(T, nstored(S, Ñ)..., L1 * L3))
  backend = KA.get_backend(F)
  gp = KA.zeros(backend, T, ((Ñ .+ 1) .+ W)..., L1 * L3)
  real_core(Val(S), N, Ñ, L1, L3, W, nd, map(storage, cfwd), map(storage, cadj), gp, F, backend)
end

function real_core(::Val{S}, N::NTuple{D}, Ñ, L1, L3, W, nd, cfwd, cadj, gp, F, backend) where {S,D}
  E = ntuple(d -> KA.zeros(backend, complex(eltype(F)), Base.setindex(size(F), 2Ñ[d], d)...), D)
  pE = ntuple(d -> plan_fft!(E[d], d), D)
  RealGPUCore{S,D,typeof(nd),eltype(cfwd),typeof(gp),typeof(E),typeof(pE),typeof(backend)}(
    N, Ñ, L1, L3, W, nd, cfwd, cadj, gp, F, E, pE, backend)
end

Base.copy(c::RealGPUCore{S}) where {S} =
  real_core(Val(S), c.N, c.Ñ, c.L1, c.L3, c.W, c.nodes, c.cfwd, c.cadj, similar(c.gp), similar(c.F), c.backend)

modes(c::RealGPUCore, f) = reshape(f, c.L1, c.N..., c.L3)

# Correction between the modes f and the stored points F: mode i (1-based) sits at F index i for
# both transforms (point l = i − 1 of the DCT-I, l = i of the DST-I).
@kernel function real_correction_kernel!(F, f, @Const(cs), L1, ::Val{D}, adjoint::Bool) where {D}
  I = Tuple(@index(Global, Cartesian))
  m = front(I, Val(D)); l = I[D+1]
  a, b = batch(l, L1)
  c = one(eltype(F))
  for d in 1:D
    @inbounds c *= cs[d][m[d]]
  end
  if adjoint
    @inbounds f[a, m..., b] = c * F[m..., l]
  else
    @inbounds F[m..., l] = c * f[a, m..., b]
  end
end

# DCT-I / DST-I along dimension d: each line extended to the even (odd) sequence of length 2Ñ, a
# complex FFT, then the real part (cosine: Y_k = Σ_l x_l (2 − δ_l0 − δ_lÑ) cos(πkl/Ñ)) or minus the
# imaginary part (sine: Y_k = 2 Σ_l x_l sin(πkl/Ñ)).
@kernel function extend_kernel!(E, @Const(F), Ñ::Int, d::Int, ::Val{S}) where {S}
  I = Tuple(@index(Global, Cartesian))
  @inbounds E[I...] = extended(F, I, I[d] - 1, Ñ, d, Val(S))
end
@inline function extended(F, I, m, Ñ, d, ::Val{S}) where {S}
  r, σ = mirror(m, Ñ, -1)
  if S === :cos
    @inbounds F[Base.setindex(I, r + 1, d)...]
  elseif 0 < r < Ñ
    @inbounds σ * F[Base.setindex(I, r, d)...]
  else
    zero(eltype(F))
  end
end

@kernel function take_kernel!(F, @Const(E), d::Int, ::Val{S}) where {S}
  I = Tuple(@index(Global, Cartesian))
  if S === :cos
    @inbounds F[I...] = real(E[I...])
  else
    @inbounds F[I...] = -imag(E[Base.setindex(I, I[d] + 1, d)...])
  end
end

function r2r!(c::RealGPUCore{S}, d) where {S}
  launch(c.backend, extend_kernel!, c.E[d], c.F, c.Ñ[d], d, Val(S); ndrange=size(c.E[d]))
  c.pE[d] * c.E[d]
  launch(c.backend, take_kernel!, c.F, c.E[d], d, Val(S); ndrange=size(c.F))
end

# Padded grids from the stored points: every padded point is a stored point of the even (odd)
# extension, with a sign; the sine grid vanishes at l = 0 and l = Ñ.
@kernel function pad_mirror_kernel!(gp, @Const(F), Ñ, h::Int, ::Val{D}, ::Val{S}) where {D,S}
  I = Tuple(@index(Global, Cartesian))
  @inbounds gp[I...] = mirrored(F, front(I, Val(D)), I[D+1], Ñ, h, Val(S))
end
@inline function mirrored(F, p, l, Ñ, h, ::Val{S}) where {S}
  rs = map((x, n) -> mirror(x - 1 - h, n, -1), p, Ñ)
  r = map(first, rs)
  if S === :cos
    @inbounds F[map(x -> x + 1, r)..., l]
  elseif all(map((x, n) -> 0 < x < n, r, Ñ))
    @inbounds prod(map(last, rs)) * F[r..., l]
  else
    zero(eltype(F))
  end
end

# Stored points from the padded grids: every stored point sums its images with their signs (folds
# the halo without atomics). The transposed DCT-I has the end points l = 0 and l = Ñ doubled.
@kernel function fold_mirror_kernel!(F, @Const(gp), Ñ, h::Int, ::Val{D}, ::Val{S}) where {D,S}
  I = Tuple(@index(Global, Cartesian))
  r = map(x -> S === :cos ? x - 1 : x, front(I, Val(D)))
  acc = zero(eltype(F))
  for c in CartesianIndices(ntuple(_ -> 5, Val(D)))
    w, p = mirror_image(r, Tuple(c), Ñ, h, front(size(gp), Val(D)), Val(S))
    if w != 0
      @inbounds acc += w * gp[p..., I[D+1]]
    end
  end
  @inbounds F[I...] = endpoint_factor(r, Ñ, Val(S)) * acc
end
@inline endpoint_factor(r, Ñ, ::Val{:cos}) = prod(map((x, n) -> (x == 0 || x == n) ? 2 : 1, r, Ñ))
@inline endpoint_factor(r, Ñ, ::Val{:sin}) = 1

# Image c[d] ∈ 1:5 of the stored point r[d] in every dimension (point l = r, −r, 2Ñ − r, r − 2Ñ,
# 2Ñ + r): its padded index and the product of the signs; weight 0 if an image lies outside the
# padded grid, repeats an earlier one, or is not mapped back onto r by `mirror`.
@inline function mirror_image(r, c, Ñ, h, P, ::Val{S}) where {S}
  ws = map((x, k, n, q) -> image_sign(x, k, n, h, q, S === :cos ? 1 : -1), r, c, Ñ, P)
  prod(map(first, ws)), map(last, ws)
end
@inline candidate(k, r, Ñ) = k == 1 ? r : k == 2 ? -r : k == 3 ? 2Ñ - r : k == 4 ? r - 2Ñ : 2Ñ + r
@inline function image_sign(r, k, Ñ, h, P, s)
  l = candidate(k, r, Ñ)
  dup = false
  for j in 1:k-1
    dup |= candidate(j, r, Ñ) == l
  end
  rr, σ = mirror(l, Ñ, s)
  p = l + h + 1
  (1 <= p <= P && !dup && rr == r ? σ : zero(σ)), p
end

function NFFT.trafo_core!(out, c::RealGPUCore{S,D}, f) where {S,D}
  check_device(c.backend, out, f)
  fill!(c.F, 0)
  launch(c.backend, real_correction_kernel!, c.F, modes(c, f), c.cfwd, c.L1, Val(D), false; ndrange=(c.N..., nbatch(c)))
  for d in D:-1:1
    r2r!(c, d)
  end
  launch(c.backend, pad_mirror_kernel!, c.gp, c.F, c.Ñ, halo(c.W)[1], Val(D), Val(S); ndrange=size(c.gp))
  resample!(c.backend, c.nodes, node_values(c, out), c.gp, c.L1, Val(D))
  out
end

function NFFT.adjoint_core!(f, c::RealGPUCore{S,D}, vals) where {S,D}
  check_device(c.backend, f, vals)
  resample_adjoint!(c.backend, c.nodes, c.gp, node_values(c, vals), c.L1, Val(D))
  launch(c.backend, fold_mirror_kernel!, c.F, c.gp, c.Ñ, halo(c.W)[1], Val(D), Val(S); ndrange=size(c.F))
  for d in 1:D
    r2r!(c, d)
  end
  launch(c.backend, real_correction_kernel!, c.F, modes(c, f), c.cadj, c.L1, Val(D), true; ndrange=(c.N..., nbatch(c)))
  f
end

# the core handles all batch dimensions at once
NFFT.slice_cores(c::RealGPUCore, Nd, J, L) = Any[]
NFFT.real_transform!(p, c::RealGPUCore, fHat, f) = NFFT.trafo_core!(fHat, c, f)
NFFT.real_transform_adjoint!(p, c::RealGPUCore, f, fHat) = NFFT.adjoint_core!(f, c, fHat)

end
