# NFFT on GPUs: the stages as KernelAbstractions kernels, the FFT through AbstractFFTs (cuFFT,
# rocFFT, Metal's FFT). The plan is set up on the CPU (parameters, binning, window polynomial); the
# device holds, per node in binned order, the offset into the padded grid and the fractional
# coordinate s ∈ [−1, 1] of every dimension, so that the kernels evaluate the window in the working
# precision T (Metal has no Float64) while the grid position itself was computed in Float64.
#
# Batch dimensions: the data are viewed as (L1, N..., L3) and (L1, J, L3), the grids as (Ñ..., L)
# with L = L1·L3; every kernel runs over all L transforms at once and the FFT is batched.
#
# Closures inside a @kernel that capture its local variables are boxed (KernelAbstractions wraps the
# body), and a boxed variable crashes the Metal compiler instead of raising an error: kernels call
# helper functions that take everything as arguments.
module NFFTGPUExt

using NFFT, NFFT.AbstractNFFTs
using NFFT: SymKernel, PolyKernel, NFFTPlan, halo, gridpos, weights, resample_node, kernel_ft
using NFFT.FFTW: plan_fft!, plan_bfft!
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

struct GPUCore{D,K,V,O,S,P,G,PF,PB,B,TL}
  N::NTuple{D,Int}
  Ñ::NTuple{D,Int}
  L1::Int                    # batch size before and after the transformed dimensions
  L3::Int
  W::Int
  kernel::K
  invphi::NTuple{D,V}        # 1/φ̂ of the kept modes per dimension
  offs::O                    # (D, J) Int32: index of support point 0 in the padded grid, minus 1
  ss::S                      # (D, J) fractional coordinates s = 2t − 1 in T
  perm::P                    # binned order → input index
  gp::G                      # padded grids (Ñ .+ W..., L) (halo as in geometry.jl)
  gf::G                      # grids (Ñ..., L) for the FFT
  pf::PF
  pb::PB
  backend::B
  tiles::TL                  # Tiles: adjoint resampling by tiles; nothing: by atomic additions
end

strides(n) = (s = 1; map(x -> (r = s; s *= x; r), n))

# Tile edges for the adjoint resampling by tiles: at least W − 1, so that a tile needs the nodes of
# its own bin and of the previous one per dimension; nothing in 1D (one work-item per node with
# atomics is fast there) and when the local buffers do not fit.
function tile_edges(D, W, T)
  D in (2, 3) || return nothing
  B = ntuple(_ -> max(tile_edge(D), W - 1), D)
  bytes = 2sizeof(T) * prod(B) + TILE_CHUNK * D * (W * sizeof(T) + 4) + 2TILE_CHUNK * sizeof(T)
  bytes <= LOCAL_BYTES ? B : nothing
end

function NFFT.device_core(storage::Type{<:AbstractGPUArray}, k::Matrix, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, W, φ,
                          ::Type{T}, nthreads, L1, L3) where {D,T}
  K = SymKernel(PolyKernel(φ, W, W + 3; T))
  invphi = ntuple(d -> T.(1 ./ kernel_ft(φ, W, N[d], Ñ[d]; nthreads)), D)
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
  nt = cld.(Ñ .+ W, B)
  tstr = strides(nt)
  tile = [1 + sum(ntuple(d -> (offs[d, j] ÷ B[d]) * tstr[d], D)) for j in 1:J]
  perm = sortperm(tile; alg=Base.Sort.DEFAULT_STABLE)
  counts = zeros(Int, prod(nt))
  for t in tile
    counts[t] += 1
  end
  tileptr = cumsum([1; counts])
  gf = storage(zeros(Complex{T}, Ñ..., L1 * L3))
  backend = KA.get_backend(gf)
  gp = KA.zeros(backend, Complex{T}, (Ñ .+ W)..., L1 * L3)
  tiles = Bt === nothing ? nothing : Tiles(B, nt, tstr, strides(B), storage(Int32.(tileptr)))
  GPUCore(N, Ñ, L1, L3, W, K, map(storage, invphi), storage(offs[:, perm]), storage(ss[:, perm]),
          storage(Int32.(perm)), gp, gf, plan_fft!(gf, 1:D), plan_bfft!(gf, 1:D), backend, tiles)
end

# A copy has its own grids and FFT plans and shares the rest.
function Base.copy(c::GPUCore)
  gp = similar(c.gp); gf = similar(c.gf)
  D = length(c.N)
  GPUCore(c.N, c.Ñ, c.L1, c.L3, c.W, c.kernel, c.invphi, c.offs, c.ss, c.perm, gp, gf, plan_fft!(gf, 1:D),
          plan_bfft!(gf, 1:D), c.backend, c.tiles)
end

function check_device(c::GPUCore, xs...)
  for x in xs
    KA.get_backend(x) == c.backend ||
      throw(ArgumentError("the data must be on the device of the plan, got $(typeof(x))"))
  end
end

## kernels

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

# `gr` is the padded grid reinterpreted as real numbers with a leading dimension (re, im): atomic
# additions exist for real numbers only.
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
      c = wt * w1v[i]
      KA.@atomic gr[1, off[1]+i, jt..., l] += real(c)
      KA.@atomic gr[2, off[1]+i, jt..., l] += imag(c)
    end
  end
end

# Adjoint resampling by tiles, without atomics and deterministic: one workgroup per tile of the
# padded grid accumulates in local memory. It takes the nodes of the 2^D bins that can reach the tile
# in chunks (weights, offsets and values into local memory); for each node the work-items split the
# intersection of its support with the tile, and a barrier separates the nodes. The tile is then
# written to the padded grid, which it covers exactly once. D = 2, 3.
# One workgroup per tile and transform.
@kernel function adjoint_tile_kernel!(gp::AbstractArray{Complex{T}}, K, @Const(offs), @Const(ss), @Const(perm),
                                      @Const(vals), @Const(tileptr), nt, B, tstr, bstr, L1,
                                      ::Val{D}, ::Val{W}, ::Val{S}, ::Val{C}) where {T,D,W,S,C}
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
      @inbounds gp[gi..., lb] = complex(bre[i], bim[i])
    end
  end
end

@inline lin2cart(i, str, n) = map((s, m) -> ((i - 1) ÷ s) % m, str, n)

# Support points lo .+ (0:len-1) (of 1:W) of a node with offset `off` inside the tile o .+ (1:B).
@inline function intersection(off, o, B, W)
  lo = max(Int32(1), Int32(o) + Int32(1) - Int32(off))
  hi = min(Int32(W), Int32(o) + Int32(B) - Int32(off))
  lo, max(hi - lo + Int32(1), Int32(0))
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
  gi = map((i, n, ñ) -> mod(i - 1 - n ÷ 2, ñ) + 1, m, N, front(size(gf), Val(D)))
  if adjoint
    @inbounds f[a, m..., b] = c * gf[gi..., l]
  else
    @inbounds gf[gi..., l] = c * f[a, m..., b]
  end
end

# Padded grids from the grids: every padded point reads its periodic partner (interior and halo).
@kernel function pad_kernel!(gp, @Const(gf), h::Int, ::Val{D}) where {D}
  I = Tuple(@index(Global, Cartesian))
  src = map((i, ñ) -> mod(i - 1 - h, ñ) + 1, front(I, Val(D)), front(size(gf), Val(D)))
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

## stages

launch(c, k, args...; ndrange) = prod(ndrange) == 0 || k(c.backend, WORKGROUP)(args...; ndrange)

# The stage functions take f as (L1, N..., L3), values as (L1, J, L3) and grids as (Ñ..., L).
correction!(c::GPUCore{D}, g, f) where {D} =
  (fill!(g, 0); launch(c, correction_kernel!, g, f, c.invphi, c.N, c.L1, Val(D), false; ndrange=(c.N..., nbatch(c))); g)
correction_adjoint!(c::GPUCore{D}, f, g) where {D} =
  (launch(c, correction_kernel!, g, f, c.invphi, c.N, c.L1, Val(D), true; ndrange=(c.N..., nbatch(c))); f)

nbatch(c::GPUCore) = c.L1 * c.L3

function resampling!(c::GPUCore{D}, out, g) where {D}
  launch(c, pad_kernel!, c.gp, g, halo(c.W)[1], Val(D); ndrange=size(c.gp))
  launch(c, resample_kernel!, out, c.gp, c.kernel, c.offs, c.ss, c.perm, c.L1, Val(D);
         ndrange=(length(c.perm), nbatch(c)))
  out
end

function resampling_adjoint!(c::GPUCore{D}, g, vals) where {D}
  tl = c.tiles
  if tl === nothing
    fill!(c.gp, 0)
    gr = reinterpret(reshape, real(eltype(c.gp)), c.gp)
    launch(c, resample_adjoint_kernel!, gr, c.kernel, c.offs, c.ss, c.perm, vals, c.L1, Val(D), Val(c.W);
           ndrange=(length(c.perm), nbatch(c)))
  else
    adjoint_tile_kernel!(c.backend, TILE_WORKGROUP)(c.gp, c.kernel, c.offs, c.ss, c.perm, vals, tl.tileptr,
      tl.nt, tl.B, tl.tstr, tl.bstr, c.L1, Val(D), Val(c.W), Val(prod(tl.B)), Val(TILE_CHUNK);
      ndrange=prod(tl.nt) * nbatch(c) * TILE_WORKGROUP)
  end
  launch(c, unpad_kernel!, g, c.gp, halo(c.W)[1], Val(D); ndrange=size(g))
  g
end

modes(c::GPUCore, f) = reshape(f, c.L1, c.N..., c.L3)
values(c::GPUCore, v) = reshape(v, c.L1, length(c.perm), c.L3)

function NFFT.trafo_core!(out, c::GPUCore, f)
  check_device(c, out, f)
  correction!(c, c.gf, modes(c, f))
  c.pf * c.gf
  resampling!(c, values(c, out), c.gf)
  out
end

function NFFT.adjoint_core!(f, c::GPUCore, vals)
  check_device(c, f, vals)
  resampling_adjoint!(c, c.gf, values(c, vals))
  c.pb * c.gf
  correction_adjoint!(c, modes(c, f), c.gf)
  f
end

# Stage interface: plans without batch dimensions, g of size Ñ.
grids(c::GPUCore, g) = reshape(g, c.Ñ..., 1)
NFFT.correction_stage!(c::GPUCore, g, f) =
  (check_device(c, g, f); correction!(c, grids(c, g), modes(c, f)); g)
NFFT.correction_adjoint_stage!(c::GPUCore, f, g) =
  (check_device(c, f, g); correction_adjoint!(c, modes(c, f), grids(c, g)); f)
NFFT.resampling_stage!(c::GPUCore, fHat, g) =
  (check_device(c, fHat, g); resampling!(c, values(c, fHat), grids(c, g)); fHat)
NFFT.resampling_adjoint_stage!(c::GPUCore, g, fHat) =
  (check_device(c, g, fHat); resampling_adjoint!(c, grids(c, g), values(c, fHat)); g)

end
