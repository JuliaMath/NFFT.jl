# L transforms with the same nodes, batch dimension first: data (L, N…) and (L, J). Every grid point
# carries L contiguous values, so a support row covers W·L contiguous values and window weights,
# support and buffer logic are shared by all L transforms (1D/2D up to 1.6× faster than FINUFFT's
# ntrans or L ducc calls; in 3D the W³ updates per node dominate and the batch kernels lose
# against L single transforms, so 3D batches use the single core).
#
# The batch dimension is processed in chunks of CH = min(L, 4) values held in registers
# (SVector); the grid has Lp = CH·cld(L, CH) rows, the extra rows stay zero. Accumulating into
# memory instead made every update wait for the previous one (store-to-load forwarding).
struct BatchCore{D,K<:SymKernel,T,Tk,D1,CH}
  L::Int
  Lp::Int
  N::NTuple{D,Int}
  Ñ::NTuple{D,Int}
  W::Int
  B::Int
  kernel::K
  invphi::NTuple{D,Vector{T}}
  binned::Binned{D,Tk}
  gp::Array{Complex{T},D1}          # (Lp, Ñ .+ W)
  fwd::PassFFT{D1}
  bwd::PassFFT{D1}
  locks::TileLocks{D}
  # 1D: FFT buffer with the batch dimension last. A 1D FFT along dimension 2 of the batch-first grid
  # (stride Lp) is ≈ 3× slower than on contiguous columns (Ñ = 354294, L = 8: 11.6 vs 3.8 ms).
  F::Union{Nothing,Matrix{Complex{T}}}
  fwdF::Any
  bwdF::Any
  nthreads::Int
end

# Tile buffers hold Lp·(B+W+1)^D values; keep them near 8192 complex values.
batch_tile(D, W, Lp) = D == 1 ? clamp(8192 ÷ Lp - W - 1, 64, 1024) :
                       max(8, floor(Int, (8192 / Lp)^(1 / D)) - W - 1)

function BatchCore(k::AbstractMatrix, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, W, φ, ::Type{T}, fft, L, B, nthreads) where {D,T}
  K = SymKernel(PolyKernel(φ, W, W + 3; T))
  invphi = ntuple(d -> T.(1 ./ kernel_ft(φ, W, N[d], Ñ[d]; nthreads)), D)
  CH = L <= 4 ? L : 4
  Lp = CH * cld(L, CH)
  B = B === nothing ? batch_tile(D, W, Lp) : B
  gp = Array{Complex{T}}(undef, Lp, (Ñ .+ W)...)
  g = view(gp, :, map((n, h) -> h[1]+1:h[1]+n, Ñ, ntuple(_ -> halo(W), D))...)
  Nf = (Lp, N...)
  F = fwdF = bwdF = nothing
  if D == 1
    F = Matrix{Complex{T}}(undef, Ñ[1], Lp)
    FFTW.set_num_threads(nthreads)
    fwdF = FFTW.plan_fft!(F, 1; flags=fft.flags); bwdF = FFTW.plan_bfft!(F, 1; flags=fft.flags)
  end
  BatchCore{D,typeof(K),T,eltype(k),D + 1,CH}(L, Lp, N, Ñ, W, B, K, invphi, Binned(k, Ñ, B, nthreads), gp,
    PassFFT(fft, g, Nf; dir=:fwd, nthreads, tdims=2:D+1), PassFFT(fft, g, Nf; dir=:bwd, nthreads, tdims=2:D+1),
    TileLocks(Ñ .+ W, B), F, fwdF, bwdF, nthreads)
end

interior_b(c::BatchCore{D}) where {D} = view(c.gp, :, ntuple(d -> halo(c.W)[1]+1:halo(c.W)[1]+c.Ñ[d], D)...)

# Halo fill/fold over the NFFT dimensions 2…D+1 only.
function halo_b!(gp, h, fold::Bool, nthreads)
  nd = ndims(gp)
  for d in (fold ? (nd:-1:2) : (2:nd))
    Ñ = size(gp, d) - h[1] - h[2]
    e = d == nd ? 2 : nd
    e == d && (e = 1)                                  # 1D NFFT: split along the batch dimension
    Threads.@threads for c in chunks(axes(gp, e), nthreads)
      g = selectdim(gp, e, c)
      lo, hi = selectdim(g, d, 1:h[1]), selectdim(g, d, Ñ+h[1]+1:Ñ+h[1]+h[2])
      if fold
        selectdim(g, d, Ñ+1:Ñ+h[1]) .+= lo
        selectdim(g, d, h[1]+1:h[1]+h[2]) .+= hi
      else
        copyto!(lo, selectdim(g, d, Ñ+1:Ñ+h[1]))
        copyto!(hi, selectdim(g, d, h[1]+1:h[1]+h[2]))
      end
    end
  end
  gp
end

function correction_b!(g, f, c::BatchCore{D}, adjoint::Bool) where {D}
  N, Ñ, L = c.N, c.Ñ, c.L
  Threads.@threads for r in chunks(1:N[D], c.nthreads)
    @inbounds for il in r
      jl = gridindex(il, N[D], Ñ[D]); cl = c.invphi[D][il]
      for I in CartesianIndices(Base.front(N))
        s = cl
        for d in 1:D-1
          s *= c.invphi[d][I[d]]
        end
        gi = (ntuple(d -> gridindex(I[d], N[d], Ñ[d]), D - 1)..., jl)
        if adjoint
          @simd for b in 1:L; f[b, I, il] = s * g[b, gi...]; end
        else
          @simd for b in 1:L; g[b, gi...] = s * f[b, I, il]; end
        end
      end
    end
  end
end

# 1D: batch-last FFT buffer F (Ñ, Lp) ⇄ batch-first grid interior g (Lp, Ñ), blocked and threaded.
function transpose_F!(g, F, toF::Bool, nthreads)
  Lp = size(F, 2)
  Threads.@threads for r in chunks(axes(F, 1), 8nthreads)
    @inbounds for i in r, b in 1:Lp
      toF ? (F[i, b] = g[b, i]) : (g[b, i] = F[i, b])
    end
  end
end

function correction_F!(F, f, c::BatchCore, adjoint::Bool)
  N, Ñ, L = c.N[1], c.Ñ[1], c.L
  Threads.@threads for r in chunks(1:N, c.nthreads)
    @inbounds for b in 1:L, il in r
      s = c.invphi[1][il]; jl = gridindex(il, N, Ñ)
      adjoint ? (f[b, il] = s * F[jl, b]) : (F[jl, b] = s * f[b, il])
    end
  end
end

# One node, one chunk of CH batch values starting after b0.
@inline function resample_chunk(g, b0, off::NTuple{D,Int}, ws::NTuple{D,SVector{W,T}}, ::Val{CH}) where {D,W,T,CH}
  C = eltype(g)
  acc = zero(SVector{CH,C})
  @inbounds for It in CartesianIndices(ntuple(_ -> Base.OneTo(W), Val(D - 1)))
    wt = prod(ntuple(d -> ws[d+1][It[d]], Val(D - 1)); init=one(T))
    jt = Base.tail(off) .+ Tuple(It)
    for i in 1:W
      wi = ws[1][i] * wt
      # @inbounds inside the closure: it does not reach into it from outside
      v = SVector{CH,C}(ntuple(b -> @inbounds(g[b0+b, off[1]+i, jt...]), Val(CH)))
      acc = muladd.(wi, v, acc)
    end
  end
  acc
end

@inline function resample_adjoint_chunk!(g, b0, off::NTuple{D,Int}, ws::NTuple{D,SVector{W,T}}, v::SVector{CH}) where {D,W,T,CH}
  @inbounds for It in CartesianIndices(ntuple(_ -> Base.OneTo(W), Val(D - 1)))
    wt = prod(ntuple(d -> ws[d+1][It[d]], Val(D - 1)); init=one(T))
    jt = Base.tail(off) .+ Tuple(It)
    for i in 1:W
      wi = ws[1][i] * wt
      for b in 1:CH
        g[b0+b, off[1]+i, jt...] = muladd(wi, v[b], g[b0+b, off[1]+i, jt...])
      end
    end
  end
end

function resample_b!(out, c::BatchCore{D,K,T,Tk,D1,CH}) where {D,K,T,Tk,D1,CH}
  b, W, B, L = c.binned, c.W, c.B, c.L
  ntile = length(b.tileptr) - 1; next = Threads.Atomic{Int}(1)
  @sync for _ in 1:min(4c.nthreads, ntile)
    Threads.@spawn begin
      buf = zeros(Complex{T}, c.Lp, ntuple(_ -> B + W + 1, D)...)
      while (i = Threads.atomic_add!(next, 1)) <= ntile
        t = b.order1[i]
        r = b.tileptr[t]:b.tileptr[t+1]-1
        isempty(r) && continue
        o = b.origins[t]
        hi = min.(o .+ size(buf)[2:end], size(c.gp)[2:end])
        copyto!(view(buf, :, map((a, z) -> 1:z-a, o, hi)...), view(c.gp, :, map((a, z) -> a+1:z, o, hi)...))
        @inbounds for q in r
          sw = map(@inline((x, n) -> support(c.kernel, x, n)), b.xs[q], c.Ñ)
          off = map(first, sw) .- o; ws = map(last, sw); col = b.perm[q]
          for b0 in 0:CH:L-1
            acc = resample_chunk(buf, b0, off, ws, Val(CH))
            for j in 1:min(CH, L - b0)
              out[b0+j, col] = acc[j]
            end
          end
        end
      end
    end
  end
  out
end

function flush_b!(gp, buf, o::NTuple{D,Int}, Lk::TileLocks{D}) where {D}
  B = Lk.B; L = size(buf, 1)
  lo = o .+ 1
  hi = min.(lo .+ size(buf)[2:end] .- 1, size(gp)[2:end])
  for I in CartesianIndices(ntuple(d -> fld(lo[d] - 1, B):fld(hi[d] - 1, B), D))
    r = ntuple(d -> max(lo[d], I[d] * B + 1):min(hi[d], (I[d] + 1) * B), D)
    lock(Lk.locks[(Tuple(I) .+ 1)...])
    @inbounds for Jt in CartesianIndices(Base.tail(r))
      jt = Tuple(Jt); bt = jt .- Base.tail(o)
      for j1 in r[1]
        @simd for b in 1:L
          gp[b, j1, jt...] += buf[b, j1 - o[1], bt...]
        end
      end
    end
    unlock(Lk.locks[(Tuple(I) .+ 1)...])
  end
end

function resample_adjoint_b!(c::BatchCore{D,K,T,Tk,D1,CH}, vals) where {D,K,T,Tk,D1,CH}
  b, W, B, L = c.binned, c.W, c.B, c.L
  ntile = length(b.tileptr) - 1; next = Threads.Atomic{Int}(1)
  @sync for _ in 1:min(4c.nthreads, ntile)
    Threads.@spawn begin
      buf = zeros(Complex{T}, c.Lp, ntuple(_ -> B + W + 1, D)...)
      while (t = Threads.atomic_add!(next, 1)) <= ntile
        r = b.tileptr[t]:b.tileptr[t+1]-1
        isempty(r) && continue
        o = b.origins[t]
        fill!(buf, 0)
        @inbounds for q in r
          sw = map(@inline((x, n) -> support(c.kernel, x, n)), b.xs[q], c.Ñ)
          off = map(first, sw) .- o; ws = map(last, sw); col = b.perm[q]
          for b0 in 0:CH:L-1
            v = SVector{CH,Complex{T}}(ntuple(j -> b0 + j <= L ? vals[b0+j, col] : zero(Complex{T}), Val(CH)))
            resample_adjoint_chunk!(buf, b0, off, ws, v)
          end
        end
        flush_b!(c.gp, buf, o, c.locks)
      end
    end
  end
  c.gp
end

# out: (L, J), f: (L, N…)
function trafo_batch!(out, c::BatchCore{D}, f) where {D}
  h = halo(c.W); g = interior_b(c)
  if D == 1
    threaded_fill!(c.F, 0, c.nthreads)
    correction_F!(c.F, f, c, false)
    c.fwdF * c.F
    transpose_F!(g, c.F, false, c.nthreads)
  else
    threaded_fill!(c.gp, 0, c.nthreads)
    correction_b!(g, f, c, false)
    apply!(c.fwd, g)
  end
  halo_b!(c.gp, h, false, c.nthreads)
  resample_b!(out, c)
end

function adjoint_batch!(f, c::BatchCore{D}, vals) where {D}
  h = halo(c.W); g = interior_b(c)
  threaded_fill!(c.gp, 0, c.nthreads)
  resample_adjoint_b!(c, vals)
  halo_b!(c.gp, h, true, c.nthreads)
  if D == 1
    transpose_F!(g, c.F, true, c.nthreads)
    c.bwdF * c.F
    correction_F!(c.F, f, c, true)
  else
    apply!(c.bwd, g)
    correction_b!(g, f, c, true)
  end
  f
end

# Batch dimension last, data (M, L) ⇄ batch first (L, M), threaded over M.
function transpose_ml!(dst, src, nthreads)
  M = size(src, 1); L = size(src, 2)
  Threads.@threads for r in chunks(1:M, 8nthreads)
    @inbounds for m in r, b in 1:L
      dst[b, m] = src[m, b]
    end
  end
  dst
end
function transpose_lm!(dst, src, nthreads)
  L = size(src, 1); M = size(src, 2)
  Threads.@threads for r in chunks(1:M, 8nthreads)
    @inbounds for b in 1:L, m in r
      dst[m, b] = src[b, m]
    end
  end
  dst
end
