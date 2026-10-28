# L transforms with the same nodes, batch dimension first: data (L, N…) and (L, J). Every grid point
# carries L contiguous values, so a support row covers W·L contiguous values and window weights,
# support and buffer logic are shared by all L transforms (1D/2D up to 1.6× faster than FINUFFT's
# ntrans or L ducc calls; in 3D the W³ updates per node dominate and the batch kernels lose
# against L single transforms, so 3D batches use the single core).
#
# The batch dimension is processed in chunks of CH = min(L, 4) values held in registers
# (SVector); the grid has Lp = CH·cld(L, CH) rows, the extra rows stay zero. Accumulating into
# memory instead makes every update wait for the previous one (store-to-load forwarding).
struct BatchCore{D,K<:SymKernel,T,Tk,D1,CH,PF<:PassFFT,PB<:PassFFT}
  L::Int
  N::NTuple{D,Int}
  Ñ::NTuple{D,Int}
  W::Int
  B::Int
  kernel::K
  invphi::NTuple{D,Vector{T}}
  binned::Binned{D,Tk}
  gp::Array{Complex{T},D1}          # (Lp, Ñ .+ W)
  fwd::PF                           # on the grid interior, in 1D on F
  bwd::PB
  locks::TileLocks{D}
  bufs::Vector{Array{Complex{T},D1}} # tile buffers (Lp, B+W+1, …)
  # 1D: FFT buffer with the batch dimension last. A 1D FFT along dimension 2 of the batch-first grid
  # (stride Lp) is ≈ 3× slower than on contiguous columns (Ñ = 354294, L = 8: 11.6 vs 3.8 ms).
  F::Union{Nothing,Matrix{Complex{T}}}
  fft::AbstractFFTBackend
  nthreads::Int
end

# Tile buffers hold Lp·(B+W+1)^D values; keep them near 8192 complex values.
batch_tile(D, W, Lp) = D == 1 ? clamp(8192 ÷ Lp - W - 1, 64, 1024) :
                       max(8, floor(Int, (8192 / Lp)^(1 / D)) - W - 1)

function BatchCore(k::AbstractMatrix, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, W, φ, ::Type{T}, fft, L, B, nthreads;
                   kernel=nothing, invphi=nothing, binned=nothing) where {D,T}
  K = kernel === nothing ? make_kernel(φ, W, T) : kernel
  invphi = invphi === nothing ? inv_kernel_ft(φ, W, N, Ñ, T, nthreads) : invphi
  CH = L <= 4 ? L : 4
  Lp = CH * cld(L, CH)
  B = B === nothing ? batch_tile(D, W, Lp) : B
  gp = Array{Complex{T}}(undef, Lp, (Ñ .+ W)...)
  g = interior_b(gp, halo(W))
  if D == 1
    F = Matrix{Complex{T}}(undef, Ñ[1], Lp)
    x, Nf, tdims = F, (N[1], Lp), 1:1
  else
    F = nothing
    x, Nf, tdims = g, (Lp, N...), 2:D+1
  end
  binned = binned === nothing ? Binned(k, Ñ, B, nthreads) : binned
  fwd = PassFFT(fft, x, Nf; dir=:fwd, nthreads, tdims)
  bwd = PassFFT(fft, x, Nf; dir=:bwd, nthreads, tdims)
  BatchCore{D,typeof(K),T,nodetype(binned),D + 1,CH,typeof(fwd),typeof(bwd)}(L, N, Ñ, W, B, K, invphi, binned,
    gp, fwd, bwd, TileLocks(Ñ .+ W, B), task_buffers(Complex{T}, (Lp, ntuple(_ -> B + W + 1, D)...), nthreads),
    F, fft, nthreads)
end

copy_core(c::BatchCore{D,K,T}, nthreads) where {D,K,T} =
  BatchCore(zeros(0, 0), c.N, c.Ñ, c.W, nothing, T, c.fft, c.L, c.B, nthreads;
            kernel=c.kernel, invphi=c.invphi, binned=c.binned)

function correction_b!(g, f, c::BatchCore{D}, adjoint::Bool) where {D}
  N, Ñ, L = c.N, c.Ñ, c.L
  @maybe_threads c.nthreads for r in chunks(1:N[D], c.nthreads)
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
  @maybe_threads nthreads for r in chunks(axes(F, 1), 8nthreads)
    @inbounds for i in r, b in 1:Lp
      toF ? (F[i, b] = g[b, i]) : (g[b, i] = F[i, b])
    end
  end
end

function correction_F!(F, f, c::BatchCore, adjoint::Bool)
  N, Ñ, L = c.N[1], c.Ñ[1], c.L
  @maybe_threads c.nthreads for r in chunks(1:N, c.nthreads)
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
  b, L = c.binned, c.L
  ntile = length(b.tileptr) - 1; next = Threads.Atomic{Int}(1)
  tasks(min(length(c.bufs), ntile)) do it
    buf = c.bufs[it]
    fill!(buf, 0)
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
  b, L = c.binned, c.L
  ntile = length(b.tileptr) - 1; next = Threads.Atomic{Int}(1)
  tasks(min(length(c.bufs), ntile)) do it
    buf = c.bufs[it]
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
  c.gp
end

# out: (L, J), f: (L, N…)
function trafo_batch!(out, c::BatchCore{D}, f) where {D}
  h = halo(c.W); g = interior_b(c.gp, h)
  if D == 1
    threaded_fill!(c.F, 0, c.nthreads)
    correction_F!(c.F, f, c, false)
    apply!(c.fwd, c.F)
    transpose_F!(g, c.F, false, c.nthreads)
  else
    threaded_fill!(c.gp, 0, c.nthreads)
    correction_b!(g, f, c, false)
    apply!(c.fwd, g)
  end
  fill_halo!(c.gp, h, c.nthreads, 2:ndims(c.gp))
  resample_b!(out, c)
end

function adjoint_batch!(f, c::BatchCore{D}, vals) where {D}
  h = halo(c.W); g = interior_b(c.gp, h)
  threaded_fill!(c.gp, 0, c.nthreads)
  resample_adjoint_b!(c, vals)
  fold_halo!(c.gp, h, c.nthreads, 2:ndims(c.gp))
  if D == 1
    transpose_F!(g, c.F, true, c.nthreads)
    apply!(c.bwd, c.F)
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
  @maybe_threads nthreads for r in chunks(1:M, 8nthreads)
    @inbounds for m in r, b in 1:L
      dst[b, m] = src[m, b]
    end
  end
  dst
end
function transpose_lm!(dst, src, nthreads)
  L = size(src, 1); M = size(src, 2)
  @maybe_threads nthreads for r in chunks(1:M, 8nthreads)
    @inbounds for b in 1:L, m in r
      dst[m, b] = src[b, m]
    end
  end
  dst
end
