# NFFT on GPUs: the stages as KernelAbstractions kernels, the FFT through AbstractFFTs (cuFFT,
# rocFFT, Metal's FFT). The plan is set up on the CPU (parameters, binning, window polynomial); the
# device holds, per node in binned order, the offset into the padded grid and the fractional
# coordinate s ∈ [−1, 1] of every dimension, so that the kernels evaluate the window in the working
# precision T (Metal has no Float64) while the grid position itself was computed in Float64.
module NFFTGPUExt

using NFFT, NFFT.AbstractNFFTs
using NFFT: SymKernel, PolyKernel, Binned, NFFTPlan, halo, gridpos, weights, resample_node,
            kernel_ft, default_tile
using NFFT.FFTW: plan_fft!, plan_bfft!
using GPUArrays: AbstractGPUArray
using KernelAbstractions
const KA = KernelAbstractions

const WORKGROUP = 128

struct GPUCore{D,K,V,O,S,P,G,PF,PB,B}
  N::NTuple{D,Int}
  Ñ::NTuple{D,Int}
  W::Int
  kernel::K
  invphi::NTuple{D,V}        # 1/φ̂ of the kept modes per dimension
  offs::O                    # (D, J) Int32: index of support point 0 in the padded grid, minus 1
  ss::S                      # (D, J) fractional coordinates s = 2t − 1 in T
  perm::P                    # binned order → input index
  gp::G                      # padded grid Ñ .+ W (halo as in geometry.jl)
  gf::G                      # grid Ñ for the FFT
  pf::PF
  pb::PB
  backend::B
end

function NFFT.device_core(storage::Type{<:AbstractGPUArray}, k::Matrix, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, W, φ,
                          ::Type{T}, nthreads) where {D,T}
  K = SymKernel(PolyKernel(φ, W, W + 3; T))
  invphi = ntuple(d -> T.(1 ./ kernel_ft(φ, W, N[d], Ñ[d]; nthreads)), D)
  # binned order: work-items with neighbouring indices touch neighbouring parts of the grid
  b = Binned(k, Ñ, default_tile(D, W), nthreads)
  J = size(k, 2)
  offs = Matrix{Int32}(undef, D, J)
  ss = Matrix{T}(undef, D, J)
  for q in 1:J, d in 1:D
    # as `support` in geometry.jl
    v = gridpos(b.xs[q][d], Ñ[d]) - W / 2
    f = floor(v)
    offs[d, q] = unsafe_trunc(Int, f) + 1 + fld(W, 2)
    ss[d, q] = 2 * T(v - f) - one(T)
  end
  gf = storage(zeros(Complex{T}, Ñ))
  backend = KA.get_backend(gf)
  gp = KA.zeros(backend, Complex{T}, (Ñ .+ W)...)
  GPUCore(N, Ñ, W, K, map(storage, invphi), storage(offs), storage(ss), storage(Int32.(b.perm)),
          gp, gf, plan_fft!(gf), plan_bfft!(gf), backend)
end

# A copy has its own grids and FFT plans and shares the rest.
function Base.copy(c::GPUCore)
  gp = similar(c.gp); gf = similar(c.gf)
  GPUCore(c.N, c.Ñ, c.W, c.kernel, c.invphi, c.offs, c.ss, c.perm, gp, gf, plan_fft!(gf), plan_bfft!(gf), c.backend)
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

@kernel function resample_kernel!(out, @Const(gp), K, @Const(offs), @Const(ss), @Const(perm), ::Val{D}) where {D}
  q = @index(Global)
  ws = node_weights(K, ss, q, Val(D))
  off = node_offset(offs, q, Val(D))
  @inbounds out[perm[q]] = resample_node(gp, off, ws)
end

# `gr` is the padded grid reinterpreted as real numbers with a leading dimension (re, im): atomic
# additions exist for real numbers only.
@kernel function resample_adjoint_kernel!(gr, K, @Const(offs), @Const(ss), @Const(perm), @Const(vals),
                                          ::Val{D}, ::Val{W}) where {D,W}
  q = @index(Global)
  ws = node_weights(K, ss, q, Val(D))
  off = node_offset(offs, q, Val(D))
  v = @inbounds vals[perm[q]]
  w1v = ws[1] .* v
  for It in CartesianIndices(ntuple(_ -> Base.OneTo(W), Val(D - 1)))
    wt = prod(ntuple(d -> ws[d+1][It[d]], Val(D - 1)); init=one(eltype(gr)))
    jt = Base.tail(off) .+ Tuple(It)
    for i in 1:W
      a = wt * w1v[i]
      KA.@atomic gr[1, off[1]+i, jt...] += real(a)
      KA.@atomic gr[2, off[1]+i, jt...] += imag(a)
    end
  end
end

# Resampling correction between the modes f and the grid gf (no halo), one work-item per mode.
@kernel function correction_kernel!(gf, f, @Const(invphi), N, ::Val{D}, adjoint::Bool) where {D}
  I = @index(Global, Cartesian)
  Ñ = size(gf)
  c = one(real(eltype(gf)))
  for d in 1:D
    @inbounds c *= invphi[d][I[d]]
  end
  gi = ntuple(d -> mod(I[d] - 1 - N[d] ÷ 2, Ñ[d]) + 1, Val(D))
  if adjoint
    @inbounds f[I] = c * gf[gi...]
  else
    @inbounds gf[gi...] = c * f[I]
  end
end

# Padded grid from the grid: every padded point reads its periodic partner (interior and halo).
@kernel function pad_kernel!(gp, @Const(gf), h::Int, ::Val{D}) where {D}
  I = @index(Global, Cartesian)
  Ñ = size(gf)
  @inbounds gp[I] = gf[ntuple(d -> mod(I[d] - 1 - h, Ñ[d]) + 1, Val(D))...]
end

# Grid from the padded grid: every point sums its periodic images (folds the halo without atomics).
# The grid has at least W points, so the images are p − Ñ, p, p + Ñ.
@kernel function unpad_kernel!(gf, @Const(gp), h::Int, ::Val{D}) where {D}
  I = @index(Global, Cartesian)
  Ñ = size(gf); P = size(gp)
  acc = zero(eltype(gf))
  for c in CartesianIndices(ntuple(_ -> 3, Val(D)))
    p = ntuple(d -> I[d] + h + (c[d] - 2) * Ñ[d], Val(D))
    if all(map((x, s) -> 1 <= x <= s, p, P))
      @inbounds acc += gp[p...]
    end
  end
  @inbounds gf[I] = acc
end

## stages

launch(c, k, args...; ndrange) = prod(ndrange) == 0 || k(c.backend, WORKGROUP)(args...; ndrange)

correction!(c::GPUCore{D}, g, f) where {D} =
  (fill!(g, 0); launch(c, correction_kernel!, g, f, c.invphi, c.N, Val(D), false; ndrange=c.N); g)
correction_adjoint!(c::GPUCore{D}, f, g) where {D} =
  (launch(c, correction_kernel!, g, f, c.invphi, c.N, Val(D), true; ndrange=c.N); f)

function resampling!(c::GPUCore{D}, out, g) where {D}
  launch(c, pad_kernel!, c.gp, g, halo(c.W)[1], Val(D); ndrange=size(c.gp))
  launch(c, resample_kernel!, out, c.gp, c.kernel, c.offs, c.ss, c.perm, Val(D); ndrange=length(c.perm))
  out
end

function resampling_adjoint!(c::GPUCore{D}, g, vals) where {D}
  fill!(c.gp, 0)
  gr = reinterpret(reshape, real(eltype(c.gp)), c.gp)
  launch(c, resample_adjoint_kernel!, gr, c.kernel, c.offs, c.ss, c.perm, vals, Val(D), Val(c.W);
         ndrange=length(c.perm))
  launch(c, unpad_kernel!, g, c.gp, halo(c.W)[1], Val(D); ndrange=c.Ñ)
  g
end

function NFFT.trafo_core!(out, c::GPUCore, f)
  check_device(c, out, f)
  correction!(c, c.gf, f)
  c.pf * c.gf
  resampling!(c, out, c.gf)
end

function NFFT.adjoint_core!(f, c::GPUCore, vals)
  check_device(c, f, vals)
  resampling_adjoint!(c, c.gf, vals)
  c.pb * c.gf
  correction_adjoint!(c, f, c.gf)
end

NFFT.correction_stage!(c::GPUCore, g, f) = (check_device(c, g, f); correction!(c, g, f))
NFFT.correction_adjoint_stage!(c::GPUCore, f, g) = (check_device(c, f, g); correction_adjoint!(c, f, g))
NFFT.resampling_stage!(c::GPUCore, fHat, g) = (check_device(c, fHat, g); resampling!(c, fHat, g))
NFFT.resampling_adjoint_stage!(c::GPUCore, g, fHat) = (check_device(c, g, fHat); resampling_adjoint!(c, g, fHat))

end
