# NFFT with real values at the nodes (RNFFT): f̂_j = Re Σ_{n₁ ≥ 0} h_n exp(−2πi n·k_j) and its adjoint
# h_n = Σ_j y_j exp(2πi n·k_j), for the half n₁ = 0…N₁÷2 of the coefficients. The oversampled grid
# g_l = Re Σ_n h_n/φ̂(n) exp(−2πi n·l/Ñ) is real: the FFT stage is complex FFTs along dimensions 2…D on
# the half spectrum (Ñ₁÷2+1, Ñ₂, …) followed by a c2r FFT along dimension 1 straight into the padded
# real grid, and the resampling loops of the NFFT run on that real grid.
#
# The c2r FFT computes Re Q(0) + 2 Re Σ_{n₁≥1} Q(n₁) exp(2πi n₁ l₁/Ñ₁) with the positive sign, so the
# correction stores conj(h_n)/φ̂(n), halved for n₁ ≥ 1 (n₁ ≤ N₁÷2 < Ñ₁/2, the grid's Nyquist bin
# stays empty). The adjoint is the r2c FFT (negative sign) and forward FFTs, conjugated at the end.

struct RNFFTCore{D,K<:SymKernel,T,Tk}
  N::NTuple{D,Int}                     # logical size
  Nh::Int                              # modes n₁ = 0…Nh−1 = N₁÷2
  Ñ::NTuple{D,Int}
  W::Int
  B::Int
  kernel::K
  invphi::NTuple{D,Vector{T}}          # dimension 1: n₁ = 0…Nh−1; others as in the NFFT
  binned::Binned{D,Tk}
  gp::Array{T,D}                       # padded real grid, Ñ .+ W
  H::Array{Complex{T},D}               # half spectrum, (Ñ₁÷2+1, Ñ₂, …)
  pfwd::PassFFT{D}                     # direct: backward FFTs along 2…D, zero input skipped
  pbwd::PassFFT{D}                     # adjoint: forward FFTs along 2…D, only kept outputs
  boxes::Vector{UnitRange{Int}}        # chunks along dimension D for the c2r/r2c FFTs
  c2r::Vector{Any}
  r2c::Vector{Any}
  locks::TileLocks{D}
  bufs::Vector{Array{T,D}}
  flags::UInt32
  nthreads::Int
end

function RNFFTCore(k::AbstractMatrix, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, W, φ, ::Type{T}, fft, B, nthreads;
                   binned=nothing, kernel=nothing, invphi=nothing) where {D,T}
  K = kernel === nothing ? SymKernel(PolyKernel(φ, W, W + 3; T)) : kernel
  Nh = N[1] ÷ 2 + 1
  if invphi === nothing
    phi1 = kernel_ft(φ, W, 2Nh, Ñ[1]; nthreads)[Nh+1:2Nh]          # modes −Nh…Nh−1, keep 0…Nh−1
    invphi = ntuple(d -> d == 1 ? T.(1 ./ phi1) : T.(1 ./ kernel_ft(φ, W, N[d], Ñ[d]; nthreads)), D)
  end
  B = B === nothing ? default_tile(D, W) : B
  binned = binned === nothing ? Binned(k, Ñ, B, nthreads) : binned
  gp = zeros(T, Ñ .+ W)
  g = interior(gp, halo(W))
  H = zeros(Complex{T}, Ñ[1] ÷ 2 + 1, Base.tail(Ñ)...)
  Hk = view(H, 1:Nh, ntuple(_ -> Colon(), D - 1)...)
  Nk = (Nh, Base.tail(N)...)
  pfwd = PassFFT(fft, Hk, Nk; dir=:bwd, pattern=:fwd, nthreads, tdims=2:D, fftw_threads=false)
  pbwd = PassFFT(fft, Hk, Nk; dir=:fwd, pattern=:bwd, nthreads, tdims=2:D, fftw_threads=false)
  boxes = D == 1 ? [1:1] : chunks(1:Ñ[D], 4nthreads)
  FFTW.set_num_threads(D == 1 ? nthreads : 1)
  c2r = Any[]; r2c = Any[]
  for b in boxes
    X = box(H, b); Y = box(g, b)
    push!(c2r, FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,false,D}(X, Y, (1,), fft.flags, FFTW.NO_TIMELIMIT))
    push!(r2c, FFTW.rFFTWPlan{T,FFTW.FORWARD,false,D}(Y, X, (1,), fft.flags, FFTW.NO_TIMELIMIT))
  end
  RNFFTCore{D,typeof(K),T,nodetype(binned)}(N, Nh, Ñ, W, B, K, invphi, binned, gp, H, pfwd, pbwd, boxes, c2r, r2c,
    TileLocks(size(gp), B), task_buffers(T, ntuple(_ -> B + W + 1, D), nthreads), fft.flags, nthreads)
end

function Base.copy(c::RNFFTCore{D,K,T}) where {D,K,T}
  RNFFTCore(zeros(0, 0), c.N, c.Ñ, c.W, nothing, T, FFTWBackend(flags=c.flags), c.B, c.nthreads;
            binned=c.binned, kernel=c.kernel, invphi=c.invphi)
end

nodetype(::Binned{D,T}) where {D,T} = T

# the part of an array in a chunk of its last dimension (all of it in 1D)
box(x::AbstractArray{<:Any,1}, r) = x
box(x::AbstractArray{<:Any,D}, r) where {D} = view(x, ntuple(_ -> Colon(), D - 1)..., r)

half_view(c::RNFFTCore{D}) where {D} = view(c.H, 1:c.Nh, ntuple(_ -> Colon(), D - 1)...)

function c2r_boxes!(c::RNFFTCore{D}, plans, forward::Bool) where {D}
  g = interior(c.gp, halo(c.W))
  @maybe_threads c.nthreads :dynamic for i in eachindex(c.boxes)
    X = box(c.H, c.boxes[i]); Y = box(g, c.boxes[i])
    forward ? mul!(Y, plans[i], X) : mul!(X, plans[i], Y)
  end
end

# h (Nh, N₂, …) ↔ half spectrum; mode n₁ = i₁ − 1 sits at H[i₁, …], the other dimensions as in the NFFT
function rnfft_correction!(H, h, c::RNFFTCore{D,K,T}) where {D,K,T}
  N, Ñ, ip = c.N, c.Ñ, c.invphi
  @maybe_threads c.nthreads for r in chunks(1:size(h, D), c.nthreads)
    @inbounds for I in CartesianIndices((ntuple(d -> 1:size(h, d), D - 1)..., r))
      a = ip[1][I[1]] * (I[1] == 1 ? one(T) : T(0.5))
      for d in 2:D
        a *= ip[d][I[d]]
      end
      H[I[1], ntuple(d -> gridindex(I[d+1], N[d+1], Ñ[d+1]), D - 1)...] = a * conj(h[I])
    end
  end
  H
end
function rnfft_correction_adjoint!(h, H, c::RNFFTCore{D,K,T}) where {D,K,T}
  N, Ñ, ip = c.N, c.Ñ, c.invphi
  @maybe_threads c.nthreads for r in chunks(1:size(h, D), c.nthreads)
    @inbounds for I in CartesianIndices((ntuple(d -> 1:size(h, d), D - 1)..., r))
      a = ip[1][I[1]]
      for d in 2:D
        a *= ip[d][I[d]]
      end
      h[I] = a * conj(H[I[1], ntuple(d -> gridindex(I[d+1], N[d+1], Ñ[d+1]), D - 1)...])
    end
  end
  h
end

function trafo_core!(out, c::RNFFTCore{D}, h) where {D}
  threaded_fill!(c.H, 0, c.nthreads)
  rnfft_correction!(c.H, h, c)
  D > 1 && apply!(c.pfwd, half_view(c))
  c2r_boxes!(c, c.c2r, true)
  fill_halo!(c.gp, halo(c.W), c.nthreads)
  resample!(out, c.gp, c.kernel, c.binned, c.Ñ, c.bufs)
  out
end

function adjoint_core!(h, c::RNFFTCore{D}, vals) where {D}
  threaded_fill!(c.gp, 0, c.nthreads)
  resample_adjoint!(c.gp, c.kernel, c.binned, vals, c.Ñ, c.locks, c.bufs)
  fold_halo!(c.gp, halo(c.W), c.nthreads)
  c2r_boxes!(c, c.r2c, false)
  D > 1 && apply!(c.pbwd, half_view(c))
  rnfft_correction_adjoint!(h, c.H, c)
  h
end

"""
    RNFFTPlan{T,D,R}

RNFFT plan of NFFT.jl (real values at the nodes), created by `plan_rnfft(k, N; kwargs...)`, see
[`plan_rnfft`](@ref). Options as for [`NFFTPlan`](@ref); batch dimensions loop over the slices.
"""
mutable struct RNFFTPlan{T,D,R} <: AbstractRNFFTPlan{T,D,R}
  N::NTuple{D,Int}                     # logical size
  Nin::NTuple{D,Int}                   # size of the half array
  NOut::NTuple{R,Int}
  dims::UnitRange{Int}
  k::AbstractMatrix
  kernel::KernelEntry
  accuracy::Float64
  options::GenericOptions{T}
  fft::AbstractFFTBackend
  tilesize::Union{Nothing,Int}
  core::Any
  slicecores::Vector{Any}              # single-threaded cores, one per task, for many slices
end

function AbstractNFFTs.plan_rnfft(b::NFFTBackend, k::AbstractMatrix, N::Dims{D}; kwargs...) where {D}
  g, rest = generic_options(k, N; kwargs...)
  check_options(b, rest, NFFT_OPTIONS)
  g.storage <: Array || throw(ArgumentError("storage = $(g.storage) is not supported yet"))
  RNFFTPlan(k, N, g; rest...)
end

RNFFTPlan(k::AbstractVecOrMat, N; kwargs...) = plan_rnfft(NFFTBackend(), k, N; kwargs...)

function RNFFTPlan(k::AbstractMatrix, N::NTuple{D,Int}, g::GenericOptions{T}; window=:auto, W=nothing,
                   σ=nothing, fft=FFTWBackend(), tilesize=nothing) where {D,T}
  dims = g.dims
  J = size(k, 2)
  if window isa CustomWindow
    (W === nothing || σ === nothing) && throw(ArgumentError("a CustomWindow needs W and σ"))
    kernel, acc = KernelEntry(window, W, σ, NaN, NaN, NaN, NaN), NaN
  elseif window === :auto || window isa AbstractWindow
    kernel, acc = choose_kernel(g.reltol, N[dims], J, T; window, W, σ)
  else
    throw(ArgumentError("window must be :auto or an AbstractWindow, got $window"))
  end
  d1 = first(dims)
  Nin = Base.setindex(N, N[d1] ÷ 2 + 1, d1)
  NOut = (N[1:d1-1]..., J, N[last(dims)+1:end]...)
  core = rnfft_core(k, N[dims], kernel, T, fft, tilesize, g.nthreads)
  RNFFTPlan{T,D,length(NOut)}(N, Nin, NOut, dims, k, kernel, acc, g, fft, tilesize, core,
                              slice_cores(core, N[dims], J, prod(batch_sizes(N, dims))))
end

# Many or small slices run in parallel on single-threaded cores (`use_slices`); a small transform
# cannot use many threads (OCT, 512 A-scans of 2048 samples: 100 ms one after the other, 2 ms in
# parallel).
slice_cores(c::RNFFTCore, Nd, J, L) = use_slices(L, J + prod(Nd), c.nthreads) ? single_cores(c, L) : Any[]

single_core(c::RNFFTCore{D,K,T}) where {D,K,T} =
  RNFFTCore(zeros(0, 0), c.N, c.Ñ, c.W, nothing, T, FFTWBackend(flags=c.flags), c.B, 1;
            binned=c.binned, kernel=c.kernel, invphi=c.invphi)

function rnfft_core(k, Nd, kernel::KernelEntry, ::Type{T}, fft, tilesize, nthreads) where {T}
  check_nodes(k)
  φ = window_function(kernel.window, kernel.W, kernel.β, kernel.e0)
  RNFFTCore(k, Nd, grid_size(kernel.σ, Nd, kernel.W), kernel.W, φ, T, fft, tilesize, nthreads)
end

AbstractNFFTs.size_in(p::RNFFTPlan) = p.Nin
AbstractNFFTs.size_out(p::RNFFTPlan) = p.NOut
AbstractNFFTs.nodes(p::RNFFTPlan) = p.k
AbstractNFFTs.accuracy(p::RNFFTPlan) = p.accuracy
parameters(p::RNFFTPlan) = (window=p.kernel.window, W=p.kernel.W, σ=p.kernel.σ)

function Base.copy(p::RNFFTPlan{T,D,R}) where {T,D,R}
  core = copy(p.core)
  RNFFTPlan{T,D,R}(p.N, p.Nin, p.NOut, p.dims, p.k, p.kernel, p.accuracy, p.options, p.fft, p.tilesize, core,
                   slice_cores(core, p.N[p.dims], size(p.k, 2), prod(batch_sizes(p.N, p.dims))))
end

function AbstractNFFTs.nodes!(p::RNFFTPlan{T}, k::AbstractMatrix) where {T}
  size(k, 1) == length(p.dims) || throw(DimensionMismatch("nodes must have $(length(p.dims)) rows"))
  p.core = rnfft_core(k, p.N[p.dims], p.kernel, T, p.fft, p.tilesize, p.options.nthreads)
  p.slicecores = slice_cores(p.core, p.N[p.dims], size(k, 2), prod(batch_sizes(p.N, p.dims)))
  p.k = k
  p.NOut = (p.N[1:first(p.dims)-1]..., size(k, 2), p.N[last(p.dims)+1:end]...)
  p
end

function Base.show(io::IO, p::RNFFTPlan{T}) where {T}
  e = p.kernel
  print(io, "RNFFTPlan{$T} for a logical size $(p.N) along dims $(p.dims) at $(size(p.k, 2)) nodes: ",
        "$(nameof(typeof(e.window))) window, W = $(e.W), σ = $(e.σ), predicted error $(p.accuracy)")
end

function rnfft_views(p::RNFFTPlan, h, fHat)
  size(h) == p.Nin || throw(DimensionMismatch("the half array has size $(size(h)), the plan expects $(p.Nin)"))
  size(fHat) == p.NOut || throw(DimensionMismatch("the array at the nodes has size $(size(fHat)), the plan expects $(p.NOut)"))
  L1, L3 = batch_sizes(p.N, p.dims)
  reshape(h, L1, p.Nin[p.dims]..., L3), reshape(fHat, L1, size(p.k, 2), L3), L1, L3, length(p.dims)
end

function LinearAlgebra.mul!(fHat::AbstractArray, p::RNFFTPlan, h::AbstractArray)
  hr, gr, L1, L3, Dd = rnfft_views(p, h, fHat)
  if isempty(p.slicecores)
    rnfft_loop!(p.core, gr, hr, L1, L3, Dd)
  else
    slices_parallel(p.slicecores, L1, L3) do c, a, b
      trafo_core!(view(gr, a, :, b), c, view(hr, a, ntuple(_ -> Colon(), Dd)..., b))
    end
  end
  fHat
end

function LinearAlgebra.mul!(h::AbstractArray, pl::Adjoint{Complex{T},<:RNFFTPlan{T}}, fHat::AbstractArray) where {T}
  p = pl.parent
  hr, gr, L1, L3, Dd = rnfft_views(p, h, fHat)
  if isempty(p.slicecores)
    rnfft_adjoint_loop!(p.core, hr, gr, L1, L3, Dd)
  else
    slices_parallel(p.slicecores, L1, L3) do c, a, b
      adjoint_core!(view(hr, a, ntuple(_ -> Colon(), Dd)..., b), c, view(gr, a, :, b))
    end
  end
  h
end

# function barrier into code specialised on the kernel; batch dimensions loop over the slices
function rnfft_loop!(c, gr, hr, L1, L3, Dd)
  for b in 1:L3, a in 1:L1
    trafo_core!(view(gr, a, :, b), c, view(hr, a, ntuple(_ -> Colon(), Dd)..., b))
  end
end
function rnfft_adjoint_loop!(c, hr, gr, L1, L3, Dd)
  for b in 1:L3, a in 1:L1
    adjoint_core!(view(hr, a, ntuple(_ -> Colon(), Dd)..., b), c, view(gr, a, :, b))
  end
end
