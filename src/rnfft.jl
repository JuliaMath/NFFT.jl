# NFFT with real values at the nodes (RNFFT): f̂_j = Re Σ_{n₁ ≥ 0} h_n exp(−2πi n·k_j) and its adjoint
# h_n = Σ_j y_j exp(2πi n·k_j), for the half n₁ = 0…N₁÷2 of the coefficients. The oversampled grid
# g_l = Re Σ_n h_n/φ̂(n) exp(−2πi n·l/Ñ) is real: the FFT stage is complex FFTs along dimensions 2…D on
# the half spectrum (Ñ₁÷2+1, Ñ₂, …) followed by a c2r FFT along dimension 1 straight into the padded
# real grid, and the resampling loops of the NFFT run on that real grid.
#
# The c2r FFT computes Re Q(0) + 2 Re Σ_{n₁≥1} Q(n₁) exp(2πi n₁ l₁/Ñ₁) with the positive sign, so the
# correction stores conj(h_n)/φ̂(n), halved for n₁ ≥ 1 (n₁ ≤ N₁÷2 < Ñ₁/2, the grid's Nyquist bin
# stays empty). The adjoint is the r2c FFT (negative sign) and forward FFTs, conjugated at the end.

struct RNFFTCore{D,K<:SymKernel,T,Tk,PF<:PassFFT,PB<:PassFFT,PC,PR}
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
  pfwd::PF                             # direct: backward FFTs along 2…D, zero input skipped
  pbwd::PB                             # adjoint: forward FFTs along 2…D, only kept outputs
  boxes::Vector{UnitRange{Int}}        # chunks along dimension D for the c2r/r2c FFTs
  c2r::Vector{PC}                      # FFTW plans per box
  r2c::Vector{PR}
  locks::TileLocks{D}
  bufs::Vector{Array{T,D}}
  flags::UInt32
  nthreads::Int
end

# Correction factors: dimension 1 for the modes n₁ = 0 … N₁÷2, the others as in the NFFT.
function rnfft_invphi(φ, W, N::NTuple{D,Int}, Ñ, ::Type{T}, nthreads) where {D,T}
  Nh = N[1] ÷ 2 + 1
  phi1 = kernel_ft(φ, W, 2Nh, Ñ[1]; nthreads)[Nh+1:2Nh]          # modes −Nh…Nh−1, keep 0…Nh−1
  ntuple(d -> d == 1 ? T.(1 ./ phi1) : T.(1 ./ kernel_ft(φ, W, N[d], Ñ[d]; nthreads)), D)
end

# The c2r/r2c transforms along dimension 1 are FFTW plans: the RNFFT needs the FFTW backend.
function RNFFTCore(k::AbstractMatrix, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, W, φ, ::Type{T}, fft::FFTWBackend, B, nthreads;
                   binned=nothing, kernel=nothing, invphi=nothing) where {D,T}
  K = kernel === nothing ? make_kernel(φ, W, T) : kernel
  Nh = N[1] ÷ 2 + 1
  invphi = invphi === nothing ? rnfft_invphi(φ, W, N, Ñ, T, nthreads) : invphi
  B = B === nothing ? default_tile(D, W) : B
  binned = binned === nothing ? Binned(k, Ñ, B, nthreads) : binned
  gp = zeros(T, Ñ .+ W)
  g = interior(gp, halo(W))
  H = zeros(Complex{T}, Ñ[1] ÷ 2 + 1, Base.tail(Ñ)...)
  Hk = view(H, 1:Nh, ntuple(_ -> Colon(), D - 1)...)
  Nk = (Nh, Base.tail(N)...)
  pfwd = PassFFT(fft, Hk, Nk, Val(:bwd); pattern=:fwd, nthreads, tdims=2:D, fft_threads=false)
  pbwd = PassFFT(fft, Hk, Nk, Val(:fwd); pattern=:bwd, nthreads, tdims=2:D, fft_threads=false)
  boxes = D == 1 ? [1:1] : chunks(1:Ñ[D], 4nthreads)
  c2r, r2c = with_fftw_threads(D == 1 ? nthreads : 1) do
    ([FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,false,D}(box(H, b), box(g, b), (1,), fft.flags, FFTW.NO_TIMELIMIT)
      for b in boxes],
     [FFTW.rFFTWPlan{T,FFTW.FORWARD,false,D}(box(g, b), box(H, b), (1,), fft.flags, FFTW.NO_TIMELIMIT) for b in boxes])
  end
  RNFFTCore{D,typeof(K),T,nodetype(binned),typeof(pfwd),typeof(pbwd),eltype(c2r),eltype(r2c)}(N, Nh, Ñ, W, B, K, invphi, binned, gp, H, pfwd, pbwd, boxes, c2r, r2c,
    TileLocks(size(gp), B), task_buffers(T, ntuple(_ -> B + W + 1, D), nthreads), fft.flags, nthreads)
end

copy_core(c::RNFFTCore{D,K,T}, nthreads) where {D,K,T} =
  RNFFTCore(zeros(0, 0), c.N, c.Ñ, c.W, nothing, T, FFTWBackend(flags=c.flags), c.B, nthreads;
            binned=c.binned, kernel=c.kernel, invphi=c.invphi)

slice_size(c::RNFFTCore) = (c.Nh, Base.tail(c.N)...)

# the part of an array in a chunk of its last dimension (all of it in 1D)
box(x::AbstractArray{<:Any,1}, r) = x
box(x::AbstractArray{<:Any,D}, r) where {D} = view(x, ntuple(_ -> Colon(), D - 1)..., r)

half_view(c::RNFFTCore{D}) where {D} = view(c.H, 1:c.Nh, ntuple(_ -> Colon(), D - 1)...)

# the c2r FFTs (direct) or the r2c FFTs (adjoint) along dimension 1, per chunk of dimension D
function c2r_boxes!(c::RNFFTCore, ::Val{forward}) where {forward}
  g = interior(c.gp, halo(c.W))
  @maybe_threads c.nthreads :dynamic for i in eachindex(c.boxes)
    X = box(c.H, c.boxes[i]); Y = box(g, c.boxes[i])
    forward ? mul!(Y, c.c2r[i], X) : mul!(X, c.r2c[i], Y)
  end
end

# h (Nh, N₂, …) ↔ half spectrum; mode n₁ = i₁ − 1 sits at H[i₁, …], the other dimensions as in the NFFT.
# The transforms store conj(h)/φ̂ halved for n₁ ≥ 1 (see above); the stage interface stores h/φ̂
# (`half = 1`, `cj = identity`).
function rnfft_correction!(H, h, c::RNFFTCore{D,K,T}, half=T(0.5), cj=conj) where {D,K,T}
  N, Ñ, ip = c.N, c.Ñ, c.invphi
  @maybe_threads c.nthreads for r in chunks(1:size(h, D), c.nthreads)
    @inbounds for I in CartesianIndices((ntuple(d -> 1:size(h, d), D - 1)..., r))
      a = ip[1][I[1]] * (I[1] == 1 ? one(T) : half)
      for d in 2:D
        a *= ip[d][I[d]]
      end
      H[I[1], ntuple(d -> gridindex(I[d+1], N[d+1], Ñ[d+1]), D - 1)...] = a * cj(h[I])
    end
  end
  H
end
function rnfft_correction_adjoint!(h, H, c::RNFFTCore{D,K,T}, cj=conj) where {D,K,T}
  N, Ñ, ip = c.N, c.Ñ, c.invphi
  @maybe_threads c.nthreads for r in chunks(1:size(h, D), c.nthreads)
    @inbounds for I in CartesianIndices((ntuple(d -> 1:size(h, d), D - 1)..., r))
      a = ip[1][I[1]]
      for d in 2:D
        a *= ip[d][I[d]]
      end
      h[I] = a * cj(H[I[1], ntuple(d -> gridindex(I[d+1], N[d+1], Ñ[d+1]), D - 1)...])
    end
  end
  h
end

function trafo_core!(out, c::RNFFTCore{D}, h) where {D}
  threaded_fill!(c.H, 0, c.nthreads)
  rnfft_correction!(c.H, h, c)
  D > 1 && apply!(c.pfwd, half_view(c))
  c2r_boxes!(c, Val(true))
  resample_interior!(c, out)
  out
end

function adjoint_core!(h, c::RNFFTCore{D}, vals) where {D}
  resample_adjoint_interior!(c, vals)
  c2r_boxes!(c, Val(false))
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
  @plan_fields
end

function AbstractNFFTs.plan_rnfft(b::NFFTBackend, k::AbstractMatrix, N::Dims{D}; kwargs...) where {D}
  g, rest = generic_options(k, N; kwargs...)
  check_options(b, rest, NFFT_OPTIONS)
  fast_plan(RNFFTPlan, k, N, g, out_ndims(N, kwargs); rest...)
end

RNFFTPlan(k::AbstractVecOrMat, N; kwargs...) = plan_rnfft(NFFTBackend(), k, N; kwargs...)
RNFFTPlan(k::AbstractMatrix, N::NTuple{D,Int}, g::GenericOptions; kwargs...) where {D} =
  fast_plan(RNFFTPlan, k, N, g; kwargs...)

input_size(::Type{<:RNFFTPlan}, N, dims) = Base.setindex(N, N[first(dims)] ÷ 2 + 1, first(dims))

function make_core(::Type{<:RNFFTPlan}, k, Nd, kernel::KernelEntry, ::Type{T}, fft, tilesize, nthreads, storage,
                   L1, L3) where {T}
  φ = window_function(kernel)
  Ñ = grid_size(kernel.σ, Nd, kernel.W)
  storage <: Array || return device_rnfft_core(storage, k, Nd, Ñ, kernel.W, φ, T, nthreads, L1, L3)
  RNFFTCore(k, Nd, Ñ, kernel.W, φ, T, fft, tilesize, nthreads)
end

# Core for data on a device (GPU array types; the GPU extension), handling all batch dimensions.
device_rnfft_core(storage, args...) = device_core(storage)

# Stage interface (plans without batch dimensions, on the CPU). The resampling correction writes the
# half spectrum ĝ_n = h_n/φ̂(n), n₁ = 0…Ñ₁÷2 (size spectrum_size(p), zero outside the modes); the FFT
# stage computes the real array g_l = Re Σ_n ĝ_n exp(−2πi n·l/Ñ), summed over that half, and its
# transpose with respect to the real inner product, ĝ_n = Σ_l g_l exp(2πi n·l/Ñ). Neither changes in
# place: ĝ and g differ in size and type.
function check_rnfft_stage(p, f, ĝ, g, fHat)
  c = full_core(p)
  c isa RNFFTCore || throw(ArgumentError("the stage interface of RNFFT plans needs a plan on the CPU"))
  f === nothing || check_size(f, p.Nin, "the equidistant array")
  ĝ === nothing || check_size(ĝ, size(c.H), "the oversampled spectrum")
  g === nothing || check_size(g, c.Ñ, "the oversampled array")
  fHat === nothing || check_size(fHat, p.NOut, "the array at the nodes")
  c
end

AbstractNFFTs.oversampled_size(p::RNFFTPlan) = check_rnfft_stage(p, nothing, nothing, nothing, nothing).Ñ
AbstractNFFTs.spectrum_size(p::RNFFTPlan) = size(check_rnfft_stage(p, nothing, nothing, nothing, nothing).H)

function AbstractNFFTs.resampling_correction!(p::RNFFTPlan{T}, f::AbstractArray, ĝ::AbstractArray) where {T}
  c = check_rnfft_stage(p, f, ĝ, nothing, nothing)
  fill!(ĝ, 0)
  rnfft_correction!(ĝ, f, c, one(T), identity)
end
AbstractNFFTs.resampling_correction_adjoint!(p::RNFFTPlan, ĝ::AbstractArray, f::AbstractArray) =
  rnfft_correction_adjoint!(f, ĝ, check_rnfft_stage(p, f, ĝ, nothing, nothing), identity)

# FFTW's c2r transform has the positive sign and counts the modes 0 < n₁ < Ñ₁/2 twice: it gets the
# conjugate, halved there.
function AbstractNFFTs.oversampled_fft!(p::RNFFTPlan{T}, ĝ::AbstractArray, g::AbstractArray) where {T}
  c = check_rnfft_stage(p, nothing, ĝ, g, nothing)
  X = conj.(ĝ)
  selectdim(X, 1, 2:cld(c.Ñ[1], 2)) .*= T(0.5)
  copyto!(g, FFTW.brfft(X, c.Ñ[1]))
end
function AbstractNFFTs.oversampled_bfft!(p::RNFFTPlan, g::AbstractArray, ĝ::AbstractArray)
  check_rnfft_stage(p, nothing, ĝ, g, nothing)
  ĝ .= conj.(FFTW.rfft(g))
end
AbstractNFFTs.oversampled_fft!(::RNFFTPlan, ::AbstractArray) =
  throw(ArgumentError("the FFT stages of an RNFFT plan only exist out of place: oversampled_fft!(p, ĝ, g)"))
AbstractNFFTs.oversampled_bfft!(::RNFFTPlan, ::AbstractArray) =
  throw(ArgumentError("the FFT stages of an RNFFT plan only exist out of place: oversampled_bfft!(p, g, ĝ)"))

AbstractNFFTs.resampling!(p::RNFFTPlan, g::AbstractArray, fHat::AbstractArray) =
  (resampling_stage!(check_rnfft_stage(p, nothing, nothing, g, fHat), vec(fHat), g); fHat)
AbstractNFFTs.resampling_adjoint!(p::RNFFTPlan, fHat::AbstractArray, g::AbstractArray) =
  (resampling_adjoint_stage!(check_rnfft_stage(p, nothing, nothing, g, fHat), g, vec(fHat)); g)
