# The GPU code path on the CPU: JLArrays is the reference implementation of a GPU array
# (KernelAbstractions' CPU backend), so that every CI run covers the GPU kernels.
using JLArrays, LinearAlgebra, NFFT
using NFFT.FFTW
const AbstractFFTs = FFTW.AbstractFFTs

# JLArrays has no FFTs: plans that run FFTW on a host copy. They accept only the regions that
# rocFFT (AMDGPU.jl) accepts, ranges that start at dimension 1 or end at the last one.
struct HostPlan{P}
  p::P
end
Base.:*(h::HostPlan, x::JLArray) = copyto!(x, h.p * Array(x))
LinearAlgebra.mul!(y::JLArray, h::HostPlan, x::JLArray) = copyto!(y, h.p * Array(x))
function rocfft_region(x, region)
  first(region) == 1 || last(region) == ndims(x) ||
    throw(ArgumentError("rocFFT plans no FFTs along $region of a $(ndims(x))-dimensional array"))
  region
end
AbstractFFTs.plan_fft!(x::JLArray, region; kwargs...) = HostPlan(plan_fft(Array(x), rocfft_region(x, region)))
AbstractFFTs.plan_bfft!(x::JLArray, region; kwargs...) = HostPlan(plan_bfft(Array(x), rocfft_region(x, region)))
AbstractFFTs.plan_rfft(x::JLArray, region; kwargs...) = HostPlan(plan_rfft(Array(x), rocfft_region(x, region)))
AbstractFFTs.plan_brfft(x::JLArray, d::Integer, region; kwargs...) =
  HostPlan(plan_brfft(Array(x), d, rocfft_region(x, region)))

# its kernels run as CPU loops, which cannot hold the barriers inside loops of the tile kernel
const NFFTGPUExt = Base.get_extension(NFFT, :NFFTGPUExt)
NFFTGPUExt.use_tiles(::JLArrays.JLBackend) = false

arrayTypes = [JLArray]
floatTypes = [Float32, Float64]

include(joinpath(@__DIR__, "..", "gpu.jl"))
