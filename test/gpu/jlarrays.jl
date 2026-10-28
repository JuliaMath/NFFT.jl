# The GPU code path on the CPU: JLArrays is the reference implementation of a GPU array
# (KernelAbstractions' CPU backend), so that every CI run covers the GPU kernels.
using JLArrays, LinearAlgebra, NFFT
using NFFT.FFTW
const AbstractFFTs = FFTW.AbstractFFTs

# JLArrays has no FFTs: plans that run FFTW on a host copy.
struct HostPlan{P}
  p::P
end
Base.:*(h::HostPlan, x::JLArray) = copyto!(x, h.p * Array(x))
LinearAlgebra.mul!(y::JLArray, h::HostPlan, x::JLArray) = copyto!(y, h.p * Array(x))
AbstractFFTs.plan_fft!(x::JLArray, region; kwargs...) = HostPlan(plan_fft(Array(x), region))
AbstractFFTs.plan_bfft!(x::JLArray, region; kwargs...) = HostPlan(plan_bfft(Array(x), region))
AbstractFFTs.plan_rfft(x::JLArray, region; kwargs...) = HostPlan(plan_rfft(Array(x), region))
AbstractFFTs.plan_brfft(x::JLArray, d::Integer, region; kwargs...) = HostPlan(plan_brfft(Array(x), d, region))

# its kernels run as CPU loops, which cannot hold the barriers inside loops of the tile kernel
const NFFTGPUExt = Base.get_extension(NFFT, :NFFTGPUExt)
NFFTGPUExt.use_tiles(::JLArrays.JLBackend) = false

arrayTypes = [JLArray]
floatTypes = [Float32, Float64]

include(joinpath(@__DIR__, "..", "gpu.jl"))
