# GPU

This guide shows how to run the transforms of NFFT.jl on a GPU. All transforms (NFFT, RNFFT, NFCT,
NFST, NNFFT) run there, including batch dimensions. The kernels are written with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and the FFTs go through
AbstractFFTs, so the same code serves CUDA.jl, AMDGPU.jl and Metal.jl arrays. The GPU code is a
package extension that is loaded together with the GPU package.

## Create a plan on the GPU

A plan works on the GPU when its storage is a GPU array type: give the nodes as a GPU array, or pass
`storage` with nodes on the CPU.

```julia
using NFFT, CUDA                     # or AMDGPU, Metal

k = rand(Float32, 2, 100_000) .- 0.5f0
f = CUDA.randn(ComplexF32, 256, 256)

p = plan_nfft(k, size(f); reltol=1e-5, storage=CuArray)   # or plan_nfft(CuArray(k), size(f))
fHat = p * f                         # CuArray
g = adjoint(p) * fHat
nodes(p)                             # the nodes as a CuArray (in the precision of the plan)
```

The other transforms take the same option:

```julia
plan_rnfft(k, (256, 256); storage=CuArray)
plan_nfct(abs.(k), (256, 256); storage=CuArray)
plan_nnfft(k, y; storage=CuArray)
plan_nfft(k, (256, 256, 16); dims=1:2, storage=CuArray)   # 16 coils in one call
```

`sdc` from NFFTTools.jl and the stage interface work with GPU plans as well. Data on the CPU passed
to a GPU plan raise an error instead of being copied.

The options `fft` and `tilesize` belong to the CPU code; GPU plans reject them. Timings are on the
[Performance](@ref "On GPUs") page, the kernels are explained in [How NFFT.jl Works](@ref "GPUs").

## Parameter choice on a GPU

A plan chooses window width and oversampling factor by a cost model (see
[Accuracy and Parameters](@ref)). GPUs have their own: resampling, FFT and fixed costs weigh
differently there than on the CPU. It is fitted on an Apple M1 Max. For another GPU,
`NFFT.calibrate_cost` times NFFTs for a range of parameters (about three minutes) and makes the
plans of the session use the fit:

```julia
coef = NFFT.calibrate_cost(CuArray; eltype=Float32)
NFFT.set_cost_model!(CuArray, Float32, coef)   # in a later session
```

`measure = true` times the four best candidates of the cost model when the plan is created.

## Run the GPU tests

`test/gpu.jl` compares every transform on the GPU with the CPU plan. It runs with

```
julia test/gpu/run.jl cuda      # or rocm, metal
```

in a temporary environment with the GPU package.
