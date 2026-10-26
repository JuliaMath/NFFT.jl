# GPU

All transforms of NFFT.jl (NFFT, RNFFT, NFCT, NFST, NNFFT) run on GPUs, including batch dimensions. The
kernels are written with [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
and the FFTs go through AbstractFFTs, so the same code serves CUDA.jl, AMDGPU.jl and Metal.jl
arrays. The GPU path is a package extension: it is loaded with the GPU package.

## Usage

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

## How it works

The plan is set up on the CPU: parameter choice, binning of the nodes and the polynomial
approximation of the window. The GPU holds, per node, the offset of its support in the grid and its
fractional position, so that the window is evaluated in the working precision (Metal has no
`Float64`) while the grid position itself was computed in double precision. No sparse matrix is
stored: memory grows like the grid plus the nodes.

- **Resampling** (grid to nodes): one work-item per node and transform.
- **Adjoint resampling** (nodes to grid) in 2D and 3D: one workgroup per tile of the grid collects
  the contributions of the nodes near the tile in local memory and writes the tile once. This needs
  no atomic operations, and the result is the same in every call. In 1D, and for tiles that do not
  fit into local memory, the work-items add their contributions with atomic additions.
- **Halo**: filled and folded by gathering, without atomics.
- **Batches** (`dims`): every kernel and FFT processes all transforms at once.

## Speed on GPUs

On an Apple M1 Max (Metal, single precision, ``2^{18}`` nodes) the direct transforms on the GPU are
as fast as or faster than on the 8 performance cores of the CPU (3D, ``64^3``, tolerance
``10^{-4}``: 1.7 ms against 3.7 ms). The adjoint transforms are slower than on the CPU there, since
Apple GPUs have slow floating point atomics and little local memory (3D: 14.5 ms against 3.9 ms).
GPUs with fast atomics and more local memory, such as those of Nvidia, are expected to behave
differently; measurements there are still missing.

## Testing

`test/gpu.jl` compares every transform on the GPU with the CPU plan. It runs with

```
julia test/gpu/run.jl cuda      # or rocm, metal
```

in a temporary environment with the GPU package.
