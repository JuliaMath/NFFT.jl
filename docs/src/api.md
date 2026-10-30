```@meta
CurrentModule = NFFT
```

# API

## Options

Keyword arguments of `plan_nfft`, `plan_rnfft`, `plan_nfct` and `plan_nfst`; `plan_nnfft` takes
them without `tilesize` and `measure`, its `dims` is the dimension of the sources in an array of the
size given as third argument, and its `window`, `W` and `σ` apply to its inner NFFT:

| keyword | meaning |
|:--|:--|
| `reltol` | requested relative error, default `1e-8` |
| `dims` | dimensions to transform, see [Transforms along Dimensions](@ref) |
| `eltype` | `Float32` or `Float64`, default from the nodes |
| `storage` | array type of data and workspace, default from the nodes; see [GPU](@ref) |
| `nthreads` | number of threads, default `Threads.nthreads()` |
| `window` | `:auto`, `KaiserBessel()`, `ExpSemicircle()`, `CustomWindow(φ)` |
| `W`, `σ` | window width and oversampling factor |
| `fft` | FFT backend: `FFTWBackend()` (default) or `FFTABackend()` (pure Julia, `using FFTA`; NFFT and NNFFT) |
| `tilesize` | edge length of the tiles the nodes are sorted into (expert option) |
| `measure` | `true`: time the four parameter sets of lowest predicted cost at plan time and keep the fastest |

The first five are part of the [Abstract Interface](@ref) and mean the same for every
implementation; the others are specific to NFFT.jl. An unknown option is an error, and so are
`fft` and `tilesize` for plans on a GPU. How to use
`window`, `W`, `σ` and `fft` is shown in [Choosing Parameters](@ref).

## Plans and transforms

```@docs
plan_nfft
plan_nnfft
plan_nfct
plan_nfst
plan_rnfft
NFFTPlan
NFCTPlan
NFSTPlan
RNFFTPlan
NNFFTPlan
NFFTBackend
NFFT.parameters
accuracy
NFFT.calibrate_cost
NFFT.set_cost_model!
nodes
nodes!
size_in
size_out
mul!
*
nfft
nfft_adjoint
nfct
nfct_transpose
nfst
nfst_transpose
```

## Windows and FFT

```@docs
AbstractWindow
KaiserBessel
ExpSemicircle
CustomWindow
AbstractFFTBackend
FFTWBackend
FFTABackend
```

## Stages

```@docs
oversampled_size
spectrum_size
resampling_correction!
oversampled_fft!
resampling!
resampling_adjoint!
oversampled_bfft!
resampling_correction_adjoint!
Main.SparseArrays.sparse(::NFFTPlan)
```

## NFFTTools

```@meta
CurrentModule = NFFTTools
```

```@docs
sdc
sdc!
calculateToeplitzKernel
calculateToeplitzKernel!
convolveToeplitzKernel!
```

```@meta
CurrentModule = NFFT
```

## Direct transforms

```@docs
NDFTPlan
NDCTPlan
NDSTPlan
NNDFTPlan
```

## Abstract interface

```@docs
AbstractFTPlan
AbstractComplexFTPlan
AbstractRealFTPlan
AbstractNFFTPlan
AbstractNNFFTPlan
AbstractNFCTPlan
AbstractNFSTPlan
AbstractRNFFTPlan
AbstractNFFTs.set_active_backend!
DEFAULT_RELTOL
default_reltol
GenericOptions
generic_options
check_options
```

## Index

```@index
```
