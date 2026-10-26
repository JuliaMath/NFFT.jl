```@meta
CurrentModule = NFFT
```

# API

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
NFFTBackend
NFFT.parameters
accuracy
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
```

## Stages

```@docs
oversampled_size
resampling_correction!
oversampled_fft!
resampling!
resampling_adjoint!
oversampled_bfft!
resampling_correction_adjoint!
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
GenericOptions
generic_options
check_options
```

## Index

```@index
```
