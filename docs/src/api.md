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
NFFTPlan
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

## Abstract interface

```@docs
AbstractFTPlan
AbstractComplexFTPlan
AbstractRealFTPlan
AbstractNFFTPlan
AbstractNNFFTPlan
AbstractNFCTPlan
AbstractNFSTPlan
AbstractNFFTs.set_active_backend!
DEFAULT_RELTOL
GenericOptions
generic_options
check_options
```

## Index

```@index
```
