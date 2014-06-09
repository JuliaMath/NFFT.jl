NFFT.jl
=======

This package provides a Julia implementation of the Non-equidistant Fast Fourier Transform (NFFT).
This algorithm is also referred as Gridding in the literature (e.g. in MRI literature) 
For a detailed introduction into the NFFT and its application please have a look at www.nfft.org.
The NFFT is a fast implementation of the Non-equidistant Discrete Fourier Transform (NDFT) that is
basically a DFT with non-equidistant sampling nodes in either Fourier or time/space domain. In contrast
to the FFT, the NFFT is an approximative algorithm whereas the accuracy can be controlled by two parameters:
the window width m and the oversampling factor sigma.

Basic usage of NFFT.jl is shown in the following example:

    using NFFT 
    
    N = 1024
    x = linspace(-0.4, 0.4, N)      # nodes at which the NFFT is evaluated
    fHat = randn(N)+randn(N)*1im    # data to be transformed
    p = NFFTPlan(x, N)              # create plan. m and sigma are optional parameters
    f = nfft_adjoint(p, fHat)       # calculate adjoint NFFT
    g = nfft(p, f)                  # calculate forward NFFT
    
There are currently some open issues:
  - The library is currently only fast for 1D, 2D, and 3D NFFTs. Higher order NFFTs use a slow fallback implementation.
  - The accuracy currently is not better than 1e-5 although it actually should be in the range of mashine accuracy for m=6, sigma=2.0. Have not found the reason yet.
