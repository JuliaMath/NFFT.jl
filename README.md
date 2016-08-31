NFFT.jl
=======

[![Build Status](https://travis-ci.org/tknopp/NFFT.jl.svg?branch=master)](https://travis-ci.org/tknopp/NFFT.jl)

This package provides a Julia implementation of the Non-equidistant Fast Fourier Transform (NFFT).
This algorithm is also referred as Gridding in the literature (e.g. in MRI literature) 
For a detailed introduction into the NFFT and its application please have a look at www.nfft.org.
The NFFT is a fast implementation of the Non-equidistant Discrete Fourier Transform (NDFT) that is
basically a DFT with non-equidistant sampling nodes in either Fourier or time/space domain. In contrast
to the FFT, the NFFT is an approximative algorithm whereas the accuracy can be controlled by two parameters:
the window width `m` and the oversampling factor `sigma`.

Basic usage of NFFT.jl is shown in the following example for 1D:

    using NFFT 
    
	M, N = 1024, 512
    x = linspace(-0.4, 0.4, M)      # nodes at which the NFFT is evaluated
    fHat = randn(M) + randn(M)*im   # data to be transformed
    p = NFFTPlan(x, N)              # create plan. m and sigma are optional parameters
    f = nfft_adjoint(p, fHat)       # calculate adjoint NFFT
    g = nfft(p, f)                  # calculate forward NFFT

In 2D:

	M, N = 1024, 16
	x = rand(2, M) - 0.5
	fHat = randn(M) + randn(M)*im
	p = NFFTPlan(x, (N,N))
    f = nfft_adjoint(p, fHat)
    g = nfft(p, f)

