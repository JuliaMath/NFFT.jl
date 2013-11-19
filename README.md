NFFT.jl
=======
This package provides a Julia implementation of the Non-equidistant Fast Fourier Transform (NFFT).
For a detailed introduction into the NFFT and its application please have a look at www.nfft.org.
Basic usage is shown in the following example:

  x = linspace(-0.4, 0.4, N)      # nodes at which the NFFT is evaluated
  fHat = randn(N)+randn(N)*1im    # data to be transformed
  p = NFFTPlan(x, N, m, sigma);   # create plan

  f = ndft_adjoint(p, fHat)       # calculate adjoint NFFT
  g = nfft_adjoint(p, f)          # calculate forward NFFT