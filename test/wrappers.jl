@testset "Accuracy Wrappers" begin

include("../Wrappers/FINUFFT.jl")

@testset "FINUFFT Wrapper in multiple dimensions" begin
  for (u,N) in enumerate([(256,), (30,32), (10,12,14)]) # can only do D=1:3
    
    eps = 1e-7
      
    D = length(N)
    @info "Testing in $D dimensions"

    M = prod(N)
    x = rand(Float64,D,M) .- 0.5
    p = FINUFFTPlan(x, N) #; m, σ, LUTSize, precompute = pre, fftflags = FFTW.ESTIMATE)
    pNDFT = NDFTPlan(x, N)

    fHat = rand(Float64,M) + rand(Float64,M)*im
    f = ndft_adjoint(pNDFT, fHat)
    fApprox = nfft_adjoint(p, fHat)

    e = norm(f[:] - fApprox[:]) / norm(f[:])
    @debug "error adjoint nfft "  e
    @test e < eps

    gHat = ndft(pNDFT, f)
    gHatApprox = nfft(p, f)
    e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
    @debug "error nfft "  e
    @test e < eps

  end
end


end



include("../Wrappers/NFFT3.jl")

@testset "NFFT3 Wrapper in multiple dimensions" begin

  m = 5
  σ = 2.0
  LUTSize = 20000

  for (u,N) in enumerate([(256,), (30,32), (10,12,14), (6,6,6,6)])
    for pre in [NFFT.LUT, NFFT.FULL]
      eps = 1e-7
      
      D = length(N)
      @info "Testing in $D dimensions"

      M = prod(N)
      x = rand(Float64,D,M) .- 0.5
      p = NFFT3Plan(x, N; m, σ, LUTSize, precompute = pre,
                    fftflags = FFTW.ESTIMATE)
      pNDFT = NDFTPlan(x, N)

      fHat = rand(Float64,M) + rand(Float64,M)*im
      f = ndft_adjoint(pNDFT, fHat)
      fApprox = nfft_adjoint(p, fHat)
      e = norm(f[:] - fApprox[:]) / norm(f[:])
      @debug "error adjoint nfft "  e
      @test e < eps

      gHat = ndft(pNDFT, f)
      gHatApprox = nfft(p, f)
      e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
      @debug "error nfft "  e
      @test e < eps
    end
  end
end
