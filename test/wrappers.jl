@testset "Accuracy Wrappers" begin

include("../Wrappers/FINUFFT.jl")

@testset "FINUFFT Wrapper in multiple dimensions" begin
  for (u,N) in enumerate([(256,), (30,32), (10,12,14)]) # can only do D=1:3
    
    eps = 1e-7
      
    D = length(N)
    @info "Testing in $D dimensions"

    J = prod(N)
    k = rand(Float64,D,J) .- 0.5
    p = FINUFFTPlan(k, N) #; m, σ, precompute = pre, fftflags = FFTW.ESTIMATE)
    pNDFT = NDFTPlan(k, N)

    fHat = rand(Float64,J) + rand(Float64,J)*im
    f = adjoint(pNDFT) * fHat
    fApprox = adjoint(p) * fHat

    e = norm(f[:] - fApprox[:]) / norm(f[:])
    @debug "error adjoint nfft "  e
    @test e < eps

    gHat = pNDFT * f
    gHatApprox = p * f
    e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
    @debug "error nfft "  e
    @test e < eps

  end
end



@testset "FINNUFFT Wrapper in multiple dimensions" begin
  for (u,N) in enumerate([(40,1), (40,2), (40,3)])
    
    eps = 1e-7
    D = N[2]

    J =N[1]
    k = rand(Float64,D,J) .- 0.5
    y = (rand(Float64,D,N[1]) .- 0.5) .* 10

    p = FINNUFFTPlan(k, y) 
    pNNDFT = NNDFTPlan(k, y)

    fHat = rand(Float64,J) + rand(Float64,J)*im
    f = adjoint(pNNDFT) * fHat
    fApprox = adjoint(p) * fHat

    e = norm(f[:] - fApprox[:]) / norm(f[:])
    @debug "error adjoint nnfft "  e
    @test e < eps

    gHat = pNNDFT * f
    gHatApprox = p * f

    e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
    @debug "error nnfft "  e
    @test e < eps

  end
end



end



include("../Wrappers/NFFT3.jl")

@testset "NFFT3 Wrapper in multiple dimensions" begin

  m = 5
  σ = 2.0

  for (u,N) in enumerate([(256,), (30,32), (10,12,14), (6,6,6,6)])
    for pre in [NFFT.LINEAR, NFFT.FULL]
      eps = 1e-7
      
      D = length(N)
      @info "Testing in $D dimensions"

      J = prod(N)
      k = rand(Float64,D,J) .- 0.5
      p = NFFT3Plan(k, N; m, σ, precompute = pre,
                    fftflags = FFTW.ESTIMATE)
      pNDFT = NDFTPlan(k, N)

      fHat = rand(Float64,J) + rand(Float64,J)*im
      f = adjoint(pNDFT) * fHat
      fApprox = adjoint(p) * fHat
      e = norm(f[:] - fApprox[:]) / norm(f[:])
      @debug "error adjoint nfft "  e
      @test e < eps

      gHat = pNDFT * f
      gHatApprox = p * f
      e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
      @debug "error nfft "  e
      @test e < eps
    end
  end
end



@testset "NFFT3 Wrapper NFCT" begin

  m = 5
  σ = 2.0

  for (u,N) in enumerate([(256,), (30,32), (10,12,14), (6,6,6,6)])
    eps = 1e-7
    
    D = length(N)
    @info "Testing in $D dimensions"

    J = prod(N)
    k =0.5.*rand(Float64,D,J) 
    p = NFCT3Plan(k, N; m, σ)
    pNDCT = NDCTPlan(k, N)

    fHat = rand(Float64,J) 
    f = transpose(pNDCT) * fHat
    fApprox = transpose(p) * fHat
    e = norm(f[:] - fApprox[:]) / norm(f[:])
    @debug "error transpose nfct "  e
    @test e < eps

    gHat = pNDCT * f
    gHatApprox = p * f
    e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
    @debug "error ndct "  e
    @test e < eps
  end
end


@testset "NFFT3 Wrapper NFST" begin

  m = 5
  σ = 2.0

  for (u,N) in enumerate([(256,), (8,8), (10,12,14), (6,6,6,6)])
    eps = 1e-7
    
    D = length(N)
    @info "Testing in $D dimensions"

    J = prod(N)
    k =0.5.*rand(Float64,D,J) 
    p = NFST3Plan(k, N .+ 1; m, σ)
    pNDST = NDSTPlan(k, N .+ 1)

    fHat = rand(Float64,J) 
    f = transpose(pNDST) * fHat
    fApprox = transpose(p) * fHat
    e = norm(f[:] - fApprox[:]) / norm(f[:])
    @debug "error transpose nfst "  e
    @test e < eps

    gHat = pNDST * f
    gHatApprox = p * f
    e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
    @debug "error ndct "  e
    @test e < eps
  end
end