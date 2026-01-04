m = 5
σ = 2.0

@testset "GPU NFFT Plans" begin
  for arrayType in arrayTypes

    @testset "GPU_NFFT in multiple dimensions" begin
      for (u, N) in enumerate([(256,), (32, 32), (12, 12, 12)])
        eps = [1e-7, 1e-3, 1e-6, 1e-4]
        for (l, window) in enumerate([:kaiser_bessel, :gauss, :kaiser_bessel_rev, :spline])
          D = length(N)
          @info "Testing $arrayType in $D dimensions using $window window"

          J = prod(N)
          k = rand(Float64, D, J) .- 0.5
          p = plan_nfft(Array, k, N; m, σ, window, precompute=NFFT.FULL,
            fftflags=FFTW.ESTIMATE)
          p_d = plan_nfft(arrayType, k, N; m, σ, window, precompute=NFFT.FULL)
          p_d_infer = plan_nfft(arrayType(k), N; m, σ, window, precompute=NFFT.FULL)
          pNDFT = NDFTPlan(k, N)

          fHat = rand(Float64, J) + rand(Float64, J) * im
          f = adjoint(pNDFT) * fHat
          fHat_d = arrayType(fHat)
          
          # GPU NFFT
          fApprox_d = adjoint(p_d) * fHat_d
          fApprox_d_infer = adjoint(p_d_infer) * fHat_d
          @test fApprox_d ≈ fApprox_d_infer

          fApprox = Array(fApprox_d)
          e = norm(f[:] - fApprox[:]) / norm(f[:])
          @debug "error adjoint nfft " e
          @test e < eps[l]

          gHat = pNDFT * f
          gHatApprox = Array(p_d * arrayType(f))
          gHatApproxInfer = Array(p_d_infer * arrayType(f))
          @test gHatApprox ≈ gHatApproxInfer
          e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
          @debug "error nfft " e
          @test e < eps[l]
        end
      end
    end

    @testset "GPU_NFFT Sampling Density" begin

      # create a 10x10 grid of unit spaced sampling points
      N = 10
      g = (0:(N-1)) ./ N .- 0.5
      x = vec(ones(N) * g')
      y = vec(g * ones(N)')
      nodes = cat(x', y', dims=1)

      # approximate the density weights
      p = plan_nfft(arrayType, nodes, (N, N); m=5, σ=2.0)
      weights = Array(sdc(p, iters=5))

      # Infer the correct plan_nfft type
      p_infer = plan_nfft(arrayType(nodes), (N, N); m=5, σ=2.0)
      weights_infer = Array(sdc(p_infer, iters=5))
      @test weights ≈ weights_infer

      @info extrema(vec(weights))

      @test all((≈).(vec(weights), 1 / (N * N), rtol=1e-7))

    end
  end
end