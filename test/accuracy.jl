@testset "Accuracy" begin

m = 5
σ = 2.0

@testset "High-level NFFT" begin
  @info "High-level NFFT"
  N = (33,35)
  eps =  1e-3
  D = length(N)

  J = prod(N)
  k = rand(Float64,D,J) .- 0.5

  fHat = rand(Float64,J) + rand(Float64,J)*im
  f = ndft_adjoint(k, N, fHat)
  fApprox = nfft_adjoint(k, N, fHat, reltol=1e-9)
  e = norm(f[:] - fApprox[:]) / norm(f[:])
  @debug "error adjoint nfft "  e
  @test e < eps

  gHat = ndft(k, f)
  gHatApprox = nfft(k, f, reltol=1e-9)
  e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
  @debug "error nfft "  e
  @test e < eps

  f = ndft_adjoint(k, N, ComplexF32.(fHat))
  fApprox = nfft_adjoint(k, N, ComplexF32.(fHat))
  e = norm(f[:] - fApprox[:]) / norm(f[:])
  @debug "error adjoint nfft "  e
  @test e < eps

  gHat = ndft(k, ComplexF32.(f))
  gHatApprox = nfft(k, ComplexF32.(f))
  e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
  @debug "error nfft "  e
  @test e < eps
end

@testset "NFFT in multiple dimensions" begin
    for (u,N) in enumerate([(255,), (31,33), (11,12,14), (6,5,6,6)])
      for (pre, storeDeconv, blocking) in zip([ NFFT.LINEAR, NFFT.LINEAR, NFFT.LINEAR, NFFT.FULL, NFFT.TENSOR, NFFT.POLYNOMIAL, NFFT.POLYNOMIAL],
                                           [false, true, false, false, false, false, false],
                                           [true, true, false, false, true, true, false])
        eps = [1e-7, 1e-7, 1e-3, 1e-6, 1e-4]
        for (l,window) in enumerate([:kaiser_bessel, :cosh_type, :gauss, :kaiser_bessel_rev, :spline])
            D = length(N)
            @info "Testing $D, window=$window, pre=$pre, storeDeconv=$storeDeconv, block=$blocking"

            J = prod(N)
            k = rand(Float64,D,J) .- 0.5
            p = plan_nfft(k, N; m, σ, window, precompute = pre, storeDeconvolutionIdx = storeDeconv,
                         fftflags = FFTW.ESTIMATE, blocking)

            pNDFT = NDFTPlan(k, N)

            fHat = rand(Float64,J) + rand(Float64,J)*im
            f = adjoint(pNDFT) * fHat
            fApprox = adjoint(p) * fHat
            e = norm(f[:] - fApprox[:]) / norm(f[:])
            @debug "error adjoint nfft "  e
            @test e < eps[l]

            gHat = pNDFT * f
            gHatApprox = p * f
            e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
            @debug "error nfft "  e
            @test e < eps[l]
        end
      end
    end
end


@testset "Abstract sampling points" begin
  @info "Abstract sampling points"
    M, N = rand(100:2:200, 2)
    k =range(-0.4, stop=0.4, length=M)
    p = plan_nfft(k, N, fftflags = FFTW.ESTIMATE)
end

@testset "Directional NFFT $D dim" for D in 2:3 begin
    # NFFT along a specified dimension should give the same result as
    # running a 1D NFFT on every slice along that dimension
        eps = 1e-4
        N = tuple( 2*rand(4:8,D)... )
        J = prod(N)
        for d in 1:D
            @info "Testing in $D dimensions directional NFFT along dim=$d"
            k = rand(J) .- 0.5

            f = rand(ComplexF64,N)
            p_dir = plan_nfft(k, N, dims=d)
            fHat_dir = p_dir * f
            g_dir = adjoint(p_dir) * fHat_dir

            p = plan_nfft(k, N[d])
            fHat = similar(fHat_dir)
            g = similar(g_dir)

            sz = size(fHat)
            Rpre = CartesianIndices( sz[1:d-1] )
            Rpost = CartesianIndices( sz[d+1:end] )
            for Ipost in Rpost, Ipre in Rpre
                idx = [Ipre, :, Ipost]
                fview = f[idx...]
                fHat[idx...] = p * vec(fview)

                fHat_view = fHat_dir[idx...]
                g[idx...] = adjoint(p) * vec(fHat_view)
            end

            e = norm( fHat_dir[:] - fHat[:] )
            @test e < eps

            e = norm( g_dir[:] - g[:] ) / norm(g[:])
            @test e < eps
        end
    end
end

@testset "Directional NFFT $D dim" for D in 3:3 begin
  # NFFT along  specified dimensions (1:2 or 2:3) should give the same result as
  # running a 2D NFFT on every slice along that dimensions
      eps = 1e-4
      N = tuple( 2*rand(4:8,D)... )
      J = prod(N)
      for d in 1:(D-1)
          dims = d:(d+1)
          @info "Testing in $D dimensions directional NFFT along dim=$(dims)"
          k = rand(2,J) .- 0.5

          f = rand(ComplexF64, N)
          p_dir = plan_nfft(k, N, dims=dims)
          fHat_dir = p_dir * f
          g_dir = adjoint(p_dir) * fHat_dir

          p = plan_nfft(k, N[dims])
          fHat = similar(fHat_dir)
          g = similar(g_dir)

          Rpre = CartesianIndices( N[1:dims[1]-1] )
          Rpost = CartesianIndices( N[(dims[end]+1):end] )
          for Ipost in Rpost, Ipre in Rpre
              idxf = [Ipre, :, :, Ipost]
              idxfhat = [Ipre, :, Ipost]

              fview = f[idxf...]
              fHat[idxfhat...] = p * fview

              fHat_view = fHat_dir[idxfhat...]
              g[idxf...] = adjoint(p) * fHat_view
          end

          e = norm( fHat_dir[:] - fHat[:] )
          @test e < eps

          e = norm( g_dir[:] - g[:] ) / norm(g[:])
          @test e < eps
      end
  end
end

end
