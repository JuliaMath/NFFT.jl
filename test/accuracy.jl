using LinearAlgebra
using FFTW

@testset "Accuracy" begin

m = 5
σ = 2.0
LUTSize = 20000

@testset "NFFT in multiple dimensions" begin
    for (u,N) in enumerate([(256,), (30,32), (10,12,14), (6,6,6,6)])
      for (pre,storeApod) in zip([NFFT.LUT, NFFT.FULL, NFFT.FULL_LUT, NFFT.LUT],
                                 [false, false, false, true])
        if !storeApod || length(N) == 1 # right now storApod works only in 1D...
        eps = [1e-7, 1e-3, 1e-6, 1e-4]
        for (l,window) in enumerate([:kaiser_bessel, :gauss, :kaiser_bessel_rev, :spline])
            D = length(N)
            @info "Testing in $D dimensions using window=$window precompute=$pre storeApod=$storeApod"

            M = prod(N)
            x = rand(Float64,D,M) .- 0.5
            p = plan_nfft(x, N; m, σ, window, LUTSize, precompute = pre, storeApodizationIdx = storeApod,
                         fftflags = FFTW.ESTIMATE)
            pNDFT = NDFTPlan(x, N)

            fHat = rand(Float64,M) + rand(Float64,M)*im
            f = ndft_adjoint(pNDFT, fHat)
            fApprox = nfft_adjoint(p, fHat)
            e = norm(f[:] - fApprox[:]) / norm(f[:])
            @debug "error adjoint nfft "  e
            @test e < eps[l]

            gHat = ndft(pNDFT, f)
            gHatApprox = nfft(p, f)
            e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
            @debug "error nfft "  e
            @test e < eps[l]
        end
        end
      end
    end
end

@testset "High level NFFT " begin
    N = (32,32)
    eps =  1e-3
    D = length(N)

    M = prod(N)
    x = rand(Float64,D,M) .- 0.5

    fHat = rand(Float64,M) + rand(Float64,M)*im
    f = ndft_adjoint(x, N, fHat)
    fApprox = nfft_adjoint(x, N, fHat)
    e = norm(f[:] - fApprox[:]) / norm(f[:])
    @debug "error adjoint nfft "  e
    @test e < eps

    gHat = ndft(x, f)
    gHatApprox = nfft(x, f)
    e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
    @debug "error nfft "  e
    @test e < eps

    f = ndft_adjoint(x, N, ComplexF32.(fHat))
    fApprox = nfft_adjoint(x, N, ComplexF32.(fHat))
    e = norm(f[:] - fApprox[:]) / norm(f[:])
    @debug "error adjoint nfft "  e
    @test e < eps

    gHat = ndft(x, ComplexF32.(f))
    gHatApprox = nfft(x, ComplexF32.(f))
    e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
    @debug "error nfft "  e
    @test e < eps
end

@testset "Abstract sampling points" begin
    M, N = rand(100:2:200, 2)
    x = range(-0.4, stop=0.4, length=M)
    p = plan_nfft(x, N, fftflags = FFTW.ESTIMATE)
end

@testset "Directional NFFT $D dim" for D in 2:3 begin
    # NFFT along a specified dimension should give the same result as
    # running a 1D NFFT on every slice along that dimension
        eps = 1e-4
        N = tuple( 2*rand(4:8,D)... )
        M = prod(N)
        for d in 1:D
            @info "Testing in $D dimensions directional NFFT along dim=$d"
            x = rand(M) .- 0.5

            f = rand(ComplexF64,N)
            p_dir = plan_nfft(x, N, dims=d)
            fHat_dir = nfft(p_dir, f)
            g_dir = nfft_adjoint(p_dir, fHat_dir)

            p = plan_nfft(x, N[d])
            fHat = similar(fHat_dir)
            g = similar(g_dir)

            sz = size(fHat)
            Rpre = CartesianIndices( sz[1:d-1] )
            Rpost = CartesianIndices( sz[d+1:end] )
            for Ipost in Rpost, Ipre in Rpre
                idx = [Ipre, :, Ipost]
                fview = f[idx...]
                fHat[idx...] = nfft(p, vec(fview))

                fHat_view = fHat_dir[idx...]
                g[idx...] = nfft_adjoint(p, vec(fHat_view))
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
      M = prod(N)
      for d in 1:(D-1)
          dims = d:(d+1)
          @info "Testing in $D dimensions directional NFFT along dim=$(dims)"
          x = rand(2,M) .- 0.5

          f = rand(ComplexF64, N)
          p_dir = plan_nfft(x, N, dims=dims)
          fHat_dir = nfft(p_dir, f)
          g_dir = nfft_adjoint(p_dir, fHat_dir)

          p = plan_nfft(x, N[dims])
          fHat = similar(fHat_dir)
          g = similar(g_dir)

          Rpre = CartesianIndices( N[1:dims[1]-1] )
          Rpost = CartesianIndices( N[(dims[end]+1):end] )
          for Ipost in Rpost, Ipre in Rpre
              idxf = [Ipre, :, :, Ipost]
              idxfhat = [Ipre, :, Ipost]

              fview = f[idxf...]
              fHat[idxfhat...] = nfft(p, (fview))

              fHat_view = fHat_dir[idxfhat...]
              g[idxf...] = nfft_adjoint(p, (fHat_view))
          end

          e = norm( fHat_dir[:] - fHat[:] )
          @test e < eps

          e = norm( g_dir[:] - g[:] ) / norm(g[:])
          @test e < eps
      end
  end
end

include("NFFT3.jl")

@testset "NFFT3 Wrapper in multiple dimensions" begin
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


# test CuNFFT
#=if CuNFFT.CUDA.functional()
    @testset "CuNFFT in multiple dimensions" begin
    for (u,N) in enumerate([(256,), (32,32), (12,12,12)])
        eps = [1e-7, 1e-3, 1e-6, 1e-4]
        for (l,window) in enumerate([:kaiser_bessel, :gauss, :kaiser_bessel_rev, :spline])
            D = length(N)
            @info "Testing CuNFFT in $D dimensions using $window window"

            M = prod(N)
            x = rand(Float64,D,M) .- 0.5
            p = plan_nfft(Array, x, N, m, σ, window, K, precompute = NFFT.FULL,
                         fftflags = FFTW.ESTIMATE, device=NFFT.CPU)
            p_d = plan_nfft(CuArray, x, N, m, σ, window, K)
            pNDFT = NDFTPlan(x, N)

            fHat = rand(Float64,M) + rand(Float64,M)*im
            f = ndft_adjoint(pNDFT, fHat)
            fHat_d = CuArray(fHat)
            fApprox_d = nfft_adjoint(p_d, fHat_d)
            fApprox = Array(fApprox_d)
            e = norm(f[:] - fApprox[:]) / norm(f[:])
            @debug "error adjoint nfft "  e
            @test e < eps[l]

            gHat = ndft(pNDFT, f)
            gHatApprox = Array( nfft(p_d, CuArray(f)) )
            e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
            @debug "error nfft "  e
            @test e < eps[l]
        end
    end
end

end
=#

end