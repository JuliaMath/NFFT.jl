using Test
using NFFT
using LinearAlgebra
using FFTW

eps = 1e-3
m = 5
sigma = 2.0

@testset "NFFT in multiple dimensions" begin
    for N in [(128,), (16,16), (12,12,12), (6,6,6,6)]
        for window in [:kaiser_bessel, :gauss, :kaiser_bessel_rev, :spline]
            D = length(N)
            println("Testing in ", D, " dimensions using ", string(window)," window" )

            M = prod(N)
            x = rand(D,M) .- 0.5
            p = NFFTPlan(x, N, m, sigma, window, flags = FFTW.ESTIMATE)

            fHat = rand(M) + rand(M)*im
            f = ndft_adjoint(p, fHat)
            fApprox = nfft_adjoint(p, fHat)
            e = norm(f[:] - fApprox[:]) / norm(f[:])
            #= println(e) =#
            @test e < eps

            gHat = ndft(p, f)
            gHatApprox = nfft(p, f)
            e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
            #= println(e) =#
            @test e < eps
        end
    end
end

@testset "Abstract sampling points" begin
    M, N = rand(100:200, 2)
    x = linspace(-0.4, 0.4, M)
    p = NFFTPlan(x, N, flags = FFTW.MEASURE)
end

@testset "Directional NFFT $D dim" for D in 2:3 begin
    # NFFT along a specified dimension should give the same result as
    # running a 1D NFFT on every slice along that dimension
        N = tuple( 2*rand(4:8,D)... )
        M = prod(N)
        for d in 1:D
            x = rand(M) .- 0.5

            f = rand(N) + rand(N)*im
            p_dir = NFFTPlan(x, d, N)
            fHat_dir = nfft(p_dir, f)
            g_dir = nfft_adjoint(p_dir, fHat_dir)

            p = NFFTPlan(x, N[d])
            fHat = similar(fHat_dir)
            g = similar(g_dir)

            sz = size(fHat)
            Rpre = CartesianRange( sz[1:d-1] )
            Rpost = CartesianRange( sz[d+1:end] )
            for Ipost in Rpost, Ipre in Rpre
                idx = [Ipre, :, Ipost]
                fview = f[idx...]
                fHat[idx...] = nfft(p, vec(fview))

                fHat_view = fHat_dir[idx...]
                g[idx...] = nfft_adjoint(p, vec(fHat_view))
            end

            e = norm( fHat_dir[:] - fHat[:] )
            @test e ≈ 0

            e = norm( g_dir[:] - g[:] ) / norm(g[:])
            @test e < eps
        end
    end
end
