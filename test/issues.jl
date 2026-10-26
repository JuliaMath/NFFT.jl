@testset "Issues Encountered during Development" begin

  # https://github.com/JuliaMath/NFFT.jl/issues/106: an out-of-bounds access for Float32 nodes
  # close to a grid point (in the linear-interpolation lookup of 0.14).
  T = Float32
  Nz = 120
  N = (2Nz,)
  λ = Array{Complex{T}}(undef, N)
  trj = zeros(T, 1, 2)
  p = plan_nfft(trj, N)
  trj[1, :] .= 0.008333333
  p = plan_nfft(trj, N)
  mul!(λ, adjoint(p), ones(Complex{T}, size(trj, 2)))
  @test all(isfinite, λ)

  # A node just below 0 lies at the end of the grid after wrapping; binning and support once
  # computed its grid position differently and wrote outside the tile buffer (odd widths).
  for W in (9, 10, 11)
    k = [-eps(0.5) 0.0 -1e-9 0.25]
    q = plan_nfft(k, 16; W, σ=2.0, window=KaiserBessel())
    @test all(isfinite, adjoint(q) * ones(ComplexF64, 4))
  end

  # arrays smaller than the window: the grid must hold the halo (a grid of 2 points with W = 10
  # gave errors of 1e-2)
  @testset "tiny arrays" begin
    rng = Xoshiro(3)
    for N in (1, 2, 3), D in 1:3
      NN = ntuple(_ -> N, D)
      k = rand(rng, D, 50) .- 0.5
      p = plan_nfft(k, NN; reltol=1e-9); q = NDFTPlan(k, NN)
      f = randn(rng, ComplexF64, NN); y = randn(rng, ComplexF64, 50)
      @test norm(p * f - q * f) < 1e-8 * norm(q * f)
      @test norm(adjoint(p) * y - adjoint(q) * y) < 1e-8 * norm(adjoint(q) * y)
    end
  end
end
