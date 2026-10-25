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
end
