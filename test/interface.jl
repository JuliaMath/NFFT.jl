# The operator interface of AbstractNFFTs on the plans of NFFT.jl: sizes, adjoint and transpose,
# copies, printing, and the active backend.
using NFFT: AbstractNFFTs
using NFFT.AbstractNFFTs: AbstractRealFTPlan
@testset "AbstractNFFTs interface" begin
  Random.seed!(3)
  N = (8, 6); J = 20
  k = rand(2, J) .- 0.5
  kr = rand(2, J) ./ 2
  for p in (plan_nfft(k, N), plan_rnfft(k, N), plan_nnfft(k, rand(2, 7) .* 4), plan_nfct(kr, N), plan_nfst(kr, N))
    real = p isa AbstractRealFTPlan
    q = real ? transpose(p) : adjoint(p)
    @test size(p) == (prod(size_out(p)), prod(size_in(p)))
    @test size(q) == reverse(size(p))
    @test length(p) == prod(size(p))
    @test eltype(p) == (real ? Float64 : ComplexF64)
    @test size_in(q) == size_out(p) && size_out(q) == size_in(p)
    c = copy(q)
    @test typeof(c) == typeof(q) && c.parent !== p
    s = sprint(show, q); m = sprint(show, MIME("text/plain"), q)
    @test s == m && startswith(s, real ? "Transposed " : "Adjoint ") && endswith(s, sprint(show, p))
  end

  # the adjoint RNFFT with batch dimensions
  p = plan_rnfft(k, (8, 6, 3); dims=1:2)
  fHat = randn(size_out(p))
  h = adjoint(p) * fHat
  @test size(h) == size_in(p)
  @test h[:, :, 2] ≈ adjoint(plan_rnfft(k, N)) * fHat[:, 2]

  # NNFFT nodes as vectors in 1D
  x = rand(5); y = rand(4)
  @test nodes(plan_nnfft(x, y)) == reshape(x, 1, :)

  # the active backend, as module or backend object; without one, a clear error
  b = AbstractNFFTs.active_backend()
  try
    AbstractNFFTs.set_active_backend!(missing)
    @test_throws "No default backend" plan_nfft(k, N)
    @test_throws "No default backend" plan_nnfft(k, k)
    @test_throws "No default backend" nfft(k, randn(ComplexF64, N))
    @test_throws "No default backend" nfft_adjoint(k, N, randn(ComplexF64, J))
    AbstractNFFTs.set_active_backend!(NFFT)
    @test AbstractNFFTs.active_backend() isa NFFT.NFFTBackend
  finally
    AbstractNFFTs.set_active_backend!(b)
  end
end
