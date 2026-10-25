# Batch dimensions (`dims`): every strategy (batch first, batch last, loop over slices) has to agree
# with one plan per slice.

@testset "Batch dimensions" begin
  rng = Xoshiro(5)
  J = 200
  cases = [((6, 40), 2, :lead), ((40, 5), 1, :trail), ((3, 18, 20), 2:3, :lead),
           ((18, 20, 4), 1:2, :trail), ((3, 30, 2), 2, :loop), ((10, 8, 9, 2), 1:3, :loop),
           ((1, 32), 2, :single)]
  for (N, dims, strategy) in cases
    Dd = length(dims)
    k = rand(rng, Dd, J) .- 0.5
    p = plan_nfft(k, N; dims, reltol=1e-9)
    @test p.strategy == strategy
    q = plan_nfft(k, N[dims]; reltol=1e-9)
    f = randn(rng, ComplexF64, N)
    NOut = (N[1:first(dims)-1]..., J, N[last(dims)+1:end]...)
    fHat = randn(rng, ComplexF64, NOut)
    L1 = prod(N[1:first(dims)-1]; init=1); L3 = prod(N[last(dims)+1:end]; init=1)
    fr = reshape(f, L1, N[dims]..., L3); gr = reshape(fHat, L1, J, L3)
    ref = similar(gr); refadj = similar(fr)
    for b in 1:L3, a in 1:L1
      ref[a, :, b] = q * fr[a, ntuple(_ -> Colon(), Dd)..., b]
      refadj[a, ntuple(_ -> Colon(), Dd)..., b] = adjoint(q) * gr[a, :, b]
    end
    @test reshape(p * f, size(ref)) ≈ ref
    @test reshape(adjoint(p) * fHat, size(refadj)) ≈ refadj
    # copies and new nodes keep the strategy and give the same result
    c = copy(p)
    @test c * f == p * f
    k2 = rand(rng, Dd, J + 10) .- 0.5
    nodes!(c, k2)
    @test c.strategy == strategy
    @test size(c * f) == (N[1:first(dims)-1]..., J + 10, N[last(dims)+1:end]...)
  end
end
