using Enzyme

# Enzyme through mul! with a constant plan (AbstractNFFTsEnzymeCoreExt): the reverse gradient of
# L(x) = Σ |w ∘ (A x)|² against a central difference along a random direction, the forward tangent
# against A applied to the direction.
@testset "Enzyme" begin
  rng = Xoshiro(11)
  apply(A, z, To) = (o = zeros(To, size_out(A)); mul!(o, A, z); o)
  loss(z, A, w, To) = sum(abs2, w .* apply(A, z, To))
  tangent!(o, A, z) = (mul!(o, A, z); nothing)

  N = (12, 10); J = 40
  k = rand(rng, 2, J) .- 0.5
  kr = rand(rng, 2, J) ./ 2
  p = plan_nfft(k, N; reltol=1e-10)
  q = plan_nfct(kr, N; reltol=1e-10)
  s = plan_nfst(kr, N; reltol=1e-10)
  r = plan_rnfft(k, N; reltol=1e-10)
  nn = plan_nnfft(8 .* k, 3 .* rand(rng, 2, 30); reltol=1e-10)
  b = plan_nfft(k, (3, N...); dims=2:3, reltol=1e-10)
  C = ComplexF64
  # (name, map, input, output type)
  cases = [("nfft", p, randn(rng, C, N), C), ("nfft, real input", p, randn(rng, N), C),
           ("adjoint nfft", adjoint(p), randn(rng, C, J), C),
           ("nfct", q, randn(rng, N), Float64), ("transposed nfct", transpose(q), randn(rng, J), Float64),
           ("nfst", s, randn(rng, N), Float64), ("transposed nfst", transpose(s), randn(rng, J), Float64),
           ("rnfft", r, randn(rng, C, size_in(r)), Float64), ("adjoint rnfft", adjoint(r), randn(rng, J), C),
           ("nnfft", nn, randn(rng, C, 30), C), ("adjoint nnfft", adjoint(nn), randn(rng, C, J), C),
           ("nfft, dims = 2:3", b, randn(rng, C, 3, N...), C)]

  @testset "$name" for (name, A, x, To) in cases
    w = randn(rng, size_out(A)...)
    d = oftype(x, randn(rng, eltype(x), size(x)))
    ε = 1e-6
    fd = (loss(x .+ ε .* d, A, w, To) - loss(x .- ε .* d, A, w, To)) / 2ε
    g = zero(x)
    autodiff(Reverse, loss, Active, Duplicated(copy(x), g), Const(A), Const(w), Const(To))
    @test abs(real(sum(conj.(g) .* d)) - fd) < 1e-6 * abs(fd)

    y = zeros(To, size_out(A)); dy = zero(y)
    autodiff(Forward, tangent!, Duplicated(y, dy), Const(A), Duplicated(copy(x), d))
    @test dy ≈ apply(A, d, To)
  end
end
