using Zygote: ChainRulesCore

@testset "Chainrules" begin
  rng = Xoshiro(7)
  NT = ChainRulesCore.NoTangent()

  # (one-shot transform, its adjoint/transpose, plan, direct plan, data type, nodes)
  cases = [(nfft, nfft_adjoint, plan_nfft, NDFTPlan, complex, k -> k .- one(eltype(k)) / 2),
           (nfct, nfct_transpose, plan_nfct, NDCTPlan, identity, k -> k ./ 2),
           (nfst, nfst_transpose, plan_nfst, NDSTPlan, identity, k -> k ./ 2)]

  for (op, op_t, plan, dplan, dtype, scale) in cases, T in (Float32, Float64), N in (17, 33)
    tol = T == Float64 ? 1e-8 : 1e-4
    M = 2N
    k = scale(rand(rng, T, 1, M))
    p = plan(k, (N,); reltol=tol)
    A = dplan(k, (N,))
    pt = op === nfft ? adjoint(p) : transpose(p)
    At = op === nfft ? adjoint(A) : transpose(A)
    x = dtype.(randn(rng, T, N) .+ (dtype === complex ? im .* randn(rng, T, N) : 0))
    v = dtype.(randn(rng, T, M) .+ (dtype === complex ? im .* randn(rng, T, M) : 0))

    @testset "$op $T N=$N" begin
      # gradient of ‖A x‖² is 2 Aᴴ A x
      g = 2 * (At * (A * x))
      @test relerr(Zygote.gradient(z -> sum(abs2, op(k, z; reltol=tol)), x)[1], g) < 10tol
      @test relerr(Zygote.gradient(z -> sum(abs2, p * z), x)[1], g) < 10tol
      gt = 2 * (A * (At * v))
      @test relerr(Zygote.gradient(z -> sum(abs2, op_t(k, N, z; reltol=tol)), v)[1], gt) < 10tol
      @test relerr(Zygote.gradient(z -> sum(abs2, pt * z), v)[1], gt) < 10tol

      # forward mode: the tangent of the data is transformed
      Δx = dtype.(randn(rng, T, N))
      @test relerr(ChainRulesCore.frule((NT, NT, Δx), op, k, x)[2], A * Δx) < 10tol
      @test relerr(ChainRulesCore.frule((NT, NT, Δx), *, p, x)[2], A * Δx) < 10tol
      Δv = dtype.(randn(rng, T, M))
      @test relerr(ChainRulesCore.frule((NT, NT, NT, Δv), op_t, k, N, v)[2], At * Δv) < 10tol
    end
  end
end
