using NFFTTools: sdc, sdc!

@testset "Sampling density" begin
  # nodes on a 9×8 grid: all weights equal 1/prod(N)
  N = (9, 8)
  for T in (Float32, Float64)
    grid1(n) = T.((0:n-1) ./ n .- 0.5)
    k = reduce(hcat, [[x; y] for x in grid1(N[1]), y in grid1(N[2])])
    p = plan_nfft(k, N; reltol = T == Float32 ? 1e-5 : 1e-8)
    w = sdc(p; iters = 10)
    @test w isa Vector{T}
    @test all(>(0), w)
    @test w ≈ fill(T(1 / prod(N)), length(w)) rtol = 1e-3

    # with work arrays: same result
    w2 = similar(w)
    workg = Array{Complex{T}}(undef, oversampled_size(p))
    workf = Array{Complex{T}}(undef, size(k, 2))
    workv = Array{Complex{T}}(undef, N)
    sdc!(w2, p; iters = 10, workg, workf, workv)
    @test w2 == w
  end

  # with work arrays only the task overhead of the threaded stages is allocated, far less than
  # one array of the problem size per iteration
  k = rand(MersenneTwister(1), 2, 5000) .- 0.5
  p = plan_nfft(k, (128, 128))
  w = Vector{Float64}(undef, 5000)
  workg = Array{ComplexF64}(undef, oversampled_size(p))
  workf = Array{ComplexF64}(undef, 5000)
  workv = Array{ComplexF64}(undef, 128, 128)
  sdc!(w, p; iters = 10, workg, workf, workv)
  @test (@allocated sdc!(w, p; iters = 10, workg, workf, workv)) < 10 * sizeof(workv)

  # radial trajectory: the weights grow with the distance from the centre
  nr, na = 32, 64
  r = range(-0.5, 0.5; length = nr + 1)[1:end-1]
  θ = range(0, π; length = na + 1)[1:end-1]
  k = reduce(hcat, [[ρ * cos(t); ρ * sin(t)] for ρ in r, t in θ])
  w = sdc(plan_nfft(k, (32, 32)); iters = 20)
  w = reshape(w, nr, na)
  @test all(>(0), w)
  @test w[nr ÷ 2 + 9, 1] > 2 * w[nr ÷ 2 + 3, 1]
end
