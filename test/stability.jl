# Type stability and allocations of the transforms of every plan type. A transform allocates a
# constant amount (task scheduling, views, the FFTW plans behind a Vector{Any}), never memory of
# the size of the data: the bounds below are far below the size of the arrays of the large problem.

# bytes allocated by mul!, the fewest of three calls after a warm-up (in a function: no boxing of globals)
function mul_allocs(y, p, x)
  mul!(y, p, x)
  minimum(@allocated(mul!(y, p, x)) for _ in 1:3)
end

function stability_plans(J, n, nthreads)
  rng = Xoshiro(5)
  k1 = rand(rng, 1, J) .- 0.5; k2 = rand(rng, 2, J) .- 0.5; k3 = rand(rng, 3, J) .- 0.5
  kp = abs.(k2) ./ 2
  C = ComplexF64
  # (name, plan, input type, output type)
  [("NFFT 1D", plan_nfft(k1, (n^2,); nthreads), C, C),
   ("NFFT 2D", plan_nfft(k2, (n, n); nthreads), C, C),
   ("NFFT 3D", plan_nfft(k3, (n ÷ 4, n ÷ 4, n ÷ 4); nthreads), C, C),
   ("NFFT Float32", plan_nfft(Float32.(k2), (n, n); nthreads), ComplexF32, ComplexF32),
   ("NFFT batch first", plan_nfft(k2, (4, n, n); dims=2:3, nthreads), C, C),
   ("NFFT batch last", plan_nfft(k2, (n, n, 4); dims=1:2, nthreads), C, C),
   ("NFFT batch middle", plan_nfft(k1, (3, n, 5); dims=2, nthreads), C, C),
   ("RNFFT", plan_rnfft(k2, (n, n); nthreads), C, Float64),
   ("RNFFT batch", plan_rnfft(k1, (n, 6); dims=1, nthreads), C, Float64),
   ("NFCT", plan_nfct(kp, (n, n); nthreads), Float64, Float64),
   ("NFST", plan_nfst(kp, (n, n); nthreads), Float64, Float64),
   ("NFCT batch", plan_nfct(kp[1:1, :], (n, 6); dims=1, nthreads), Float64, Float64),
   ("NNFFT", plan_nnfft(k2, rand(rng, 2, J ÷ 2) .- 0.5; nthreads), C, C),
   ("NDFT", NDFTPlan(k2[:, 1:50], (8, 8)), C, C)]
end

@testset "Type stability and allocations" begin
  for (name, p, Tin, Tout) in stability_plans(500, 16, Threads.nthreads())
    @testset "$name" begin
      back = Tin <: Real ? transpose(p) : adjoint(p)
      f = randn(Tin, size_in(p)); fHat = randn(Tout, size_out(p))
      @test @inferred(p * f) isa Array{Tout}
      @test @inferred(back * fHat) isa Array{Tin}
      @test @inferred(mul!(fHat, p, f)) === fHat
      @test @inferred(mul!(f, back, fHat)) === f
      # views of the data
      fv = view(randn(Tin, size_in(p) .+ 1), axes(f)...)
      @test @inferred(p * fv) ≈ p * collect(fv)
    end
  end

  # single-threaded: a few KB per call, the same for a small and a large problem; with threads a
  # few KB per task
  for (J, n) in ((500, 16), (20_000, 128)), (nt, bound) in ((1, 2^15), (Threads.nthreads(), 2^15 * (Threads.nthreads() + 1)))
    @testset "allocations, J = $J, nthreads = $nt: $name" for (name, p, Tin, Tout) in stability_plans(J, n, nt)
      back = Tin <: Real ? transpose(p) : adjoint(p)
      f = randn(Tin, size_in(p)); fHat = randn(Tout, size_out(p))
      # the batch plans with threaded cores per slice (:loop) schedule tasks for every slice
      L = p isa NFFT.FastPlan && p.strategy === :loop ? prod(size_in(p)) ÷ prod(p.N[p.dims]) : 1
      @test mul_allocs(fHat, p, f) <= L * bound
      @test mul_allocs(f, back, fHat) <= L * bound
    end
  end
end
