# Internal ablation switches (src/ablation.jl): each gives the default result up to rounding, with
# nodes on the periodic seam and several tasks sharing the flushes.

@testset "Ablation switches" begin
  rng = Xoshiro(5)
  switches = [(fft=:full,), (fft=:unpruned,), (fft=:pruned_fftw,), (interp=:global,), (halo=false,),
              (flush=:globallock,), (flush=:atomic,), (halo=false, flush=:atomic), (flush=:reentrant,),
              (halo=false, flush=:reentrant), (window=:fullhorner,), (coords=:permutation,), (rowdot2=false,),
              (prefetch=false,), (slide=false,), (schedule=:chunks,), (bufs=:pertile,),
              (schedule=:chunks, bufs=:pertile), (interp=:global, window=:fullhorner, coords=:permutation),
              (fft=:full, flush=:reentrant, halo=false, window=:fullhorner, coords=:permutation, rowdot2=false,
               prefetch=false, slide=false, schedule=:chunks, bufs=:pertile)]
  for N in ((3000,), (60, 50), (24, 20, 28))
    D = length(N); J = 3000
    k = rand(rng, D, J) .- 0.5
    k[:, 1] .= -0.5; k[:, 2] .= prevfloat(0.5)
    f = randn(rng, ComplexF64, N); v = randn(rng, ComplexF64, J)
    for nthreads in (1, 3)
      p = plan_nfft(k, N; nthreads)
      r, a = p * f, adjoint(p) * v
      for sw in switches
        try
          NFFT.ablate!(; sw...)
          q = plan_nfft(k, N; nthreads)
          @test q.core isa NFFT.AblatedCore
          @test q * f ≈ r rtol = 1e-13
          @test adjoint(q) * v ≈ a rtol = 1e-13
        finally
          NFFT.ablate!()
        end
      end
      @test plan_nfft(k, N; nthreads).core isa NFFT.NFFTCore
      # the copies of the resampling loops in the ablated core, all switches at their default
      q = plan_nfft(k, N; nthreads)
      q.core = NFFT.ablated_core(q.core, k, NFFT.ABLATION_DEFAULT)
      if nthreads == 1
        @test q * f == r
        @test adjoint(q) * v == a
      else
        @test q * f ≈ r rtol = 1e-13
        @test adjoint(q) * v ≈ a rtol = 1e-13
      end
    end
  end
  @test_throws ArgumentError NFFT.ablate!(fft=:other)
  @test_throws ArgumentError NFFT.ablate!(interp=:global, halo=false)
  @test_throws ArgumentError NFFT.ablate!(schedule=:static)
  @test_throws ArgumentError NFFT.ablate!(prefetch=:no)
  NFFT.ablate!()
end
