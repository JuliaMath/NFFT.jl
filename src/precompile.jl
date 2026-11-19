# Workload for the package image: NFFT plans and transforms in 1D–3D, single and double precision, for
# every window width (4–16) and both windows, and the other transforms and a batch in 2D, so that
# the first plan and transform of a session do not compile the setup and the loops. The kernels
# are specialised on the width and the window, so a width left out compiles at the first plan
# that chooses it (≈ 0.4 s); covering all of them costs ≈ 24 MB of image and 15 s of precompilation.
using PrecompileTools: @setup_workload, @compile_workload

# Deterministic nodes and data: Random is not a dependency, and Julia builds without it in the system
# image cannot run `rand` here.
golden_nodes(T, D, J) = reshape(T.(mod.((1:D*J) .* 0.6180339887498949, 1) .- 1 // 2), D, J)

@setup_workload begin
  sizes = ((256,), (24, 24), (10, 10, 10))
  @compile_workload begin
    for T in (Float32, Float64), N in sizes
      D = length(N)
      k = golden_nodes(T, D, 200)
      # the planner, through the active backend as users call it (__init__ has not activated it yet)
      with(nfft_backend => NFFTBackend()) do
        plan_nfft(k, N)
        plan_nfft(k, N; reltol = 1e-4)
      end
      # the kernels have the window width and the window function as type parameters: every
      # width and window the planner can choose
      for W in 4:16, window in (KaiserBessel(), ExpSemicircle())
        p = plan_nfft(NFFTBackend(), k, N; window, W, σ = 2.0)
        f = ones(Complex{T}, N)
        y = p * f
        adjoint(p) * y
      end
    end
    for T in (Float32, Float64)
      k = golden_nodes(T, 2, 200)
      p = plan_rnfft(NFFTBackend(), k, (24, 24))
      adjoint(p) * (p * ones(Complex{T}, size_in(p)))
      for plan in (plan_nfct, plan_nfst)
        q = plan(NFFTBackend(), abs.(k), (24, 24))
        transpose(q) * (q * ones(T, 24, 24))
      end
      r = plan_nnfft(NFFTBackend(), k, 8 .* k)
      adjoint(r) * (r * ones(Complex{T}, 200))
      b = plan_nfft(NFFTBackend(), k, (24, 24, 4); dims=1:2)
      adjoint(b) * (b * ones(Complex{T}, 24, 24, 4))
    end
  end
end
