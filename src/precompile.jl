# Workload for the package image: NFFT plans and transforms in 1D–3D, single and double precision, at
# the default tolerance and at 1e-4 (window widths 6–12), and the other transforms and a batch in 2D,
# so that the first transform of a session does not compile the setup and the common loops.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
  sizes = ((256,), (24, 24), (10, 10, 10))
  @compile_workload begin
    for T in (Float32, Float64), N in sizes, reltol in (T == Float32 ? 1e-6 : 1e-8, 1e-4)
      D = length(N)
      k = rand(T, D, 200) .- T(1 // 2)
      p = plan_nfft(NFFTBackend(), k, N; reltol)    # __init__ (activation) has not run yet
      f = randn(Complex{T}, N)
      y = p * f
      adjoint(p) * y
    end
    for T in (Float32, Float64)
      k = rand(T, 2, 200) .- T(1 // 2)
      p = plan_rnfft(NFFTBackend(), k, (24, 24))
      adjoint(p) * (p * randn(Complex{T}, size_in(p)))
      for plan in (plan_nfct, plan_nfst)
        q = plan(NFFTBackend(), abs.(k), (24, 24))
        transpose(q) * (q * randn(T, 24, 24))
      end
      r = plan_nnfft(NFFTBackend(), k, 8 .* k)
      adjoint(r) * (r * randn(Complex{T}, 200))
      b = plan_nfft(NFFTBackend(), k, (24, 24, 4); dims=1:2)
      adjoint(b) * (b * randn(Complex{T}, 24, 24, 4))
    end
  end
end
