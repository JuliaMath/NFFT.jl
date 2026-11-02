# NonuniformFFTs.jl on an Nvidia GPU (CUDA, KernelAbstractions backend), data on the device; the
# conventions and the half support from the tolerance as in nonuniformffts.jl, its default GPU method.
import NonuniformFFTs, CUDA, FFTW

supports(task) = task["transform"] != "nfft" ? "only the complex NFFT is benchmarked" :
                 get(task, "L", 1) > 1 ? "batches not benchmarked" : nothing

function plan(prob, tol)
  T = prob.T
  m = (ceil(Int, log10(1 / tol)) + 1) ÷ 2
  p = NonuniformFFTs.PlanNUFFT(Complex{T}, prob.N; m=NonuniformFFTs.HalfSupport(m), fftshift=true,
                               backend=CUDA.CUDABackend())
  NonuniformFFTs.set_points!(p, ntuple(d -> CUDA.CuArray(mod2pi.(-T(2π) .* prob.k[d, :])), size(prob.k, 1)))
  p
end
trafo!(p, out, x) = NonuniformFFTs.exec_type2!(out, p, x)
adjoint!(p, out, y) = NonuniformFFTs.exec_type1!(out, p, y)
upload(x) = CUDA.CuArray(x)
tohost(x) = Array(x)
sync() = CUDA.synchronize()
params(p) = Dict{String,Any}("W" => 2NonuniformFFTs.Kernels.half_support(first(p.kernels)), "sigma" => Float64(p.σ))

libinfo() = Dict{String,Any}("version" => string(pkgversion(NonuniformFFTs), ", CUDA ", pkgversion(CUDA)),
                             "library" => dirname(dirname(pathof(NonuniformFFTs))))
