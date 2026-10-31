# NonuniformFFTs.jl: native type 2 uses e^{+ilx}, so x = -2πk; fftshift gives increasing order.
# It has no tolerance parameter; the half support m is taken from the tolerance as in NFFT.jl 0.14
# (W = 2m), with its default oversampling σ = 2. FFTW planning flag: the task's `fft_flag`, by default
# MEASURE, the package's default.
import NonuniformFFTs, FFTW

supports(task) = task["transform"] != "nfft" ? "only the complex NFFT is benchmarked" :
                 get(task, "L", 1) > 1 ? "batches not benchmarked" : nothing
fft_flag(task) = get(task, "fft_flag", "MEASURE")

function plan(prob, tol)
  T = prob.T
  m = (ceil(Int, log10(1 / tol)) + 1) ÷ 2
  p = NonuniformFFTs.PlanNUFFT(Complex{T}, prob.N; m=NonuniformFFTs.HalfSupport(m), fftshift=true,
                               fftw_flags=getfield(FFTW, Symbol(fft_flag(prob.task))))
  NonuniformFFTs.set_points!(p, ntuple(d -> mod2pi.(-T(2π) .* prob.k[d, :]), size(prob.k, 1)))
  p
end
trafo!(p, out, x) = NonuniformFFTs.exec_type2!(out, p, x)
adjoint!(p, out, y) = NonuniformFFTs.exec_type1!(out, p, y)
params(p) = Dict{String,Any}("W" => 2NonuniformFFTs.Kernels.half_support(first(p.kernels)), "sigma" => Float64(p.σ),
                             "grid" => collect(size(first(NonuniformFFTs.output_field(p.data)))))

libinfo() = Dict{String,Any}("version" => string(pkgversion(NonuniformFFTs)), "library" => dirname(dirname(pathof(NonuniformFFTs))))

function reported(prob, tol)
  q = params(plan(prob, tol))
  Dict{String,Any}("rep_W" => q["W"], "rep_W_adj" => q["W"], "rep_sigma" => q["sigma"], "rep_sigma_adj" => q["sigma"],
                   "rep_grid" => q["grid"], "rep_grid_adj" => q["grid"])
end
