# mlx-nufft (Python, MLX on Apple GPUs, Float32) through PythonCall, with the Python interpreter in
# the environment variable MLX_NUFFT_PYTHON (a Python with `pip install mlx-nufft`). FINUFFT's
# conventions: x = 2πk, modes in increasing order; type 2 with isign −1 is the NFFT, type 1 with
# isign +1 its adjoint. The data stay on the GPU as MLX arrays: the inner plans of `mlx_nufft.Plan`
# are called with `return_np = False`, since `Plan.execute` copies through NumPy both ways. MLX
# evaluates lazily; `sync` forces the result of the last call.
const PYEXE = get(ENV, "MLX_NUFFT_PYTHON", "")
if !isempty(PYEXE)
  ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
  ENV["JULIA_PYTHONCALL_EXE"] = PYEXE
  import PythonCall
end

supports(task) = isempty(PYEXE) ? "set MLX_NUFFT_PYTHON to a Python with mlx-nufft" :
                 task["transform"] != "nfft" ? "only the NFFT" :
                 task["T"] != "Float32" ? "mlx-nufft runs in Float32" :
                 get(task, "L", 1) != 1 ? "no batches in this adapter" : nothing

py(name) = PythonCall.pyimport(name)
mutable struct MLX                       # an MLX array, replaced by every transform
  a::Any
end
Base.similar(::MLX) = MLX(nothing)
const LAST = Ref{Any}(nothing)

upload(x) = MLX(py("mlx.core").array(py("numpy").asarray(ComplexF32.(x))))
tohost(x::MLX) = PythonCall.pyconvert(Array, py("numpy").array(x.a))
sync() = LAST[] === nothing || py("mlx.core").eval(LAST[])

function plan(prob, tol)
  m = py("mlx_nufft"); np = py("numpy")
  N = Tuple(prob.N)
  xs = [np.asarray(Float32.(2π .* prob.k[d, :])) for d in axes(prob.k, 1)]
  p2 = m.Plan(2, N; eps=tol, isign=-1, dtype="complex64"); p2.setpts(xs...)
  p1 = m.Plan(1, N; eps=tol, isign=1, dtype="complex64"); p1.setpts(xs...)
  (p2, p1)
end
trafo!(p, out, x) = (out.a = LAST[] = p[1]._plan.execute(x.a; return_np=false); out)
adjoint!(p, out, y) = (out.a = LAST[] = p[2]._plan.execute(y.a; return_np=false); out)

params(p) = Dict{String,Any}("W" => PythonCall.pyconvert(Int, p[1]._plan.w),
                             "sigma" => PythonCall.pyconvert(Float64, p[1]._plan.sigma), "window" => "ES")

libinfo() = isempty(PYEXE) ? Dict{String,Any}() :
  Dict{String,Any}("version" => PythonCall.pyconvert(String, py("mlx_nufft").__version__), "library" => PYEXE)
