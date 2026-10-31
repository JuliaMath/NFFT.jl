# ducc0: periodicity 1 with the nodes as given; fft_order = false gives increasing order. Batches
# loop over the slices (ducc has no batched NUFFT). Tolerances below ducc's best achievable epsilon
# make plan creation fail, hence the clamp.
import Ducc0, ducc0_jll

supports(task) = task["transform"] == "nfft" ? nothing : "ducc has no $(task["transform"])"

function plan(prob, tol)
  D = size(prob.k, 1)
  tol = max(tol, 1.1 * Ducc0.Nufft.best_epsilon(D, prob.T == Float32))
  TOL_USED[] = tol
  Ducc0.Nufft.make_plan(prob.k, prob.N; epsilon=tol, nthreads=Threads.nthreads(), periodicity=1.0,
                        fft_order=false)
end
function trafo!(p, out, x)
  if ndims(out) == 1
    Ducc0.Nufft.u2nu_planned!(p, x, out; forward=true)
  else
    for l in axes(out, 2)
      Ducc0.Nufft.u2nu_planned!(p, selectdim(x, ndims(x), l), view(out, :, l); forward=true)
    end
  end
  out
end
function adjoint!(p, out, y)
  if ndims(y) == 1
    Ducc0.Nufft.nu2u_planned!(p, y, out; forward=false)
  else
    for l in axes(y, 2)
      Ducc0.Nufft.nu2u_planned!(p, view(y, :, l), selectdim(out, ndims(out), l); forward=false)
    end
  end
  out
end

# ducc's own choice (one plan for both directions) from its verbose output: W = supp and the
# oversampled grid.
function reported(prob, tol)
  p = plan(prob, tol)
  T = prob.T
  _, out = capture_c(() -> Ducc0.Nufft.u2nu_planned!(p, zeros(Complex{T}, prob.N...), zeros(Complex{T}, size(prob.k, 2));
                                                     forward=true, verbose=true))
  m = match(r"oversampled grid=\(([0-9x]+)\), supp=(\d+)", out)
  m === nothing && return Dict{String,Any}()
  grid = parse.(Int, split(m[1], "x"))
  W = parse(Int, m[2])
  Dict{String,Any}("rep_W" => W, "rep_W_adj" => W, "rep_grid" => grid, "rep_grid_adj" => grid)
end

libinfo() = Dict{String,Any}("version" => string(pkgversion(Ducc0)), "library" => ducc0_jll.libducc_julia_path)
