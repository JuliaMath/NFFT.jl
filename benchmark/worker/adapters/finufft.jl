# FINUFFT: x = 2πk ∈ [-π, π); modes in increasing order (modeord = 0). Type 2 with iflag -1 is the
# NFFT, type 1 with iflag +1 its adjoint; batches through ntrans. NNFFT: type 3 from the sources 2πy
# to the targets k with iflag -1, and from 2πk to y with iflag +1 for the adjoint. FINUFFT rejects tolerances below
# its rounding floor (which grows with N, in single precision above 1e-2 for N = 2^18); the
# tolerance is then raised until accepted, up to 0.5. FFTW planning flag: the task's `fft_flag`, by
# default ESTIMATE, FINUFFT's default (opts.fftw).
import FINUFFT, finufft_jll, FFTW

supports(task) = task["transform"] in ("nfft", "nnfft") ? nothing : "FINUFFT has no $(task["transform"])"
fft_flag(task) = get(task, "fft_flag", "ESTIMATE")

function plan(prob, tol; debug=0)
  T = prob.T; L = size(prob.y, 2)
  x = ntuple(d -> T(2π) .* prob.k[d, :], size(prob.k, 1))
  opts = (; dtype=T, nthreads=Threads.nthreads(), fftw=UInt32(getfield(FFTW, Symbol(fft_flag(prob.task)))), debug)
  nnfft = prob.task["transform"] == "nnfft"
  D = size(prob.k, 1)
  pad(v) = (v..., ntuple(_ -> T[], 3 - D)...)          # setpts! takes x, y, z, then s, t, u
  while true
    plans = FINUFFT.finufft_plan{T}[]
    try
      if nnfft
        ys = ntuple(d -> T(2π) .* prob.src[d, :], D)
        kt = ntuple(d -> prob.k[d, :], D)
        yt = ntuple(d -> prob.src[d, :], D)
        for (src, tgt, iflag) in ((ys, kt, -1), (x, yt, 1))
          p = FINUFFT.finufft_makeplan(3, D, iflag, L, tol; opts...)
          push!(plans, p)
          FINUFFT.finufft_setpts!(p, pad(src)..., pad(tgt)...)
        end
      else
        for (type, iflag) in ((2, -1), (1, 1))
          p = FINUFFT.finufft_makeplan(type, collect(prob.N), iflag, L, tol; opts...)
          push!(plans, p)
          FINUFFT.finufft_setpts!(p, x...)
        end
      end
      TOL_USED[] = tol
      return (plans[1], plans[2])
    catch e
      foreach(FINUFFT.finufft_destroy!, plans)
      tol < 0.25 || throw(Unsupported("tolerance below FINUFFT's rounding floor"))
      tol *= 2
    end
  end
end
trafo!(p, out, x) = FINUFFT.finufft_exec!(p[1], x, out)
adjoint!(p, out, y) = FINUFFT.finufft_exec!(p[2], y, out)
release!(p) = foreach(FINUFFT.finufft_destroy!, p)

# FINUFFT's own choice for both plans (type 2, type 1), from its debug output: σ, W = ns, β (three
# digits) and the fine grid.
function reported(prob, tol)
  p, out = capture_c(() -> plan(prob, tol; debug=1))
  release!(p)
  chose = collect(eachmatch(r"sigma=([0-9.e+-]+): chose ns=(\d+) beta=([0-9.e+-]+)", out))
  grids = collect(eachmatch(r"\(nf1,nf2,nf3\)=\((\d+),(\d+),(\d+)\)", out))
  length(chose) >= 2 && length(grids) >= 2 || return Dict{String,Any}()
  D = length(prob.N)
  # the last choice before each plan's grid is the one the plan keeps (setpts re-plans with its σ)
  pick(g) = last(filter(c -> c.offset < g.offset, chose))
  r = Dict{String,Any}()
  for (suffix, g) in (("", grids[end-1]), ("_adj", grids[end]))
    c = pick(g)
    r["rep_sigma"*suffix] = parse(Float64, c[1]); r["rep_W"*suffix] = parse(Int, c[2])
    r["rep_beta"*suffix] = parse(Float64, c[3])
    r["rep_grid"*suffix] = parse.(Int, g.captures[1:D])
  end
  r
end

libinfo() = Dict{String,Any}("version" => string(pkgversion(FINUFFT)), "library" => finufft_jll.libfinufft_path)
