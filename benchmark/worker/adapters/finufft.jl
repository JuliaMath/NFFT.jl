# FINUFFT: x = 2πk ∈ [-π, π); modes in increasing order (modeord = 0). Type 2 with iflag -1 is the
# NFFT, type 1 with iflag +1 its adjoint; batches through ntrans. NNFFT: type 3 from the sources 2πy
# to the targets k with iflag -1, and from 2πk to y with iflag +1 for the adjoint. FINUFFT rejects tolerances below
# its rounding floor (which grows with N, in single precision above 1e-2 for N = 2^18); the
# tolerance is then raised until accepted, up to 0.5.
import FINUFFT, finufft_jll, FFTW

supports(task) = task["transform"] in ("nfft", "nnfft") ? nothing : "FINUFFT has no $(task["transform"])"

function plan(prob, tol)
  T = prob.T; L = size(prob.y, 2)
  x = ntuple(d -> T(2π) .* prob.k[d, :], size(prob.k, 1))
  opts = (; dtype=T, nthreads=Threads.nthreads(), fftw=UInt32(FFTW.ESTIMATE))
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

libinfo() = Dict{String,Any}("version" => string(pkgversion(FINUFFT)), "library" => finufft_jll.libfinufft_path)
