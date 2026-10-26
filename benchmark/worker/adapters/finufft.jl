# FINUFFT: x = 2πk ∈ [-π, π); modes in increasing order (modeord = 0). Type 2 with iflag -1 is the
# NFFT, type 1 with iflag +1 its adjoint; batches through ntrans. FINUFFT rejects tolerances below
# its rounding floor (which grows with N, in single precision above 1e-2 for N = 2^18); the
# tolerance is then raised until accepted, up to 0.5.
import FINUFFT, finufft_jll, FFTW

supports(task) = task["transform"] == "nfft" ? nothing : "FINUFFT has no $(task["transform"])"

function plan(prob, tol)
  T = prob.T; L = size(prob.y, 2)
  x = ntuple(d -> T(2π) .* prob.k[d, :], size(prob.k, 1))
  opts = (; dtype=T, nthreads=Threads.nthreads(), fftw=UInt32(FFTW.ESTIMATE))
  while true
    plans = FINUFFT.finufft_plan{T}[]
    try
      for (type, iflag) in ((2, -1), (1, 1))
        p = FINUFFT.finufft_makeplan(type, collect(prob.N), iflag, L, tol; opts...)
        push!(plans, p)
        FINUFFT.finufft_setpts!(p, x...)
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
