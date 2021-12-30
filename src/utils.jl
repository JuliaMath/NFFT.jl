const _use_threads = Ref(false)

macro cthreads(loop::Expr) 
  return esc(quote
      if NFFT._use_threads[]
          @batch per=thread $loop
      else
          @inbounds $loop
      end
  end)
end

mutable struct TimingStats
  pre::Float64
  conv::Float64
  fft::Float64
  apod::Float64
  conv_adjoint::Float64
  fft_adjoint::Float64
  apod_adjoint::Float64
end

TimingStats() = TimingStats(0.0,0.0,0.0,0.0,0.0,0.0,0.0)

function Base.println(t::TimingStats)
  print("Timing: ")
  @printf "pre = %.4f s apod = %.4f / %.4f s fft = %.4f / %.4f s conv = %.4f / %.4f s\n" t.pre t.apod t.apod_adjoint t.fft t.fft_adjoint t.conv t.conv_adjoint

  total = t.apod + t.fft + t.conv 
  totalAdj = t.apod_adjoint + t.fft_adjoint + t.conv_adjoint
  @printf "                       apod = %.4f / %.4f %% fft = %.4f / %.4f %% conv = %.4f / %.4f %%\n" t.apod/total t.apod_adjoint/totalAdj t.fft/total t.fft_adjoint/totalAdj t.conv/total t.conv_adjoint/totalAdj
  
end


import Base.Cartesian.inlineanonymous


macro nloops_(N, itersym, rangeexpr, args...)
  _nloops_(N, itersym, rangeexpr, args...)
end

function _nloops_(N::Int, itersym, arraysym::Symbol, args::Expr...)
  @gensym d
  _nloops_(N, itersym, :($d->Base.axes($arraysym, $d)), args...)
end

function _nloops_(N::Int, itersym, rangeexpr::Expr, args::Expr...)
  if rangeexpr.head !== :->
      throw(ArgumentError("second argument must be an anonymous function expression to compute the range"))
  end
  if !(1 <= length(args) <= 3)
      throw(ArgumentError("number of arguments must be 1 ≤ length(args) ≤ 3, got $nargs"))
  end
  body = args[end]
  ex = Expr(:escape, body)
  for dim = 1:N
      itervar = inlineanonymous(itersym, dim)
      rng = inlineanonymous(rangeexpr, dim)
      preexpr = length(args) > 1 ? inlineanonymous(args[1], dim) : (:(nothing))
      postexpr = length(args) > 2 ? inlineanonymous(args[2], dim) : (:(nothing))
      ex = quote
        @inbounds for $(esc(itervar)) = $(esc(rng))
              $(esc(preexpr))
              $ex
              $(esc(postexpr))
          end
      end
  end
  ex
end


@generated function consistencyCheck(p::AbstractNFFTPlan{D,DIM,T}, f::AbstractArray{U,D},
                                     fHat::AbstractArray{Y}) where {D,DIM,T,U,Y}
  quote
    M = numFourierSamples(p)
    N = size(p)
    if $DIM == 0
      fHat_test = (M == length(fHat))
    elseif $DIM > 0
      fHat_test = @nall $D d -> ( d == $DIM ? size(fHat,d) == M : size(fHat,d) == N[d] )
    end

    if N != size(f) || !fHat_test
      throw(DimensionMismatch("Data is not consistent with NFFTPlan"))
    end
  end
end