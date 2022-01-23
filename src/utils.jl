const _use_threads = Ref(false)

macro cthreads(loop::Expr) 
  return esc(quote
      if NFFT._use_threads[]
          Threads.@threads $loop 
          # @batch per=thread $loop
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
  @printf "                       apod = %.4f / %.4f %% fft = %.4f / %.4f %% conv = %.4f / %.4f %%\n" 100*t.apod/total 100*t.apod_adjoint/totalAdj 100*t.fft/total 100*t.fft_adjoint/totalAdj 100*t.conv/total 100*t.conv_adjoint/totalAdj
  
end


# copy of Base.Cartesian macros, which we need to generalize

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

### consistancy check

@generated function consistencyCheck(p::AbstractNFFTPlan{T,D,R}, f::AbstractArray{U,D},
                                     fHat::AbstractArray{Y}) where {T,D,R,U,Y}
  quote
    if size_in(p) != size(f) || size_out(p) != size(fHat)
      throw(DimensionMismatch("Data is not consistent with NFFTPlan"))
    end
  end
end


### Threaded sparse matrix vector multiplications ###

# not yet threaded ...
function threaded_mul!(y::AbstractVector, A::SparseMatrixCSC{Tv}, x::AbstractVector) where {Tv}
  nzv = nonzeros(A)
  rv = rowvals(A)
  fill!(y, zero(Tv))

  @inbounds @simd for col in 1:size(A, 2)
       _threaded_mul!(y, A, x, nzv, rv, col)
  end
   y
end

@inline function _threaded_mul!(y, A::SparseMatrixCSC{Tv}, x, nzv, rv, col) where {Tv}
  tmp = x[col] 

  @inbounds @simd for j in nzrange(A, col)
      y[rv[j]] += nzv[j]*tmp
  end
  return
end

# threaded
function threaded_mul!(C, xA::Transpose{<:Any,<:SparseMatrixCSC}, B)
      A = xA.parent
      size(A, 2) == size(C, 1) || throw(DimensionMismatch())
      size(A, 1) == size(B, 1) || throw(DimensionMismatch())
      size(B, 2) == size(C, 2) || throw(DimensionMismatch())
      nzv = nonzeros(A)
      rv = rowvals(A)

      @cthreads for col in 1:size(A, 2)
          _threaded_tmul!(C, A, B, nzv, rv, col)
      end
      C
end


function _threaded_tmul!(C, A, B, nzv, rv, col)
  tmp = zero(eltype(C))
  @inbounds for j in nzrange(A, col)
      tmp += transpose(nzv[j])*B[rv[j]]
  end
  C[col] = tmp 
  return
end


