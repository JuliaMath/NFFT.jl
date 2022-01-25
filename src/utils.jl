const _use_threads = Ref(false)

macro cthreads(loop::Expr) 
  return esc(quote
      if NFFT._use_threads[]
          @floop $loop
          #Threads.@threads $loop 
          #@batch per=thread $loop
      else
          @inbounds $loop
      end
  end)
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

### consistency check

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


