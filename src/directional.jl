
function AbstractNFFTs.deconvolve!(p::NFFTPlan{T,D,R}, f::AbstractArray{U,D},
                      g::StridedArray{Complex{T},D}) where {T,D,R,U}
  dstart = Val(p.dims[1])
  dend = Val(p.dims[end])
  return _deconvolve!(p, f, g, dstart, dend)
end

@generated function _deconvolve!(p::NFFTPlan{T,D,R}, f::AbstractArray{U,D},
                                  g::StridedArray{Complex{T},D}, dstart::Val{DS}, 
                                  dend::Val{DE}) where {T,D,R,U,DS,DE}
    quote
        @nloops $D l f d->begin
            # preexpr
            if $DS ≤ d ≤ $DE
                offset_d = round(Int, p.n[d] - p.N[d]÷2) - 1
                gidx_d = rem(l_d+offset_d, p.n[d]) + 1
            else
                gidx_d = l_d
            end
        end begin
            # bodyexpr
            v = (@nref $D f l)
            @nexprs $(DE-DS+1) d -> v *= p.windowHatInvLUT[d][l_{d+$DS-1}]
            (@nref $D g gidx) = v
        end
    end
end

function AbstractNFFTs.deconvolve_transpose!(p::NFFTPlan{T,D,R},
                 g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}) where {T,D,R,U}
  dstart = Val(p.dims[1])
  dend = Val(p.dims[end])
  return _deconvolve_transpose!(p, g, f, dstart, dend)
end

@generated function _deconvolve_transpose!(p::NFFTPlan{T,D,R},
           g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}, dstart::Val{DS}, 
           dend::Val{DE}) where {T,D,R,U,DS,DE}
    quote
        @nloops $D l f d->begin
            # preexpr
            if $DS ≤ d ≤ $DE
                offset_d = round(Int, p.n[d] - p.N[d]÷2) - 1
                gidx_d = rem(l_d+offset_d, p.n[d]) + 1
            else
                gidx_d = l_d
            end
        end begin
            # bodyexpr
            v =  (@nref $D g gidx)
            @nexprs $(DE-DS+1) d -> v *= p.windowHatInvLUT[d][l_{d+$DS-1}]
            (@nref $D f l) = v
        end
    end
end

function AbstractNFFTs.convolve!(p::NFFTPlan{T,D,R}, g::AbstractArray{Complex{T},D},
      fHat::StridedArray{U,R}) where {T,D,R,U}
  l1 = Val(p.dims[1]-1)
  l2 = Val(length(p.dims))
  l3 = Val(D-p.dims[end])
  return _convolve!(p, g, fHat, l1, l2, l3)
end

@generated function _convolve!(p::NFFTPlan{T,D,R}, g::AbstractArray{Complex{T},D},
                   fHat::StridedArray{U,R}, l1::Val{L1}, l2::Val{L2}, l3::Val{L3}) where {T,D,R,U,L1,L2,L3}
  quote
    fill!(fHat, zero(T))
    scale = Int(p.params.LUTSize/(p.params.m))

    for k in 1:p.M
      ### The first L1 loops contain no NFFT
      @nloops_ $L1 d->l_{d} d->1:size(g,d) d->begin
         # preexpr
         gidx_d = l_d
         fidx_d = l_d
      end begin
          # bodyexpr
          @nexprs 1 d -> fidx_{$L1+1} = k
          @nexprs 1 d -> prodWin_{$L2} = one(T)
          @nexprs $L2 d -> xscale_d = p.x[d,k] * p.n[d+$L1]
          @nexprs $L2 d -> c_d = floor(Int, xscale_d)
          ### The next L2 loops contain the NFFT
          @nloops_ $L2 d->l_{d+$L1} d->(c_d-p.params.m+1):(c_d+p.params.m) d->begin
            # preexpr
            gidx_{d+$L1} = rem(l_{d+$L1}+p.n[d+$L1], p.n[d+$L1]) + 1
            idx = abs((xscale_d - l_{d+$L1} )*scale) + 1
            idxL = floor(idx)
            idxInt = Int(idxL)
            prodWin_{d-1} = prodWin_d * (p.windowLinInterp[idxInt] + ( idx-idxL ) * (p.windowLinInterp[idxInt+1] - p.windowLinInterp[idxInt]))
          end begin
             # bodyexpr
             ### The last L3 loops contain no NFFT
             @nloops_ $L3 d->l_{d+$L1+$L2} d->1:size(g,d+$L1+$L2) d->begin
                 # preexpr
                 gidx_{d+$L1+$L2} = l_{d+$L1+$L2}
                 fidx_{d+$L1+1} = l_{d+$L1+$L2}
              end begin  
                (@nref $R fHat fidx) += (@nref $D g gidx) * prodWin_0
              end 
          end
      end
    end
  end
end

function AbstractNFFTs.convolve_transpose!(p::NFFTPlan{T,D,R}, fHat::AbstractArray{U,R},
        g::StridedArray{Complex{T},D}) where {T,D,R,U}
  l1 = Val(p.dims[1]-1)
  l2 = Val(length(p.dims))
  l3 = Val(D-p.dims[end])
  return _convolve_transpose!(p, fHat, g, l1, l2, l3)
end

@generated function _convolve_transpose!(p::NFFTPlan{T,D,R}, fHat::AbstractArray{U,R},
            g::StridedArray{Complex{T},D}, l1::Val{L1}, l2::Val{L2}, l3::Val{L3}) where {T,D,R,U,L1,L2,L3}
  quote
    fill!(g, zero(T))
    scale = Int(p.params.LUTSize/(p.params.m))

    for k in 1:p.M
      ### The first L1 loops contain no NFFT
      @nloops_ $L1 d->l_{d} d->1:size(g,d) d->begin
         # preexpr
         gidx_d = l_d
         fidx_d = l_d
      end begin
          # bodyexpr
          @nexprs 1 d -> fidx_{$L1+1} = k
          @nexprs 1 d -> prodWin_{$L2} = one(T)
          @nexprs $L2 d -> xscale_d = p.x[d,k] * p.n[d+$L1]
          @nexprs $L2 d -> c_d = floor(Int, xscale_d)
          ### The next L2 loops contain the NFFT
          @nloops_ $L2 d->l_{d+$L1} d->(c_d-p.params.m+1):(c_d+p.params.m) d->begin
            # preexpr
            gidx_{d+$L1} = rem(l_{d+$L1}+p.n[d+$L1], p.n[d+$L1]) + 1
            idx = abs((xscale_d - l_{d+$L1})*scale) + 1
            idxL = floor(idx)
            idxInt = Int(idxL)
            prodWin_{d-1} = prodWin_d * (p.windowLinInterp[idxInt] + ( idx-idxL ) * (p.windowLinInterp[idxInt+1] - p.windowLinInterp[idxInt]))
          end begin
             # bodyexpr
             ### The last L3 loops contain no NFFT
             @nloops_ $L3 d->l_{d+$L1+$L2} d->1:size(g,d+$L1+$L2) d->begin
                 # preexpr
                 gidx_{d+$L1+$L2} = l_{d+$L1+$L2}
                 fidx_{d+$L1+1} = l_{d+$L1+$L2}
              end begin  
                (@nref $D g gidx) += (@nref $R fHat fidx) * prodWin_0
              end 
          end
      end
    end
  end
end
