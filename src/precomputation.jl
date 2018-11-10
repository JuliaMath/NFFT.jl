
@generated function precomputeB(win, x, n::NTuple{D,Int}, m, M, sigma, T, U1, U2) where D
    quote
        I = zeros(Int64, (2*m+1)^D, M)
        J = zeros(Int64, M+1)
        V = zeros(T, (2*m+1)^D, M)

        J[1] = 1
        @inbounds @simd for k in 1:M
            @nexprs $D d -> xscale_d = x[d,k] * n[d]
            @nexprs $D d -> c_d = floor(Int, xscale_d)

            @nloops $D l d -> (c_d-m):(c_d+m) d->begin
                # preexpr
                gidx_d = rem(l_d+n[d], n[d]) + 1
                Iidx_d = l_d - c_d + m + 1
                idx = abs( (xscale_d - l_d) )
                tmpWin_d =  win(idx / n[d], n[d], m, sigma)

            end begin
                # bodyexpr
                v = 1
                @nexprs $D d -> v *= tmpWin_d
                i1 = 1
                @nexprs $D d -> i1 += (Iidx_d-1) * U1[d]
                i2 = 1
                @nexprs $D d -> i2 += (gidx_d-1) * U2[d]

                I[i1,k] = i2
                V[i1,k] = v
            end
            J[k+1] = J[k] + (2*m+1)^D
        end

        S = SparseMatrixCSC(prod(n), M, J, vec(I), vec(V))
        return S
    end
end
