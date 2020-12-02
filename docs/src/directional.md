# Directional NFFT

There are special methods for computing 1D NFFT's for each 1D slice along a particular dimension of a higher dimensional array.

```julia
M = 11
y = rand(M) .- 0.5
N = (16,20)
P1 = plan_nfft(y, 1, N)
f = randn(ComplexF64,N)
fHat = nfft(P1, f)
```

Here `size(f) = (16,20)` and `size(fHat) = (11,20)` since we compute an NFFT along the first dimension.
To compute the NFFT along the second dimension

```julia
P2 = plan_nfft(y, 2, N)
fHat = nfft(P2, f)
```

Now `size(fHat) = (16,11)`.
