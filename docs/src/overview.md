# Overview


## Getting Started

Basic usage of NFFT.jl is shown in the following example for 1D:

```jldoctest; output = false, setup = :(using NFFT, StableRNGs, Random; rng = StableRNG(1); Random.seed!(rng, 1))
using NFFT

J, N = 8, 16
k = range(-0.4, stop=0.4, length=J)  # nodes at which the NFFT is evaluated
f = randn(rng, ComplexF64, J)             # data to be transformed
p = plan_nfft(k, N, reltol=1e-9)     # create plan
fHat = adjoint(p) * f                # calculate adjoint NFFT
y = p * fHat                         # calculate forward NFFT

# output

8-element Vector{ComplexF64}:
  -5.938675364233124 + 1.1882912627693385im
   9.663022355397192 - 11.305650336712825im
  -24.21472942273536 - 4.876659490931492im
  14.236061952179966 + 6.549576878806631im
   3.346939821203477 + 1.4694502375273426im
 -15.528452155546569 + 3.1543790251677892im
 -19.524258202481946 - 4.130732337002719im
  3.8673096838196326 + 3.913692808467947im
```

In the same way the 2D NFFT can be applied:

```jldoctest; output = false, setup = :(using NFFT, StableRNGs, Random; rng = StableRNG(1); Random.seed!(rng, 1))
J, N = 16, 32
k = rand(rng, 2, J) .- 0.5
f = randn(rng, ComplexF64, J)
p = plan_nfft(k, (N,N), reltol=1e-9)
fHat = adjoint(p) * f
y = p * fHat

# output

16-element Vector{ComplexF64}:
  -285.6019670104969 + 34.04948699571122im
    467.104349888175 + 976.4784038667614im
   344.3320489201338 - 628.8320016658812im
    727.674910506867 - 11.600873899657769im
  2.4546154918452974 - 604.2757905588678im
   462.6886616747879 + 236.1973007163946im
   -678.919880156888 + 867.4945371065437im
  -1284.008495786262 + 1906.412756436703im
   31.57286449371202 - 385.47358628110265im
  238.76057859533697 - 7.874207231698396im
   427.7374708284582 + 210.8254013851993im
 -308.65752741489365 - 255.75822822081753im
    93.6740916304109 + 749.8046985671293im
   41.89733838302844 - 174.37086588003345im
   615.6721908123006 + 1348.2163454701201im
  -562.1209368178222 + 109.95563778211212im

```

Currently, the eltype of the arguments `f` and `fHat`
must be compatible with that of the sampling nodes `k` used in the `plan_nfft` call.
For example, if one wants to use `Float32` types to save memory, one can create the plan like this:

```jldoctest; output = false, setup = :(using NFFT, Random; N=(4,); Random.seed!(1))
k = Float32.(LinRange(-0.5,0.5,64))
p = plan_nfft(k, N)

# output

NFFTPlan with 64 sampling points for an input array of size(4,) and an output array of size(64,) with dims 1:1
```

The plan will then internally use `Float32` types.
The signals `f` and `fHat` then need to have the eltype `Complex{Float32}`
or equivalently `ComplexF32`. Otherwise there will be error messages.

In the previous example, the output vector was allocated within the `*` method. To avoid this allocation one can use the interface
```
mul!(fHat, p, f)
mul!(y, adjoint(p), fHat)
```
which allows to pass the output vector as the first argument.

One can also perform NFFT computations directly without first creating a plan:
```
fHat = nfft(k, f)
y = nfft_adjoint(k, N, fHat)
```
These forms are more forgiving about the types of the input arguments.
The versions based on a plan are more optimized for repeated use with the same sampling nodes `k`. 
Note that `nfft_adjoint` requires the extra argument `N` since this cannot be derived from the input vector as can be done for `nfft`. 

!!! note
    The constructor `plan_nfft` is meant to be a generic factory function that can be implemented in different packages. If you want to use a concrete constructor of `NFFT.jl` use `NFFTPlan` instead. 

## Parameters

The NFFT has the several parameters that can be passed as a keyword argument to the constructor or the `nfft` and the `nfft_adjoint` function.


| Parameter                          | Description      | Example Values        |
| :--------------------------------- | :--------------- | :------------------ |
| `reltol`      | Relative tolerance that the NFFT achieves.  |  `reltol` $=10^{-9}$      |
| `m`      | Kernel size parameter. The convolution matrix has `2m+1` non-zero entries around each sampling node in each dimension.  |  `m` $\in \{2,\dots,8\}$      |
| `σ`     | Oversampling factor. The inner FFT is of size `σN` | `σ` $\in [1.25, 2.0]$      |
| `window`   | Convolution window ``\hat{\varphi}``    | `:kaiser_bessel` |
| `precompute`        | Flag indicating the precomputation strategy for the convolution matrix         | `TENSOR`      |
| `blocking`        | Flag block partitioning should be used to speed up computation        | `true`      |
| `sortNodes`        | Flag if the nodes should be sorted in a lexicographic way         | `false`      |
| `storeDeconvolutionIdx`        | Flag if the deconvolve indices should be stored. Currently this option is necessary on the GPU       | `true`      |
| `fftflags`        | flags passed to the inner `AbstractFFT` as `flags`. This can for instance be `FFTW.MEASURE` in order to optimize the inner FFT    | `FFTW.ESTIMATE`      |

In practice the default values are properly chosen and there is in most cases no need to change them. 
The only parameter you sometimes need to care about are the accuracy parameters `reltol`, `m`,`\sigma` and the `fftflags`.

## Accuracy

On a high-level it is possible to set the accuracy using the parameter `reltol`. It will automatically set the low-level parameters `m` and `\sigma`. You only need to change the later if you run into memory issues. It is important that you change only `reltol` or the pair `m`,`\sigma`.

The relation between `reltol`, `m`, and `σ` depends on the window function and the NFFT implementation. We use the formula
```math
w = 2m + 1 = \left\lceil \text{log}_{10} \frac{1}{\text{reltol}} \right\rceil + 1
```
which was verified for `σ` and the default window function. If you change the window function, you should use the parameter `m`, and `σ` instead of `reltol`.

## Window Functions

It is possible to change the internal window function ``\hat{\varphi}``. Available are 
* `:gauss`
* `:spline`
* `:kaiser_bessel_rev`
* `:kaiser_bessel` 
* `:cosh_type`
and one can easily add more by extending the [windowFunctions.jl](https://github.com/JuliaMath/NFFT.jl/blob/master/src/windowFunctions.jl) file in Github.

However, the possibility of changing the window function is only important for NFFT researcher and not for NFFT users. Right now `:kaiser_bessel` provides the best accuracy and thus there is no reason to change the parameter `window` to something different.

## Precomputation

There are different precomputation strategies available:

| Value                          | Description      | 
| :--------------------------------- | :--------------- | 
| `NFFT.POLYNOMIAL`     | This option approximates the window function by a polynomial with high degree and evaluates the polynomial during the actual convolution. |   
| `NFFT.LINEAR`      | This option uses a look-up table to first sample the window function and later use linear interpolation during the actual convolution. |  
| `NFFT.FULL`      | This option precomputes the entire convolution matrix and stores it as a `SparseMatrixCSC`. This option requires more memory and the longest precomputation time. This allows simple GPU implementations like realized in CuNFFT.  | 
| `NFFT.TENSOR`      | This option calculates the window on demand but exploits the tensor product structure for multi-dimensional plans.  | 

Again you don't need to change this parameter since the default `NFFT.POLYNOMIAL` is a good choice in most situations. You may want to use `NFFT.TENSOR` if you are applying the same transform multiple times since it is a little bit faster than `NFFT.POLYNOMIAL` but has a higher pre-computation time.

## Block Partitioning

Internally NFFT can use block partitioning to speedup computation. It helps in two ways
* It helps improving the memory efficiency by grouping sampling points together which allows for better use of CPU caches.
* Block partitioning is a mandatory to enable multi-threading in the adjoint NFFT, which would otherwise not be possible because of a data dependency.

We enable block partitioning by default since it helps also in the single-threaded case and thus, there usually is no reason to switch it off.

## Multi-Threading

Most parts of NFFT are multi-threaded when running on the CPU. To this end, start Julia with the option
```
julia -t T
```
where `T` it the number of desired threads. NFFT.jl will use all threads that are specified. 

## Directional

There are special methods for computing 1D NFFT's for each 1D slice along a particular dimension of a higher dimensional array.

```jldoctest dirtest; output = false, setup = :(using NFFT, StableRNGs, Random; rng = StableRNG(1); Random.seed!(rng, 1))
J = 11

y = rand(rng, J) .- 0.5
N = (16,20)
p1 = plan_nfft(y, N, dims=1)
f = randn(rng, ComplexF64, N)
fHat = p1 * f

# output

11×20 Matrix{ComplexF64}:
  -1.0134-2.99529im     -4.38879-3.64751im   …   -2.93426-1.16624im
 -5.46607-2.40581im     -8.74106+1.56554im       -1.65475+1.64343im
 -4.56578+1.4309im       -1.2944+3.29063im        1.60156+0.0728016im
 -3.25798+0.786092im      1.1476-0.480782im     -0.245688-2.95591im
 -1.70897+1.35211im      1.13862+0.810993im       2.00677-4.1338im
 0.570612-0.681347im    0.284008+2.99097im   …  0.0720192-2.73477im
  -4.5948+2.33407im    -0.316809+4.4821im         1.45371-0.394822im
 0.534706+0.0191701im   0.561576+0.154668im      -4.16656+1.13963im
 -3.06312-2.32161im     -7.44848+0.196475im     -0.546355+1.85535im
 0.580239-0.683206im    0.287125+2.99826im      0.0686926-2.7479im
 -1.05625+4.04557im     -1.12671-1.86346im   …   -3.94121+0.122957im
```

Here `size(f) = (16,20)` and `size(fHat) = (11,20)` since we compute an NFFT along the first dimension.
To compute the NFFT along the second dimension

```jldoctest dirtest; output = false
p2 = plan_nfft(y, N, dims=2)
fHat = p2 * f

# output

16×11 Matrix{ComplexF64}:
 -2.87485+3.30892im      4.07795-1.14929im   …    6.08996+1.30273im
 -2.19738-0.269658im     1.04127+4.46635im       -2.78789+3.06132im
  3.24948-5.61056im      2.18916+2.86387im        2.31745-1.66995im
 -5.06468+4.11595im      1.36344-0.339266im       1.26474+2.19545im
  2.23309+2.32763im     -2.74011+2.67914im       -1.04576+0.276069im
  4.91439-1.15096im      4.94442-3.39179im   …  -0.421113+5.04906im
  8.13457-0.832435im     4.69688+2.21479im        -3.5927-4.2766im
 -2.21456-2.1025im       1.23604-2.51297im       0.550327-1.76364im
 0.746511-0.155959im    -2.93264+3.96379im        2.92887+4.28923im
 -2.20299+1.31202im   -0.0703468+1.58074im        2.07026+1.07033im
 -6.22334-3.57084im     -3.43678-1.43784im   …  -0.061879+3.42923im
  2.00108+0.502242im     1.64709+3.63358im       0.414365+0.3579im
 0.479841+3.42326im     0.196046+0.115921im     -0.760431+2.52307im
  -3.3883+3.40592im    -0.873708+3.4237im         2.94429-2.95127im
 -3.78246-1.59859im      1.74533-0.944574im      -1.42798+0.0727237im
 -1.03499+1.32023im      2.27659+0.651444im  …    2.85672+2.31098im
```

Now `size(fHat) = (16,11)`.
