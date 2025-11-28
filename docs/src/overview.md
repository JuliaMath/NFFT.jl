# Overview


## Getting Started

Basic usage of NFFT.jl is shown in the following example for 1D:

```jldoctest; output = false, setup = :(using NFFT, Random; Random.seed!(1))
using NFFT

J, N = 8, 16
k = range(-0.4, stop=0.4, length=J)  # nodes at which the NFFT is evaluated
f = randn(ComplexF64, J)             # data to be transformed
p = plan_nfft(k, N, reltol=1e-9)     # create plan
fHat = adjoint(p) * f                # calculate adjoint NFFT
y = p * fHat                         # calculate forward NFFT

# output

8-element Vector{ComplexF64}:
 -2.7271960750455864 + 3.845624459230172im
 -10.845195849921893 + 28.195480756172707im
  14.036233667047414 + 1.6000052060038368im
   21.40183412157684 - 11.485663942028697im
 -13.314941250713128 - 5.046096728579452im
  -6.727872247485433 + 13.675812003726714im
     4.7748026670029 - 1.3445454002024781im
  16.242902197461223 + 4.374305558242007im
```

In the same way the 2D NFFT can be applied:

```jldoctest; output = false, setup = :(using NFFT, Random; Random.seed!(1))
J, N = 16, 32
k = rand(2, J) .- 0.5
f = randn(ComplexF64, J)
p = plan_nfft(k, (N,N), reltol=1e-9)
fHat = adjoint(p) * f
y = p * fHat

# output

16-element Vector{ComplexF64}:
   720.6895322120488 + 965.5238823533357im
 -437.33501829029217 + 342.76233801511705im
 -254.25406987853404 + 443.98830703906657im
 -22.031291597632734 - 1154.6436407707442im
  -553.2097750978456 + 870.141009246982im
 -502.99856721175956 - 345.1896840492251im
 -113.39126060291248 - 58.42376028614012im
   809.2621963784361 + 440.7093087192107im
  328.73724327772777 - 1199.2654551418184im
 -135.96498544574132 - 1231.1920536402204im
 -62.366745207088044 - 1112.539501456509im
 -1232.9994883040201 - 1753.9813744814114im
   670.6456652742794 + 1123.1414479652183im
   -520.606923873964 - 69.30438918637125im
  -347.3001526664351 - 1573.5948177902883im
   654.2300617098213 + 458.56688308083926im

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

```jldoctest dirtest; output = false, setup = :(using NFFT, Random; Random.seed!(1))
J = 11

y = rand(J) .- 0.5
N = (16,20)
p1 = plan_nfft(y, N, dims=1)
f = randn(ComplexF64, N)
fHat = p1 * f

# output

11×20 Matrix{ComplexF64}:
     1.40604-0.698522im  -0.685229+2.73485im   …  -0.285721-0.454103im
     2.07978+8.13884im     1.20303+1.77911im        1.49108+2.67146im
    -0.68432-1.77989im    -3.19686-5.85921im       -6.45729+4.92725im
 -0.00523926-0.137267im  -0.412231+3.47908im       -1.84342-1.1709im
    -4.71538-0.996383im    2.08015-2.48335im       -4.50342+3.1773im
     3.97041+1.66097im     1.37838+0.046498im  …   0.778448-2.33714im
     1.36497+3.29897im      1.2157+0.447584im       2.15635+3.35114im
    -2.24051-4.87592im     2.65738-0.546488im      -7.34172+3.19776im
    0.300783-1.40044im      1.7188-0.308963im      -1.48061+0.821495im
    -4.89909-2.60014im     5.63273-1.37166im        -2.8338+1.11567im
    0.359316-3.52236im    -3.60067-3.3054im    …  0.0717944+0.602666im
```

Here `size(f) = (16,20)` and `size(fHat) = (11,20)` since we compute an NFFT along the first dimension.
To compute the NFFT along the second dimension

```jldoctest dirtest; output = false
p2 = plan_nfft(y, N, dims=2)
fHat = p2 * f

# output

16×11 Matrix{ComplexF64}:
   -1.2312-3.45007im     6.43491-4.46477im   …   -1.88826-2.91321im
 -0.281379+1.37813im    -4.88081-0.889356im      -0.94608+2.51003im
   1.49572+3.05331im   -0.525319-3.37755im        1.43442-0.676892im
   8.00332-5.61567im     2.67785+3.48995im      -0.324491+3.55733im
   1.96454+4.89765im    -1.77692-3.68016im        4.09195+1.99918im
  -2.74732-4.93112im    -1.19153-5.33387im   …   0.649073+1.72837im
  -5.85823+2.67833im     1.91555+3.08714im       -3.72587+4.33492im
   -4.4272-1.14999im     5.80954-2.01982im        0.18063+1.87824im
  -1.83403+5.13397im    -4.28511-5.76414im       0.879507-0.137346im
   2.06692-4.06805im     1.81543+1.60781im      -0.882862+3.75418im
 -0.840898-4.7404im    -0.876844+0.745518im  …     5.3795-4.14843im
 -0.694815+2.00596im    -3.01205-4.17965im        8.24652+4.83723im
   3.78288+3.9056im    -0.710089-4.57289im       -1.25317+0.670477im
   1.24214+1.29899im    -2.48109+2.64126im        4.87461+2.83695im
   1.26369-6.68109im   -0.535623-0.938771im       1.33986+3.18496im
  -1.24022-0.748321im  -0.733792-4.42309im   …   -1.23914+5.39389im
```

Now `size(fHat) = (16,11)`.
