# Overview


## Getting Started

Basic usage of NFFT.jl is shown in the following example for 1D:

```jldoctest; output = false, setup = :(using NFFT, Random; Random.seed!(1))
using NFFT

M, N = 8, 16
x = range(-0.4, stop=0.4, length=M)  # nodes at which the NFFT is evaluated
fHat = randn(ComplexF64,M)           # data to be transformed
p = plan_nfft(x, N)                  # create plan. m and σ are optional parameters
f = nfft_adjoint(p, fHat)            # calculate adjoint NFFT
g = nfft(p, f)                       # calculate forward NFFT

# output

8-element Vector{ComplexF64}:
 -2.7271964441683756 + 3.8456253205787014im
 -10.845196223492461 + 28.19548180087219im
  14.036233742227378 + 1.6000061110655517im
   21.40183545854721 - 11.48566387092235im
 -13.314940379821042 - 5.046095924063817im
  -6.727871850561189 + 13.675811696584672im
   4.774803144961602 - 1.3445459029406543im
   16.24290276811969 + 4.374304822278348im
```

In 2D:

```jldoctest; output = false, setup = :(using NFFT, Random; Random.seed!(1))
M, N = 16, 32
x = rand(2, M) .- 0.5
fHat = randn(ComplexF64,M)
p = plan_nfft(x, (N,N))
f = nfft_adjoint(p, fHat)
g = nfft(p, f)

# output

16-element Vector{ComplexF64}:
   720.6895267216654 + 965.5238591742165im
 -437.33501456685804 + 342.76233641923324im
   -254.254070512323 + 443.9883021245498im
  -22.03129825160205 - 1154.6436292131266im
  -553.2097621294794 + 870.1409830364512im
  -502.9985565512272 - 345.1896574276233im
 -113.39125282787599 - 58.423751026745364im
   809.2621970089365 + 440.70931029977845im
  328.73723810616923 - 1199.2654321193654im
 -135.96497569072204 - 1231.1920040819257im
 -62.366728736105514 - 1112.539464186238im
  -1232.999473906676 - 1753.981344569756im
   670.6456551087276 + 1123.1414336887754im
  -520.6069052478605 - 69.30440960758962im
  -347.3001518675821 - 1573.5947824870434im
   654.2300268394347 + 458.56688056931813im

```

Currently, the eltype of the arguments `f` and `fHat`
must be compatible that of the variable `x` used in the `plan_nfft` call.
For example, if one wants to use `Float32` types to save memory,
then one can make the plan using something like this:

```jldoctest; output = false, setup = :(using NFFT, Random; N=(4,); Random.seed!(1))
x = Float32.(LinRange(-0.5,0.5,64))
p = plan_nfft(x, N)

# output

NFFTPlan with 64 sampling points for an input array of size(4,) and an output array of size(64,) with dims 1:1
```

The plan will then internally use `Float32` types.
Then the arguments `f` and `fHat` above should have eltype `Complex{Float32}`
or equivalently `ComplexF32`, otherwise there will be error messages.

One can also perform NFFT computations directly without first creating a plan:
```
g = nfft(x, f)
f = nfft_adjoint(x, N, fHat)
```
These forms are more forgiving about the types of the input arguments.
The versions based on a plan are more optimized for repeated use with the same `x`

## Parameters

The NFFT has the following parameters that can be passed as a keyword argument to the constructor 

| Parameter                          | Description      | Example Values        |
| :--------------------------------- | :--------------- | :------------------ |
| `m`      | Kernel size. The convolution matrix has `2m+1` non-zero entries around each sampling node in each dimension.  |  `m` $\in \{2,\dots,8\}$      |
| `σ`     | Oversampling factor. The inner FFT is of size `σN` | `σ` $\in [1.25, 2.0]$      |
| `window`   | Convolution window: Available are `:gauss`,  `:spline`, `:kaiser_bessel_rev`, `:kaiser_bessel`.    | `:kaiser_bessel` |
| `precompute`        | Flag indicating the precomputation strategy for the convolution matrix         | `LUT`      |
| `sortNodes`        | Flag if the nodes should be sorted in a lexicographic way         | `false`      |
| `storeApodizationIdx`        | Flag if the apodization indices should be stored. Currently this option is necessary on the GPU       | `false`      |
| `LUTSize`        | Size of the look up table when using `precompute == NFFT.LUT`     | `20000`      |
| `fftflags`        | flags passed to the inner `AbstractFFT` as `flags`. This can for instance be `FFTW.MEASURE` in order to optimize the inner FFT    | `FFTW.ESTIMATE`      |

In practice you can use for most of the values the default values. If you want to maximize speed you can set `σ=1.25, m=3`. If you need machine precision accuracy use  `σ=2.0, m=8`. 

## Precomputation

There are different pre-computation strategies available. Again you don't need to change this if you satisfied with the default speed. The reason to have some sort of precomputation is that the NFFT is much slower if the window function is evaluated on demand.

| Value                          | Description      | 
| :--------------------------------- | :--------------- | 
| `NFFT.LUT`      | This option uses a look-up table to first sample the window function and later use linear interpolation during the actual convolution. `LUTSize` controls the size of the look-up table. We don't have error estimates but a value of 20,000 is in practice large enough.  |  
| `NFFT.FULL`      | This option precomputes the entire convolution matrix and stores it as a `SparseMatrixCSC`. This option requires more memory an requires the most precomputation time. However, the transformation itself has maximum performance. In addition it is easy to perform the NFFT on the GPU with this option, see CuNFFT.  | 
| `NFFT.TENSOR`      | This option calculates the window on demand but exploits the tensor structure for multi-dimensional plans. Hence, this option makes no approximation but reaches a similar performance as `NFFT.LUT`. This option is right now only available in the NFFT3 backend.  | 

## Multi-Threading

Most parts of NFFT are multi-threaded when running on the CPU. To this end, start Julia with the option
```
julia -t T
```
where `T` it the number of desired threads. NFFT.jl will use all threads that are specified. You can dynamically switch off multi-threading by calling the internal function
```
  NFFT._use_threads[] = false
```
Currently, the number of threads used for the FFTW is not changed, i.e. you likely want to set
```
  FFTW.set_num_threads(Threads.nthreads())
```
The following operations are currently multi-threaded:
 * precomputation
 * apodization and its adjoint
 * FFT (using FFTW)
 * convolution

 What is currently missing is a multi-threaded version of the adjoint convolution, which is challenging to implement efficiently (needs blocking techniques). When using the `precompute = FULL` operation one could potentially exploit a sparse matrix library that supports blocking.

## Directional

There are special methods for computing 1D NFFT's for each 1D slice along a particular dimension of a higher dimensional array.

```jldoctest dirtest; output = false, setup = :(using NFFT, Random; Random.seed!(1))
M = 11

y = rand(M) .- 0.5
N = (16,20)
P1 = plan_nfft(y, N, dims=1)
f = randn(ComplexF64,N)
fHat = nfft(P1, f)

# output

11×20 Matrix{ComplexF64}:
     1.40604-0.698522im  -0.685229+2.73485im    …  -0.285721-0.454103im
     2.07978+8.13884im     1.20303+1.77911im         1.49108+2.67146im
    -0.68432-1.77989im    -3.19686-5.85921im        -6.45729+4.92725im
 -0.00523924-0.137267im  -0.412231+3.47908im        -1.84342-1.1709im
    -4.71538-0.996382im    2.08015-2.48335im        -4.50342+3.1773im
     3.97041+1.66097im     1.37838+0.0464978im  …   0.778448-2.33714im
     1.36497+3.29897im      1.2157+0.447584im        2.15635+3.35114im
    -2.24051-4.87592im     2.65738-0.546488im       -7.34172+3.19776im
    0.300783-1.40044im      1.7188-0.308963im       -1.48061+0.821495im
    -4.89909-2.60014im     5.63273-1.37166im         -2.8338+1.11567im
    0.359315-3.52236im    -3.60067-3.3054im     …  0.0717944+0.602666im
```

Here `size(f) = (16,20)` and `size(fHat) = (11,20)` since we compute an NFFT along the first dimension.
To compute the NFFT along the second dimension

```jldoctest dirtest; output = false
P2 = plan_nfft(y, N, dims=2)
fHat = nfft(P2, f)

# output

16×11 Matrix{ComplexF64}:
   -1.2312-3.45007im     6.43491-4.46477im   …   -1.88826-2.91321im
 -0.281379+1.37813im    -4.88081-0.889357im      -0.94608+2.51003im
   1.49572+3.05331im   -0.525319-3.37755im        1.43442-0.676892im
   8.00332-5.61567im     2.67785+3.48995im      -0.324491+3.55733im
   1.96454+4.89765im    -1.77692-3.68016im        4.09195+1.99918im
  -2.74732-4.93112im    -1.19153-5.33387im   …   0.649073+1.72837im
  -5.85823+2.67833im     1.91555+3.08714im       -3.72587+4.33492im
   -4.4272-1.14999im     5.80954-2.01982im        0.18063+1.87824im
  -1.83403+5.13397im    -4.28511-5.76414im       0.879507-0.137346im
   2.06692-4.06805im     1.81543+1.60781im      -0.882862+3.75418im
  -1.04477-4.80547im     -1.0337+0.599923im  …    5.50257-4.32352im
 -0.694815+2.00596im    -3.01205-4.17965im        8.24652+4.83723im
   3.78288+3.9056im    -0.710089-4.57289im       -1.25317+0.670477im
   1.24214+1.29899im    -2.48109+2.64126im        4.87461+2.83695im
   1.26369-6.68109im   -0.535623-0.938771im       1.33986+3.18496im
  -1.24022-0.748321im  -0.733792-4.42309im   …   -1.23914+5.39389im
```

Now `size(fHat) = (16,11)`.
