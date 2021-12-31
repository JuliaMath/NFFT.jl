## ################################################################
# constructors
###################################################################

"""
    calculateToeplitzKernel(shape, tr::Matrix{T}[, m = 4, sigma = 2.0, window = :kaiser_bessel, K = 2000; fftplan = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE), kwargs...])

Calculate the kernel for an implementation of the Gram matrix that utilizes its Toeplitz structure. The output is an array of twice the size of `shape`, as the Toeplitz trick requires an oversampling factor of 2 (cf. [Wajer, F. T. A. W., and K. P. Pruessmann. "Major speedup of reconstruction for sensitivity encoding with arbitrary trajectories." Proc. Intl. Soc. Mag. Res. Med. 2001.](https://cds.ismrm.org/ismrm-2001/PDF3/0767.pdf)). The type of the kernel is `Complex{T}`, i.e. the complex of the k-space trajectory's type; for speed and memory efficiecy, call this function with `Float32.(tr)`, and the kernel will also be `Float32`.

# Required Arguments
- `shape::NTuple(Int)`: size of the image; e.g. `(256, 256)` for 2D imaging, or `(256,256,128)` for 3D imaging
- `tr::Matrix{T}`: non-Cartesian k-space trajectory in units revolutions/voxel, i.e. `tr[i] ∈ [-0.5, 0.5] ∀ i`. The matrix has the size `2 x Nsamples` for 2D imaging with a trajectory length `Nsamples`, and `3 x Nsamples` for 3D imaging.

# Optional Arguments:
- `m::Int`: nfft kernel size (used to calculate the Toeplitz kernel); `default = 4`
- `sigma::Number`: nfft oversampling factor during the calculation of the Toeplitz kernel; `default = 2`
- `window::Symbol`: Window function of the nfft (c.f. [`getWindow`](@ref)); `default = :kaiser_bessel`
- `K::Int`: `default= 2000` # TODO: describe meaning of k

# Keyword Arguments:
- `fftplan`: plan for the final FFT of the Kernal from image to k-space. Therefore, it has to have twice the size of the original image. `default = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE)`. If this constructor is used many times, it is worth to precompute the plan with `plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.MEASURE)` and reuse it.
- `kwargs`: Passed on to [`plan_fft!`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.plan_fft!) via `NFFTPlan`; can be used to modify the flags `flags=FFTW.ESTIMATE, timelimit=Inf`.

# Examples
```jldoctest
julia> Nx = 32;

julia> trj = Float32.(rand(2, 1000) .- 0.5);

julia> λ = calculateToeplitzKernel((Nx, Nx), trj)
64×64 Matrix{ComplexF32}:
  1697.69+483.218im   -147.57-257.719im  …   604.041-479.841im
  1858.23-220.871im   3723.91-4.62838im      3917.46+217.494im
  3260.49+249.208im   1719.05-23.7088im      1935.48-245.831im
  1115.56-210.368im  -85.7251-15.1312im      499.092+206.991im
  3232.19+666.149im  -927.182-440.649im      1048.98-662.772im
 -2818.12-640.593im   2892.04+415.094im  …   -435.71+637.216im
  3019.67+377.944im   1978.27-152.445im      4267.19-374.567im
  711.755-499.0im     11091.8+273.5im        822.519+495.623im
 -436.376+576.158im   2085.76-350.659im      280.188-572.781im
 -460.382-658.799im  -402.648+433.299im      1579.94+655.422im
         ⋮                               ⋱
  863.552-940.384im  -571.564+714.885im  …   325.992+937.007im
  -554.64+283.37im   -416.581-57.8705im     -295.217-279.993im
  680.462-381.469im   437.636+155.97im       381.239+378.092im
 -1769.07+404.678im   291.916-179.179im      192.257-401.301im
  1502.99-212.153im   3114.76-13.346im       3939.76+208.776im
  1192.88+274.105im   728.321-48.6058im  …   593.152-270.728im
  318.763-370.158im  -273.715+144.659im     -323.591+366.781im
 -810.074+257.924im  -144.068-32.4241im      -44.218-254.546im
  6650.49-187.076im  -1154.77-38.4236im      1493.98+183.699im

julia> x = randn(ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(x, λ)
32×32 Matrix{ComplexF32}:
 -2548.17+350.061im  -83.5555-1118.35im  …  -369.412+1283.74im
 -232.842+353.766im   950.106+2080.18im     -598.713-944.801im
 -1009.52-1121.52im    93.405+693.621im      1390.72+197.977im
  1812.78-9.56223im   275.549+599.843im      458.824-1033.59im
 -1213.11-163.169im  -329.302-1185.4im       356.961-1026.23im
  1479.05-1515.57im   1149.94-1208.13im  …   410.585+209.635im
  1021.91+1114.42im   571.518+509.203im     -1047.46+647.597im
 -12.9759+900.257im   880.053+758.609im     -1017.49-1609.38im
  398.851+171.48im     830.65+1170.5im       1277.94-2639.44im
  188.979+789.158im   415.594+370.381im       101.68-1864.72im
         ⋮                               ⋱
  -1458.0-1250.08im   2466.35+374.447im     -352.156+306.172im
  828.511-333.487im   2105.72+344.53im      -401.738-782.178im
 -602.072-108.482im  -1368.96+1090.78im  …  -129.634-1951.61im
  169.226-1887.22im   1637.13-460.677im      1077.74+775.838im
  593.828+1301.49im  -1322.09+194.759im     -1700.84+531.625im
  775.017-208.648im    1361.0-528.349im     -434.659+206.32im
  1044.89-874.581im  -286.478+795.039im     -1186.11+879.18im
 -665.938+1052.24im   -34.669-1059.52im  …   -1700.0-1452.94im
 -303.172-1110.12im   2510.89+163.711im     -500.883-1606.86im

```
"""
function calculateToeplitzKernel(shape, tr::Matrix{T}, m = 4, sigma = 2.0, window = :kaiser_bessel, K = 2000; fftplan = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE), kwargs...) where {T}

    shape_os = 2 .* shape

    p = NFFTPlan(tr, shape_os, m, sigma, window, K; kwargs...)
    eigMat = nfft_adjoint(p, ones(Complex{T}, size(tr,2)))
    return fftplan * fftshift(eigMat)
end

"""
    calculateToeplitzKernel!(f::Array{Complex{T}}, p::AbstractNFFTPlan, tr::Matrix{T}, fftplan)

Calculate the kernel for an implementation of the Gram matrix that utilizes its Toeplitz structure and writes it in-place in `f`, which has to be twice the size of the desired image matrix, as the Toeplitz trick requires an oversampling factor of 2 (cf. [Wajer, F. T. A. W., and K. P. Pruessmann. "Major speedup of reconstruction for sensitivity encoding with arbitrary trajectories." Proc. Intl. Soc. Mag. Res. Med. 2001.](https://cds.ismrm.org/ismrm-2001/PDF3/0767.pdf)). The type of the kernel `f` has to be `Complex{T}`, i.e. the complex of the k-space trajectory's type; for speed and memory efficiecy, call this function with `Float32.(tr)`, and set the type of `f` accordingly.

# Required Arguments
- `f::Array{T}`: Array in which the kernel will be written.
- `p::AbstractNFFTPlan`: NFFTPlan with the same dimentions as `tr`, which will be overwritten in place.
- `tr::Matrix{T}`: non-Cartesian k-space trajectory in units revolutions/voxel, i.e. `tr[i] ∈ [-0.5, 0.5] ∀ i`. The matrix has the size `2 x Nsamples` for 2D imaging with a trajectory length `Nsamples`, and `3 x Nsamples` for 3D imaging.
- `fftplan`: plan for the final FFT of the Kernal from image to k-space. Therefore, it has to have twice the size of the original image. Calculate, e.g., with `fftplan = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.MEASURE)`, where `shape` is the size of the reconstructed image.

# Examples
```jldoctest
julia> using FFTW

julia> Nx = 32;

julia> trj = Float32.(rand(2, 1000) .- 0.5);

julia> p = plan_nfft(trj, (2Nx,2Nx))
NFFTPlan with 1000 sampling points for (64, 64) array

julia> fftplan = plan_fft(zeros(ComplexF32, (2Nx,2Nx)); flags=FFTW.MEASURE);

julia> λ = Array{ComplexF32}(undef, 2Nx, 2Nx);

julia> calculateToeplitzKernel!(λ, p, trj, fftplan);

julia> x = randn(ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(x, λ);

```
"""
function calculateToeplitzKernel!(f::Array{Complex{T}}, p::AbstractNFFTPlan, tr::Matrix{T}, fftplan) where T
    NFFTPlan!(p, tr)
    f = nfft_adjoint!(p, OnesVector(Complex{T}, size(tr,2)), f)
    f2 = fftshift(f)
    mul!(f, fftplan, f2)
    return f
end


## ################################################################
# constructor for explicit/exact calculation of the Toeplitz kernel
# (slow)
###################################################################
function calculateToeplitzKernel_explicit(shape, tr::Matrix{T}, fftplan = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE)) where T

    shape_os = 2 .* shape
    λ = Array{Complex{T}}(undef, shape_os)
    Threads.@threads for i ∈ CartesianIndices(λ)
        λ[i] = getMatrixElement(i, shape_os, tr)
    end
    return fftplan * fftshift(λ)
end

function getMatrixElement(idx, shape::Tuple, nodes::Matrix{T}) where T
    elem = zero(Complex{T})
    shape = T.(shape) # ensures the correct output type

    @fastmath @simd for i ∈ eachindex(view(nodes, 1, :))
        ϕ = zero(T)
        @inbounds for j ∈ eachindex(shape)
            ϕ += nodes[j,i] * (shape[j]/2 + 1 - idx[j])
        end
        elem += exp(-2im * π * ϕ)
    end
    return elem
end

## ################################################################
# apply Toeplitz kernel
###################################################################

"""
    convolveToeplitzKernel!(x::Array{T,N}, λ::Array{T,N}[, fftplan = plan_fft(λ; flags=FFTW.ESTIMATE), ifftplan = plan_ifft(λ; flags=FFTW.ESTIMATE), xOS1 = similar(λ), xOS2 = similar(λ)])

Convolves the image `x` with the Toeplitz kernel `λ` and overwrites `x` with the result. `x` is also returned for convenience. As this function is commonly applied many times, it is highly recommened to pre-allocate / pre-compute all optional arguments. By doing so, this entire function is non-allocating.

# Required Arguments
- `x::Array{T,N}`: Input image that will be overwritten with the result. `x` is a matrix (`N=2`) for 2D imaging and a 3D tensor (`N=3`) for 3D imaging. The type of the elments `T` must match the ones of `λ`.
- `λ::Array{T,N}`: Toeplitz kernel, which as to be the same type as `x`, but twice the size due to the required oversampling (cf. [`calculateToeplitzKernel`](@ref)).

# Optional, but highly recommended Arguments
- `fftplan`: plan for the oversampled FFT, i.e. it has to have twice the size of the original image. Calculate, e.g., with `fftplan = plan_fft(λ; flags=FFTW.MEASURE)`, where `shape` is the size of the reconstructed image.
- `ifftplan`: plan for the oversampled inverse FFT. Calculate, e.g., with `ifftplan = plan_ifft(λ; flags=FFTW.MEASURE)`, where `shape` is the size of the reconstructed image.
- `xOS1`: pre-allocated array of the size of `λ`. Pre-allocate with `xOS1 = similar(λ)`.
- `xOS2`: pre-allocated array of the size of `λ`. Pre-allocate with `xOS2 = similar(λ)`.

# Examples
```jldoctest
julia> using FFTW

julia> Nx = 32;

julia> trj = Float32.(rand(2, 1000) .- 0.5);

julia> λ = calculateToeplitzKernel((Nx, Nx), trj);

julia> xOS1 = similar(λ);

julia> xOS2 = similar(λ);

julia> fftplan = plan_fft(xOS1; flags=FFTW.MEASURE);

julia> ifftplan = plan_ifft(xOS1; flags=FFTW.MEASURE);

julia> x = randn(ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(x, λ, fftplan, ifftplan, xOS1, xOS2)
32×32 Matrix{ComplexF32}:
  -1803.8+1.63312im  -1561.82-752.754im  …  -647.761-229.904im
  1114.51+41.7063im  -264.876-662.938im      401.956+968.49im
 -1380.27-315.122im   33.9136+905.85im       -1109.6+698.732im
   505.66-254.45im   -1687.54+370.488im     -195.682+412.91im
  406.557+314.59im    1861.73+506.391im      1048.28+647.188im
 -1048.58+36.8265im   27.7886+439.139im  …   526.469+1575.94im
 -291.241+786.415im   565.874+231.098im      713.937-1226.66im
  177.944+926.04im    873.483+42.0362im      88.2286-748.885im
  160.151-510.168im  -1039.33-1249.35im      1468.72-459.773im
 -857.312-441.418im   1077.05+1000.54im     -774.294-1822.92im
         ⋮                               ⋱
 -801.172+508.445im   497.832+1877.24im     -267.744-770.01im
 -513.894+153.35im    855.736+199.721im      1342.16+121.227im
  1125.26-863.306im   753.793+1884.79im  …  -401.419+1074.31im
   900.01+153.858im  -188.444-534.829im      1220.38-557.301im
   175.36-1445.92im   1968.61-3329.79im     -715.639-378.965im
  292.267+195.924im  -184.499-858.438im     -1513.19-513.17im
  477.721-1154.86im   156.387+1108.32im      8.60736-123.34im
  653.422+427.798im  -31.5388+570.621im  …  -871.204+89.9642im
   1441.1+194.813im   889.465-225.068im      1248.81+1110.32im

```
"""
function convolveToeplitzKernel!(x::Array{T,N}, λ::Array{T,N},
    fftplan = plan_fft(λ; flags=FFTW.ESTIMATE),
    ifftplan = plan_ifft(λ; flags=FFTW.ESTIMATE),
    xOS1 = similar(λ),
    xOS2 = similar(λ)
    ) where {T,N}

    fill!(xOS1, 0)
    xOS1[CartesianIndices(x)] .= x
    mul!(xOS2, fftplan, xOS1)
    xOS2 .*= λ
    mul!(xOS1, ifftplan, xOS2)
    x .= @view xOS1[CartesianIndices(x)]
    return x
end


## ################################################################
# helper class
###################################################################
struct OnesVector{T} <: AbstractVector{T}
    elements::T
    length::Int
end

function OnesVector(T::Type, length::Int)
    return OnesVector(one(T), length)
end

function Base.size(A::OnesVector)
    return (A.length,)
end

function Base.length(A::OnesVector)
    return A.length
end

function Base.getindex(A::OnesVector, i::Int)
    return A.elements
end