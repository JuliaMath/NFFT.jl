## ################################################################
# constructors
###################################################################

"""
    calculateToeplitzKernel(shape, tr::Matrix{T}[; m = 4, σ = 2.0, window = :kaiser_bessel, LUTSize = 2000, fftplan = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE), kwargs...])

Calculate the kernel for an implementation of the Gram matrix that utilizes its Toeplitz structure. The output is an array of twice the size of `shape`, as the Toeplitz trick requires an oversampling factor of 2 (cf. [Wajer, F. T. A. W., and K. P. Pruessmann. "Major speedup of reconstruction for sensitivity encoding with arbitrary trajectories." Proc. Intl. Soc. Mag. Res. Med. 2001.](https://cds.ismrm.org/ismrm-2001/PDF3/0767.pdf)). The type of the kernel is `Complex{T}`, i.e. the complex of the k-space trajectory's type; for speed and memory efficiecy, call this function with `Float32.(tr)`, and the kernel will also be `Float32`.

# Required Arguments
- `shape::NTuple(Int)`: size of the image; e.g. `(256, 256)` for 2D imaging, or `(256,256,128)` for 3D imaging
- `tr::Matrix{T}`: non-Cartesian k-space trajectory in units revolutions/voxel, i.e. `tr[i] ∈ [-0.5, 0.5] ∀ i`. The matrix has the size `2 x Nsamples` for 2D imaging with a trajectory length `Nsamples`, and `3 x Nsamples` for 3D imaging.

# Optional Arguments:
- `m::Int`: nfft kernel size (used to calculate the Toeplitz kernel); `default = 4`
- `σ::Number`: nfft oversampling factor during the calculation of the Toeplitz kernel; `default = 2`
- `window::Symbol`: Window function of the nfft (c.f. [`getWindow`](@ref)); `default = :kaiser_bessel`
- `K::Int`: `default= 2000` # TODO: describe meaning of k

# Keyword Arguments:
- `fftplan`: plan for the final FFT of the Kernal from image to k-space. Therefore, it has to have twice the size of the original image. `default = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE)`. If this constructor is used many times, it is worth to precompute the plan with `plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.MEASURE)` and reuse it.
- `kwargs`: Passed on to [`plan_fft!`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.plan_fft!) via `NFFTPlan`; can be used to modify the flags `flags=FFTW.ESTIMATE, timelimit=Inf`.

# Examples
```jldoctest; output = false, setup = :(using NFFT, Random; Random.seed!(1))
julia> Nx = 32;

julia> trj = Float32.(rand(2, 1000) .- 0.5);

julia> λ = calculateToeplitzKernel((Nx, Nx), trj)
64×64 Matrix{ComplexF32}:
 -432.209+297.833im    271.68-526.348im  …   3044.35+34.6322im
  990.057-244.569im  -582.825+473.084im      45.7631-87.8965im
  1922.01+223.525im   9248.63-452.04im       3649.78+108.941im
 -402.371-3.37619im   496.815+231.891im      322.646-329.089im
 -482.346-121.534im   559.756-106.981im      155.183+454.0im
   3293.8+194.388im  -3361.43+34.1272im  …   2672.16-526.853im
 -936.331+172.246im   4394.11-400.762im     -1121.04+160.219im
 -135.828+169.448im   3509.17+59.0678im      3883.84-501.913im
  395.143+158.638im   24.4377-387.153im       5731.3+173.827im
  925.902-117.765im    2935.3+346.28im      -1414.12-214.701im
         ⋮                               ⋱
  2239.72-295.883im   490.442+524.399im  …   2028.84-36.5824im
 -1108.11+227.146im   24.7403-455.661im     -549.699+105.319im
  1323.78+110.713im  -321.052+117.802im      651.944-443.178im
 -52.1597+288.0im    -326.042-516.516im       3619.1+44.4651im
  1180.56+73.3666im  -26.1233+155.148im     -869.065-405.832im
  3555.36+649.527im  -198.245-878.042im  …   1198.83-317.062im
 -368.958-177.954im  -360.343+406.469im     -1478.96-154.512im
  4861.29+38.9623im   6082.55-267.478im      2519.09+293.503im
  1022.55-185.869im   177.426+414.384im      3650.56-146.597im

julia> x = randn(ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(x, λ)
32×32 Matrix{ComplexF32}:
  177.717-52.0493im   10.6059+20.7185im  …   746.131+330.005im
 -311.858+988.219im  -1216.83-1295.14im      410.732+751.925im
 -1082.27+872.968im   1023.97-1556.45im     -923.699+478.63im
 -999.035-525.161im  -1694.59-658.558im     -521.044+607.005im
 -342.999+1481.82im  -1729.18-2712.81im      56.5212+1394.81im
 -1187.65-1979.55im   1389.67+970.033im  …   2968.93-744.264im
 -666.533+485.511im  -1315.83+409.855im     -610.146+132.258im
  1159.57+64.6059im  -299.169-569.622im      663.802+396.827im
 -795.112-1464.63im   -462.43+2442.77im     -1622.72+1701.27im
 -1047.67+31.5578im  -1127.65-936.043im      474.071+797.911im
         ⋮                               ⋱
  327.345+2191.71im  -904.635+558.786im     -1449.32-276.721im
  -1047.2-71.362im    363.109+567.346im      -1974.0-2348.36im
 -1540.45+1661.77im   1175.75+1279.75im  …   1110.61+653.234im
 -526.832-435.297im  -265.021-2019.08im      68.5607-323.086im
 -1076.52-2719.16im  -477.005+2232.06im      -155.59-1275.66im
  1143.76-735.966im  -380.489+2485.78im      1812.17-261.191im
  1685.44-1243.29im   1911.16-1157.72im     -991.639-42.8214im
  1054.11-1282.41im   66.9358-588.991im  …  -952.238+1026.35im
 -417.276-273.367im  -946.698+1971.77im     -890.339-882.05im

```
"""
function calculateToeplitzKernel(shape, tr::AbstractMatrix{T}; m = 4, σ = 2.0, window = :kaiser_bessel, LUTSize = 2000, fftplan = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE), kwargs...) where {T}

    shape_os = 2 .* shape

    p = plan_nfft(typeof(tr), tr, shape_os; m, σ, window, LUTSize, kwargs...)
    eigMat = nfft_adjoint(p, OnesVector(Complex{T}, size(tr,2)))
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
```jldoctest; output = false, setup = :(using NFFT, Random; Random.seed!(1))
julia> using FFTW

julia> Nx = 32;

julia> trj = Float32.(rand(2, 1000) .- 0.5);

julia> p = plan_nfft(trj, (2Nx,2Nx))
NFFTPlan with 1000 sampling points for an input array of size(64, 64) and an output array of size(1000,) with dims 1:2

julia> fftplan = plan_fft(zeros(ComplexF32, (2Nx,2Nx)); flags=FFTW.MEASURE);

julia> λ = Array{ComplexF32}(undef, 2Nx, 2Nx);

julia> calculateToeplitzKernel!(λ, p, trj, fftplan);

julia> x = randn(ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(x, λ);

```
"""
function calculateToeplitzKernel!(f::Array{Complex{T}}, p::AbstractNFFTPlan{T}, tr::Matrix{T}, fftplan) where T
    nodes!(p, tr)
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
```jldoctest; output = false, setup = :(using NFFT, Random; Random.seed!(1))
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
  177.717-52.0493im   10.6059+20.7185im  …   746.131+330.005im
 -311.858+988.219im  -1216.83-1295.14im      410.732+751.925im
 -1082.27+872.968im   1023.97-1556.45im     -923.699+478.63im
 -999.035-525.161im  -1694.59-658.558im     -521.044+607.005im
 -342.999+1481.82im  -1729.18-2712.81im      56.5212+1394.81im
 -1187.65-1979.55im   1389.67+970.033im  …   2968.93-744.264im
 -666.533+485.511im  -1315.83+409.855im     -610.146+132.258im
  1159.57+64.6059im  -299.169-569.622im      663.802+396.827im
 -795.112-1464.63im   -462.43+2442.77im     -1622.72+1701.27im
 -1047.67+31.5578im  -1127.65-936.043im      474.071+797.911im
         ⋮                               ⋱
  327.345+2191.71im  -904.635+558.786im     -1449.32-276.721im
  -1047.2-71.362im    363.109+567.346im      -1974.0-2348.36im
 -1540.45+1661.77im   1175.75+1279.75im  …   1110.61+653.234im
 -526.832-435.297im  -265.021-2019.08im      68.5607-323.086im
 -1076.52-2719.16im  -477.005+2232.06im      -155.59-1275.66im
  1143.76-735.966im  -380.489+2485.78im      1812.17-261.191im
  1685.44-1243.29im   1911.16-1157.72im     -991.639-42.8214im
  1054.11-1282.41im   66.9358-588.991im  …  -952.238+1026.35im
 -417.276-273.367im  -946.698+1971.77im     -890.339-882.05im

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