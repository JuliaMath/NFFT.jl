###################################################################
# constructors
###################################################################

"""
    calculateToeplitzKernel(shape, tr::Matrix{T}[; m = 4, σ = 2.0, window = :kaiser_bessel, fftplan = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE), kwargs...])

Calculate the kernel for an implementation of the Gram matrix that utilizes its Toeplitz structure. The output is an array of twice the size of `shape`, as the Toeplitz trick requires an oversampling factor of 2 (cf. [Wajer, F. T. A. W., and K. P. Pruessmann. "Major speedup of reconstruction for sensitivity encoding with arbitrary trajectories." Proc. Intl. Soc. Mag. Res. Med. 2001.](https://cds.ismrm.org/ismrm-2001/PDF3/0767.pdf)). The type of the kernel is `Complex{T}`, i.e. the complex of the k-space trajectory's type; for speed and memory efficiency, call this function with `Float32.(tr)`, and the kernel will also be `Float32`.

# Required Arguments
- `shape::NTuple(Int)`: size of the image; e.g. `(256, 256)` for 2D imaging, or `(256,256,128)` for 3D imaging
- `tr::Matrix{T}`: non-Cartesian k-space trajectory in units revolutions/voxel, i.e. `tr[i] ∈ [-0.5, 0.5] ∀ i`. The matrix has the size `2 x Nsamples` for 2D imaging with a trajectory length `Nsamples`, and `3 x Nsamples` for 3D imaging.

# Optional Arguments:
- `m::Int`: nfft kernel size (used to calculate the Toeplitz kernel); `default = 4`
- `σ::Number`: nfft oversampling factor during the calculation of the Toeplitz kernel; `default = 2`

# Keyword Arguments:
- `fftplan`: plan for the final FFT of the kernel from image to k-space. Therefore, it has to have twice the size of the original image. `default = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE)`. If this constructor is used many times, it is worth to precompute the plan with `plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.MEASURE)` and reuse it.
- `kwargs`: Passed on to [`plan_fft!`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.plan_fft!) via `NFFTPlan`; can be used to modify the flags `flags=FFTW.ESTIMATE, timelimit=Inf`.

# Examples
```jldoctest; output = false, setup = :(using NFFT, NFFTTools, Random; Random.seed!(1))
julia> Nx = 32;

julia> trj = Float32.(rand(2, 1000) .- 0.5);

julia> λ = calculateToeplitzKernel((Nx, Nx), trj)
64×64 Matrix{ComplexF32}:
 -432.217+297.834im   271.685-526.349im  …   3044.35+34.632im
   990.06-244.569im  -582.829+473.084im      45.7639-87.8963im
   1922.0+223.525im   9248.64-452.04im       3649.78+108.941im
  -402.37-3.37625im   496.812+231.891im      322.647-329.09im
 -482.348-121.535im   559.759-106.98im       155.184+454.001im
   3293.8+194.388im  -3361.44+34.1273im  …   2672.17-526.854im
 -936.332+172.248im   4394.11-400.763im     -1121.05+160.218im
 -135.829+169.448im   3509.17+59.0675im      3883.84-501.914im
  395.145+158.639im    24.436-387.153im       5731.3+173.827im
  925.902-117.765im    2935.3+346.28im      -1414.13-214.702im
         ⋮                               ⋱
  2239.73-295.885im   490.443+524.4im    …   2028.84-36.5809im
 -1108.11+227.146im    24.741-455.661im     -549.697+105.32im
  1323.79+110.713im  -321.055+117.802im      651.944-443.179im
 -52.1619+288.001im   -326.04-516.516im       3619.1+44.4653im
  1180.55+73.367im   -26.1242+155.148im     -869.065-405.833im
  3555.37+649.53im   -198.253-878.045im  …   1198.84-317.064im
 -368.961-177.954im  -360.347+406.47im      -1478.97-154.512im
  4861.29+38.9627im   6082.57-267.477im      2519.09+293.503im
  1022.55-185.869im   177.419+414.384im      3650.56-146.597im

julia> y = randn(ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(y, λ)
32×32 Matrix{ComplexF32}:
  178.169-48.6707im   12.9519+21.8054im  …    749.91+324.029im
 -317.004+988.44im   -1214.49-1299.27im      410.381+747.908im
 -1087.78+870.916im   1028.41-1556.78im     -920.652+477.518im
  -1007.2-526.453im   -1691.3-663.165im     -524.921+600.5im
 -348.506+1479.8im    -1725.2-2714.59im      51.0908+1390.72im
 -1180.14-1977.12im   1391.15+969.678im  …   2966.56-737.661im
 -666.762+478.188im  -1309.84+404.228im     -607.705+129.812im
  1154.85+65.4261im  -301.903-570.698im      661.818+395.093im
 -786.583-1455.54im  -465.185+2436.7im      -1623.73+1694.6im
 -1045.34+28.3079im  -1123.31-930.451im      469.764+799.731im
         ⋮                               ⋱
  324.714+2197.86im   -909.22+565.227im     -1453.51-278.173im
 -1049.93-74.558im    360.753+577.881im      -1977.2-2348.59im
 -1542.49+1658.0im    1167.51+1285.3im   …   1107.94+651.439im
 -528.651-435.216im  -274.957-2013.84im       63.009-327.94im
 -1070.11-2721.88im  -480.178+2228.54im     -160.381-1271.39im
  1143.94-738.802im  -390.363+2484.48im       1816.4-261.432im
  1683.39-1242.72im   1905.95-1150.32im     -990.915-47.443im
  1047.94-1277.64im   66.1227-586.225im  …  -960.649+1029.82im
 -419.524-274.19im   -938.769+1969.06im     -890.385-878.783im

```
"""
function calculateToeplitzKernel(shape, tr::AbstractMatrix{T}; m = 4, σ = 2.0, window = :kaiser_bessel, fftplan = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE), kwargs...) where {T}

    shape_os = 2 .* shape

    p = plan_nfft(typeof(tr), tr, shape_os; m, σ, window, kwargs...)
    eigMat = adjoint(p) * OnesVector(Complex{T}, size(tr,2))
    return fftplan * fftshift(eigMat)
end

"""
    calculateToeplitzKernel!(f::Array{Complex{T}}, p::AbstractNFFTPlan, tr::Matrix{T}, fftplan)

Calculate the kernel for an implementation of the Gram matrix that utilizes its Toeplitz structure and writes it in-place in `f`, which has to be twice the size of the desired image matrix, as the Toeplitz trick requires an oversampling factor of 2 (cf. [Wajer, F. T. A. W., and K. P. Pruessmann. "Major speedup of reconstruction for sensitivity encoding with arbitrary trajectories." Proc. Intl. Soc. Mag. Res. Med. 2001.](https://cds.ismrm.org/ismrm-2001/PDF3/0767.pdf)). The type of the kernel `f` has to be `Complex{T}`, i.e. the complex of the k-space trajectory's type; for speed and memory efficiecy, call this function with `Float32.(tr)`, and set the type of `f` accordingly.

# Required Arguments
- `f::Array{T}`: Array in which the kernel will be written.
- `p::AbstractNFFTPlan`: NFFTPlan with the same dimensions as `tr`, which will be overwritten in place.
- `tr::Matrix{T}`: non-Cartesian k-space trajectory in units revolutions/voxel, i.e. `tr[i] ∈ [-0.5, 0.5] ∀ i`. The matrix has the size `2 x Nsamples` for 2D imaging with a trajectory length `Nsamples`, and `3 x Nsamples` for 3D imaging.
- `fftplan`: plan for the final FFT of the kernel from image to k-space. Therefore, it has to have twice the size of the original image. Calculate, e.g., with `fftplan = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.MEASURE)`, where `shape` is the size of the reconstructed image.

# Examples
```jldoctest; output = false, setup = :(using NFFT, NFFTTools, NFFTTools.FFTW, Random; Random.seed!(1))
julia> using NFFTTools.FFTW

julia> Nx = 32;

julia> trj = Float32.(rand(2, 1000) .- 0.5);

julia> p = plan_nfft(trj, (2Nx,2Nx))
NFFTPlan with 1000 sampling points for an input array of size(64, 64) and an output array of size(1000,) with dims 1:2

julia> fftplan = plan_fft(zeros(ComplexF32, (2Nx,2Nx)); flags=FFTW.MEASURE);

julia> λ = Array{ComplexF32}(undef, 2Nx, 2Nx);

julia> calculateToeplitzKernel!(λ, p, trj, fftplan);

julia> y = randn(ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(y, λ);

```
"""
function calculateToeplitzKernel!(f::Array{Complex{T}}, p::AbstractNFFTPlan{T}, tr::Matrix{T}, fftplan) where T
    nodes!(p, tr)
    f = mul!(f, adjoint(p), OnesVector(Complex{T}, size(tr,2)))
    f2 = fftshift(f)
    mul!(f, fftplan, f2)
    return f
end


###################################################################
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

###################################################################
# apply Toeplitz kernel
###################################################################

"""
    convolveToeplitzKernel!(y::Array{T,N}, λ::Array{T,N}[, fftplan = plan_fft(λ; flags=FFTW.ESTIMATE), ifftplan = plan_ifft(λ; flags=FFTW.ESTIMATE), xOS1 = similar(λ), xOS2 = similar(λ)])

Convolves the image `y` with the Toeplitz kernel `λ` and overwrites `y` with the result. `y` is also returned for convenience. As this function is commonly applied many times, it is highly recommended to pre-allocate / pre-compute all optional arguments. By doing so, this entire function is non-allocating.

# Required Arguments
- `y::Array{T,N}`: Input image that will be overwritten with the result. `y` is a matrix (`N=2`) for 2D imaging and a 3D tensor (`N=3`) for 3D imaging. The type of the elements `T` must match the ones of `λ`.
- `λ::Array{T,N}`: Toeplitz kernel, which as to be the same type as `k`, but twice the size due to the required oversampling (cf. [`calculateToeplitzKernel`](@ref)).

# Optional, but highly recommended Arguments
- `fftplan`: plan for the oversampled FFT, i.e. it has to have twice the size of the original image. Calculate, e.g., with `fftplan = plan_fft(λ; flags=FFTW.MEASURE)`, where `shape` is the size of the reconstructed image.
- `ifftplan`: plan for the oversampled inverse FFT. Calculate, e.g., with `ifftplan = plan_ifft(λ; flags=FFTW.MEASURE)`, where `shape` is the size of the reconstructed image.
- `xOS1`: pre-allocated array of the size of `λ`. Pre-allocate with `xOS1 = similar(λ)`.
- `xOS2`: pre-allocated array of the size of `λ`. Pre-allocate with `xOS2 = similar(λ)`.

# Examples
```jldoctest; output = false, setup = :(using NFFT, NFFTTools, NFFTTools.FFTW, Random; Random.seed!(1))
julia> using NFFTTools.FFTW

julia> Nx = 32;

julia> trj = Float32.(rand(2, 1000) .- 0.5);

julia> λ = calculateToeplitzKernel((Nx, Nx), trj);

julia> xOS1 = similar(λ);

julia> xOS2 = similar(λ);

julia> fftplan = plan_fft(xOS1; flags=FFTW.MEASURE);

julia> ifftplan = plan_ifft(xOS1; flags=FFTW.MEASURE);

julia> y = randn(ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(y, λ, fftplan, ifftplan, xOS1, xOS2)
32×32 Matrix{ComplexF32}:
  178.169-48.6707im   12.9519+21.8054im  …    749.91+324.029im
 -317.004+988.44im   -1214.49-1299.27im      410.381+747.908im
 -1087.78+870.916im   1028.41-1556.78im     -920.652+477.518im
  -1007.2-526.453im   -1691.3-663.165im     -524.921+600.5im
 -348.506+1479.8im    -1725.2-2714.59im      51.0908+1390.72im
 -1180.14-1977.12im   1391.15+969.678im  …   2966.56-737.661im
 -666.762+478.188im  -1309.84+404.228im     -607.705+129.812im
  1154.85+65.4261im  -301.903-570.698im      661.818+395.093im
 -786.583-1455.54im  -465.185+2436.7im      -1623.73+1694.6im
 -1045.34+28.3079im  -1123.31-930.451im      469.764+799.731im
         ⋮                               ⋱
  324.714+2197.86im   -909.22+565.227im     -1453.51-278.173im
 -1049.93-74.558im    360.753+577.881im      -1977.2-2348.59im
 -1542.49+1658.0im    1167.51+1285.3im   …   1107.94+651.439im
 -528.651-435.216im  -274.957-2013.84im       63.009-327.94im
 -1070.11-2721.88im  -480.178+2228.54im     -160.381-1271.39im
  1143.94-738.802im  -390.363+2484.48im       1816.4-261.432im
  1683.39-1242.72im   1905.95-1150.32im     -990.915-47.443im
  1047.94-1277.64im   66.1227-586.225im  …  -960.649+1029.82im
 -419.524-274.19im   -938.769+1969.06im     -890.385-878.783im

```
"""
function convolveToeplitzKernel!(k::Array{T,N}, λ::Array{T,N},
    fftplan = plan_fft(λ; flags=FFTW.ESTIMATE),
    ifftplan = plan_ifft(λ; flags=FFTW.ESTIMATE),
    xOS1 = similar(λ),
    xOS2 = similar(λ)
    ) where {T,N}

    fill!(xOS1, 0)
    xOS1[CartesianIndices(k)] .= k
    mul!(xOS2, fftplan, xOS1)
    xOS2 .*= λ
    mul!(xOS1, ifftplan, xOS2)
    k .= @view xOS1[CartesianIndices(k)]
    return k
end


###################################################################
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