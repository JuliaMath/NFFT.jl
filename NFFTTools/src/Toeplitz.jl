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
```jldoctest; output = false, setup = :(using NFFT, NFFTTools, StableRNGs, Random; rng = StableRNG(1); Random.seed!(rng, 1); Random.seed!(rng, 1))
julia> Nx = 32;

julia> trj = Float32.(rand(rng, 2, 1000) .- 0.5);

julia> λ = calculateToeplitzKernel((Nx, Nx), trj)
64×64 Matrix{ComplexF32}:
 -1528.64-170.936im  -622.032+56.4626im  …   2570.27-41.6055im
   3685.0+276.482im   2282.33-162.009im      2142.57-63.9413im
 -125.272-342.877im   819.441+228.404im      4293.36+130.336im
  940.796+31.5781im  -852.504+82.8948im      2681.78+180.963im
  445.039-252.226im  -1552.49+137.753im     -2037.24+39.6851im
  493.161+173.293im   664.534-58.8202im  …   1463.51+39.2481im
  332.265-180.768im   -2650.6+66.2951im     -2778.37-31.7731im
  2079.48-127.658im   4891.04+242.131im      3734.65+340.2im
  120.836-265.792im   10931.5+151.319im      4528.12+53.2504im
 -1247.63+356.045im  -198.919-241.572im       46.912-143.504im
         ⋮                               ⋱
 -340.961+142.965im   6010.83-28.4918im  …   2779.73+69.5765im
 -1826.62-161.638im   1103.42+47.1655im      2479.09-50.9026im
  2267.81+80.567im    1044.85+33.9058im      2327.04+131.974im
  1467.38-89.7949im   3410.85-24.6781im      1980.45-122.746im
   167.15+342.008im   1476.08-227.535im      2143.39-129.466im
 -724.138-204.324im   1686.02+89.8505im  …  -244.694-8.21769im
  237.304-84.753im   -1067.19+199.226im     -128.015+297.294im
  1848.84+50.5511im   1926.33-165.024im     -937.614-263.092im
  1912.83-152.94im    741.748+267.413im     -343.194+365.481im

julia> y = randn(ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(y, λ)
32×32 Matrix{ComplexF32}:
 -718.038+2035.75im  -1771.79+191.592im  …  -1087.95+260.256im
 -472.432+473.74im   -1894.01+1138.22im     -295.522-647.603im
  139.463+1137.28im    184.53+228.995im      788.817-355.36im
  2966.54-1767.96im   255.169+590.332im      341.963+94.7469im
  -594.67-538.93im    180.487-566.905im      69.4853+481.47im
 -1004.42-111.931im   2439.67-323.525im  …   -448.47+1459.57im
 -341.784+49.591im   -268.101-750.184im      1309.23-108.091im
  189.394+638.56im   -821.709+121.441im      100.152-914.375im
 -192.401-702.179im  -1564.33-536.778im       1448.4-971.389im
  492.427-1121.14im  -3270.99+249.791im     -245.744+1659.15im
         ⋮                               ⋱
  1309.34+98.404im    73.0096-1181.38im      557.821-1096.24im
  907.312+129.232im  -44.5222+1075.8im      -879.056-180.416im
 -1020.87-671.83im   -1019.44-778.932im  …  -878.007+2165.44im
 -392.366-745.654im  -279.611-1023.18im      804.285-51.4734im
 -1139.86-549.792im  -1135.39+1236.09im      538.431+1891.79im
  391.965+974.582im  -1445.21-1113.64im      1172.98+116.43im
 -870.086-1002.25im   1100.69+846.779im     -377.282-602.989im
  928.376-163.128im  -1009.71+1075.25im  …  -444.653-140.651im
  268.106+635.919im   919.454-170.406im      233.847-1058.75im

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
```jldoctest; output = false, setup = :(using NFFT, NFFTTools, NFFTTools.FFTW, StableRNGs, Random; rng = StableRNG(1); Random.seed!(rng, 1); Random.seed!(rng, 1))
julia> using NFFTTools.FFTW

julia> Nx = 32;

julia> trj = Float32.(rand(rng, 2, 1000) .- 0.5);

julia> p = plan_nfft(trj, (2Nx,2Nx))
NFFTPlan with 1000 sampling points for an input array of size(64, 64) and an output array of size(1000,) with dims 1:2

julia> fftplan = plan_fft(zeros(ComplexF32, (2Nx,2Nx)); flags=FFTW.MEASURE);

julia> λ = Array{ComplexF32}(undef, 2Nx, 2Nx);

julia> calculateToeplitzKernel!(λ, p, trj, fftplan);

julia> y = randn(rng, ComplexF32, Nx, Nx);

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
```jldoctest; output = false, setup = :(using NFFT, NFFTTools, NFFTTools.FFTW, StableRNGs, Random; rng = StableRNG(1); Random.seed!(rng, 1))
julia> using NFFTTools.FFTW

julia> Nx = 32;

julia> trj = Float32.(rand(rng, 2, 1000) .- 0.5);

julia> λ = calculateToeplitzKernel((Nx, Nx), trj);

julia> xOS1 = similar(λ);

julia> xOS2 = similar(λ);

julia> fftplan = plan_fft(xOS1; flags=FFTW.MEASURE);

julia> ifftplan = plan_ifft(xOS1; flags=FFTW.MEASURE);

julia> y = randn(rng, ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(y, λ, fftplan, ifftplan, xOS1, xOS2)
32×32 Matrix{ComplexF32}:
  -783.38-230.709im  -41.2715-196.622im  …   1051.97+1655.19im
  306.303+931.416im  -212.223-960.781im     -270.816-4.68915im
  1018.79+974.724im   1204.28-1286.06im      398.504+298.177im
   138.68+346.405im   225.016-586.742im      397.565-205.818im
  402.152+725.267im   731.119-307.097im      810.773-244.329im
  680.202-682.887im  -7.59145-254.964im  …  -1219.87-1119.43im
 -64.0526-909.241im   61.5645+1199.98im       253.19-630.097im
 -835.246-993.775im  -1561.78+969.924im     -7.08272+1755.43im
   163.15-212.155im   1282.88+250.916im      819.356+1184.85im
  71.7218-933.054im   772.495-39.3827im      495.359+2949.17im
         ⋮                               ⋱
 -499.452-192.12im   -589.649+1561.74im      1544.92+126.5im
 -347.838+791.432im  -112.339+269.57im      -1068.77+452.493im
  1861.73-494.369im   416.406+499.465im  …  -1856.07-211.381im
   176.94+984.977im   874.282+41.8216im     -1717.71+1169.11im
  516.513+270.692im   531.069+1907.76im     -697.752+42.6127im
  -65.358-411.893im  -1299.86-868.781im     -285.473-1803.2im
 -15.1395+439.582im   84.4428+2026.06im      13.6334+24.6603im
 -1146.97+1632.87im  -208.162+1114.2im   …   -295.39+854.479im
  502.986+591.013im  -1013.11+97.8801im      617.683+17.4492im

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