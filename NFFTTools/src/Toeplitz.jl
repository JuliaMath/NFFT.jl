###################################################################
# constructors
###################################################################

"""
    calculateToeplitzKernel(shape, tr::AbstractMatrix{T}; fftplan = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE), kwargs...)

Calculate the kernel for an implementation of the Gram matrix ``A^{\\mathsf{H}} A`` that utilizes its Toeplitz structure. The output is an array of twice the size of `shape`, as the Toeplitz trick requires an oversampling factor of 2 (cf. [Wajer, F. T. A. W., and K. P. Pruessmann. "Major speedup of reconstruction for sensitivity encoding with arbitrary trajectories." Proc. Intl. Soc. Mag. Res. Med. 2001.](https://cds.ismrm.org/ismrm-2001/PDF3/0767.pdf)). The type of the kernel is `Complex{T}`, i.e. the complex of the k-space trajectory's type; for speed and memory efficiency, call this function with `Float32.(tr)`, and the kernel will also be `Float32`.

The kernel is computed with an adjoint NFFT of the active AbstractNFFTs backend (e.g. NFFT.jl after `using NFFT`).

# Required Arguments
- `shape::NTuple(Int)`: size of the image; e.g. `(256, 256)` for 2D imaging, or `(256,256,128)` for 3D imaging
- `tr::Matrix{T}`: non-Cartesian k-space trajectory in units revolutions/voxel, i.e. `tr[i] ∈ [-0.5, 0.5] ∀ i`. The matrix has the size `2 x Nsamples` for 2D imaging with a trajectory length `Nsamples`, and `3 x Nsamples` for 3D imaging.

# Keyword Arguments:
- `fftplan`: plan for the final FFT of the kernel from image to k-space. Therefore, it has to have twice the size of the original image. `default = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE)`. If this constructor is used many times, it is worth to precompute the plan with `plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.MEASURE)` and reuse it.
- `kwargs`: passed on to `plan_nfft`, e.g. `reltol` for the accuracy of the kernel.

# Examples
```julia
julia> using NFFT, NFFTTools

julia> Nx = 32;

julia> trj = Float32.(rand(2, 1000) .- 0.5);

julia> λ = calculateToeplitzKernel((Nx, Nx), trj);

julia> y = randn(ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(y, λ);
```
"""
function calculateToeplitzKernel(shape, tr::AbstractMatrix{T}; fftplan = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE), kwargs...) where {T}
    p = plan_nfft(tr, 2 .* shape; kwargs...)
    eigMat = adjoint(p) * ones(Complex{T}, size(tr, 2))
    return fftplan * fftshift(eigMat)
end

"""
    calculateToeplitzKernel!(f::Array{Complex{T}}, p::AbstractNFFTPlan, tr::Matrix{T}, fftplan)

Calculate the kernel for an implementation of the Gram matrix that utilizes its Toeplitz structure and writes it in-place in `f`, which has to be twice the size of the desired image matrix, as the Toeplitz trick requires an oversampling factor of 2 (cf. [Wajer, F. T. A. W., and K. P. Pruessmann. "Major speedup of reconstruction for sensitivity encoding with arbitrary trajectories." Proc. Intl. Soc. Mag. Res. Med. 2001.](https://cds.ismrm.org/ismrm-2001/PDF3/0767.pdf)). The type of the kernel `f` has to be `Complex{T}`, i.e. the complex of the k-space trajectory's type; for speed and memory efficiecy, call this function with `Float32.(tr)`, and set the type of `f` accordingly.

# Required Arguments
- `f::Array{T}`: Array in which the kernel will be written.
- `p::AbstractNFFTPlan`: NFFT plan for an array of twice the image size; its nodes are set to `tr`.
- `tr::Matrix{T}`: non-Cartesian k-space trajectory in units revolutions/voxel, i.e. `tr[i] ∈ [-0.5, 0.5] ∀ i`. The matrix has the size `2 x Nsamples` for 2D imaging with a trajectory length `Nsamples`, and `3 x Nsamples` for 3D imaging.
- `fftplan`: plan for the final FFT of the kernel from image to k-space. Therefore, it has to have twice the size of the original image. Calculate, e.g., with `fftplan = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.MEASURE)`, where `shape` is the size of the reconstructed image.

# Examples
```julia
julia> using NFFT, NFFTTools, NFFTTools.FFTW

julia> Nx = 32;

julia> trj = Float32.(rand(2, 1000) .- 0.5);

julia> p = plan_nfft(trj, (2Nx, 2Nx));

julia> fftplan = plan_fft(zeros(ComplexF32, (2Nx,2Nx)); flags=FFTW.MEASURE);

julia> λ = Array{ComplexF32}(undef, 2Nx, 2Nx);

julia> calculateToeplitzKernel!(λ, p, trj, fftplan);
```
"""
function calculateToeplitzKernel!(f::Array{Complex{T}}, p::AbstractNFFTPlan{T}, tr::Matrix{T}, fftplan) where T
    nodes!(p, tr)
    mul!(f, adjoint(p), ones(Complex{T}, size(tr, 2)))
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
```julia
julia> using NFFT, NFFTTools, NFFTTools.FFTW

julia> Nx = 32;

julia> trj = Float32.(rand(2, 1000) .- 0.5);

julia> λ = calculateToeplitzKernel((Nx, Nx), trj);

julia> xOS1 = similar(λ);

julia> xOS2 = similar(λ);

julia> fftplan = plan_fft(xOS1; flags=FFTW.MEASURE);

julia> ifftplan = plan_ifft(xOS1; flags=FFTW.MEASURE);

julia> y = randn(ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(y, λ, fftplan, ifftplan, xOS1, xOS2);
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
