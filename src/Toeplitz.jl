## ################################################################
# constructors
###################################################################
function calculateToeplitzKernel(shape, tr::Matrix{T}, m = 4, sigma = 2.0, window = :kaiser_bessel, K = 2000; fftplan = plan_fft(zeros(Complex{T}, 2 .* shape); flags=FFTW.ESTIMATE), kwargs...) where {T}

    shape_os = 2 .* shape

    p = NFFTPlan(tr, shape_os, m, sigma, window, K; kwargs...)
    eigMat = nfft_adjoint(p, ones(Complex{T}, size(tr,2)))
    return fftplan * fftshift(eigMat)
end

function calculateToeplitzKernel!(p::AbstractNFFTPlan, tr::Matrix{T}, fftplan) where T
    
    NFFT.NFFTPlan!(p, tr)
    eigMat = nfft_adjoint(p, OnesVector(Complex{T}, size(tr,2)))
    return fftplan * fftshift(eigMat)
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
function applyToeplitzKernel!(λ::Array{T,N}, x::Array{T,N}, 
    fftplan = plan_fft(λ; flags=FFTW.ESTIMATE),
    ifftplan = plan_ifft(λ; flags=FFTW.ESTIMATE),
    xOS1 = similar(λ),
    xOS2 = similar(λ)
    ) where {T,N}

    fill!(xOS1, 0)
    xOS1[CartesianIndices(x)] .= x
    mul!(xOS2, fftplan, xOS1)
    # println(all(xOS1[CartesianIndices(x)] .== x))
    xOS2 .*= λ
    # println(norm(xOS2))
    mul!(xOS1, ifftplan, xOS2)
    # println(norm(xOS1[CartesianIndices(x)] .- x))
    # println(norm(x))
    # # xL = ifftplan * (λ .* (fftplan * xL))
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