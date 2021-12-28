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
  1858.23-220.871im   3723.91-4.6283im       3917.46+217.494im
  3260.49+249.208im   1719.05-23.7087im      1935.48-245.831im
  1115.56-210.368im  -85.7251-15.1312im      499.092+206.991im
  3232.19+666.148im  -927.182-440.649im      1048.98-662.771im
 -2818.12-640.593im   2892.04+415.094im  …   -435.71+637.216im
  3019.67+377.945im   1978.27-152.445im      4267.19-374.567im
  711.754-499.0im     11091.8+273.5im         822.52+495.622im
 -436.376+576.158im   2085.76-350.659im      280.188-572.781im
 -460.382-658.799im  -402.648+433.299im      1579.94+655.422im
         ⋮                               ⋱
  863.552-940.385im  -571.564+714.885im  …   325.992+937.007im
  -554.64+283.37im   -416.581-57.8704im     -295.217-279.992im
  680.462-381.469im   437.636+155.97im       381.239+378.092im
 -1769.07+404.678im   291.916-179.178im      192.257-401.301im
  1502.99-212.153im   3114.76-13.3461im      3939.76+208.776im
  1192.88+274.105im   728.321-48.6055im  …   593.151-270.728im
  318.764-370.158im  -273.716+144.659im     -323.591+366.781im
 -810.074+257.923im  -144.069-32.4242im     -44.2182-254.546im
  6650.49-187.076im  -1154.77-38.4238im      1493.98+183.698im

julia> x = randn(ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(x, λ)
32×32 Matrix{ComplexF32}:
 -2548.17+350.061im  -83.5558-1118.35im  …  -369.412+1283.74im
 -232.842+353.766im   950.106+2080.18im     -598.713-944.801im
 -1009.52-1121.52im   93.4049+693.621im      1390.72+197.977im
  1812.78-9.56204im   275.548+599.843im      458.824-1033.59im
 -1213.11-163.169im  -329.302-1185.4im       356.961-1026.23im
  1479.05-1515.57im   1149.94-1208.13im  …   410.585+209.634im
  1021.91+1114.42im   571.518+509.203im     -1047.46+647.598im
 -12.9758+900.257im   880.053+758.609im     -1017.49-1609.38im
  398.851+171.48im     830.65+1170.5im       1277.94-2639.44im
  188.979+789.158im   415.594+370.381im       101.68-1864.72im
         ⋮                               ⋱
  -1458.0-1250.08im   2466.35+374.447im     -352.156+306.171im
  828.511-333.488im   2105.72+344.53im      -401.738-782.178im
 -602.072-108.482im  -1368.96+1090.78im  …  -129.634-1951.61im
  169.226-1887.22im   1637.13-460.677im      1077.74+775.838im
  593.828+1301.49im  -1322.09+194.759im     -1700.84+531.624im
  775.017-208.649im    1361.0-528.35im       -434.66+206.32im
  1044.89-874.581im  -286.478+795.039im     -1186.11+879.179im
 -665.938+1052.24im  -34.6694-1059.52im  …   -1700.0-1452.94im
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
    calculateToeplitzKernel!(p::AbstractNFFTPlan, tr::Matrix{T}, fftplan)

Calculate the kernel for an implementation of the Gram matrix that utilizes its Toeplitz structure. The output is an array of twice the size of `shape`, as the Toeplitz trick requires an oversampling factor of 2 (cf. [Wajer, F. T. A. W., and K. P. Pruessmann. "Major speedup of reconstruction for sensitivity encoding with arbitrary trajectories." Proc. Intl. Soc. Mag. Res. Med. 2001.](https://cds.ismrm.org/ismrm-2001/PDF3/0767.pdf)). The type of the kernel is `Complex{T}`, i.e. the complex of the k-space trajectory's type; for speed and memory efficiecy, call this function with `Float32.(tr)`, and the kernel will also be `Float32`.
    
# Required Arguments
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

julia> λ1 = calculateToeplitzKernel!(p, trj, fftplan)
64×64 Matrix{ComplexF32}:
  3291.57-5.23337im   11172.8+119.126im  …   549.334+30.3806im
  1825.79-197.899im   2383.35+84.0067im      3538.35+172.752im
  2101.36+86.3478im   2251.37+27.5447im      6552.28-61.2003im
  2962.63-300.765im  -1381.33+186.873im      151.743+275.618im
  1458.84+181.742im   2695.23-67.8495im      4460.57-156.595im
 -306.168+73.044im    -482.71-186.937im  …  -30.6732-98.1915im
  310.288+147.197im   132.948-33.3049im      3984.78-122.05im
 -512.951-169.173im  -44.2078+55.2804im     -305.216+144.025im
 -547.692-106.469im   764.821+220.362im      1458.72+131.617im
   4396.0-21.6088im  -553.258-92.2836im     -1772.26-3.53851im
         ⋮                               ⋱
  838.655-21.3465im   3656.22-92.5459im  …   1276.71-3.80072im
  3006.03-83.6215im    1191.5+197.514im     -865.912+108.769im
 -1397.04+47.8644im  -1044.18-161.757im      1587.03-73.0119im
   1375.2+9.20293im   2793.47+104.69im      -2070.56+15.9446im
 -150.262-257.82im     1376.3+143.927im      6531.63+232.672im
  1835.54+178.592im  -32.3303-64.6993im  …   4210.31-153.445im
  3026.88+198.916im   3208.01-312.808im      4083.19-224.063im
  2344.35-161.57im   -1564.81+275.462im      2155.31+186.717im
 -43.1378+219.072im   5435.56-332.965im      4017.53-244.22im

julia> trj = Float32.(rand(2, 1000) .- 0.5);

julia> λ2 = calculateToeplitzKernel!(p, trj, fftplan)
64×64 Matrix{ComplexF32}:
  3291.57-5.23337im   11172.8+119.126im  …   549.334+30.3806im
  1825.79-197.899im   2383.35+84.0067im      3538.35+172.752im
  2101.36+86.3478im   2251.37+27.5447im      6552.28-61.2003im
  2962.63-300.765im  -1381.33+186.873im      151.743+275.618im
  1458.84+181.742im   2695.23-67.8495im      4460.57-156.595im
 -306.168+73.044im    -482.71-186.937im  …  -30.6732-98.1915im
  310.288+147.197im   132.948-33.3049im      3984.78-122.05im
 -512.951-169.173im  -44.2078+55.2804im     -305.216+144.025im
 -547.692-106.469im   764.821+220.362im      1458.72+131.617im
   4396.0-21.6088im  -553.258-92.2836im     -1772.26-3.53851im
         ⋮                               ⋱
  838.655-21.3465im   3656.22-92.5459im  …   1276.71-3.80072im
  3006.03-83.6215im    1191.5+197.514im     -865.912+108.769im
 -1397.04+47.8644im  -1044.18-161.757im      1587.03-73.0119im
   1375.2+9.20293im   2793.47+104.69im      -2070.56+15.9446im
 -150.262-257.82im     1376.3+143.927im      6531.63+232.672im
  1835.54+178.592im  -32.3303-64.6993im  …   4210.31-153.445im
  3026.88+198.916im   3208.01-312.808im      4083.19-224.063im
  2344.35-161.57im   -1564.81+275.462im      2155.31+186.717im
 -43.1378+219.072im   5435.56-332.965im      4017.53-244.22im

julia> x = randn(ComplexF32, Nx, Nx);

julia> convolveToeplitzKernel!(x, λ2)
32×32 Matrix{ComplexF32}:
 -804.467+1917.24im   922.623-1057.46im  …   495.221-560.612im
  -832.78+283.226im  -1011.97-306.645im       1888.6-105.973im
  861.423+717.68im    1882.58-1633.39im     -1063.48-1949.81im
  -7.3927-186.439im  -424.059-121.031im        950.0-337.24im
  862.794-648.123im   300.136+841.499im      1195.18+231.645im
  68.9932-973.394im   991.002-1666.32im  …    434.08+618.368im
 -964.571+153.546im  -119.333-1856.68im     -789.721+1370.91im
  1700.86+1836.49im    1732.0-1759.43im      416.865+738.721im
 -312.759+721.84im    244.448-123.489im     -29.6288-1645.01im
 -283.416+3200.34im   1056.04-285.379im      -2113.6+98.4256im
         ⋮                               ⋱
   423.91+348.467im  -110.201-1225.77im     -1287.41-295.876im
  1264.59+133.849im   30.1002+2292.07im     -1874.74-852.603im
  1719.43-1786.84im   179.916-1548.6im   …  -906.668+387.788im
 -1230.03+483.669im   1159.37-187.133im      1800.83+338.619im
  564.672+136.74im    1447.06+244.987im     -476.902-1405.92im
   27.928-460.166im    1067.1-1771.83im      -480.74+988.426im
  115.245-848.071im  -1892.78-1016.47im      344.212-65.0698im
  1977.91+672.363im   -87.114-885.466im  …   104.742-400.369im
 -3.54703-1212.56im  -1409.44+1718.01im     -1017.39+166.616im

```
"""
function calculateToeplitzKernel!(p::AbstractNFFTPlan, tr::Matrix{T}, fftplan) where T
    
    NFFTPlan!(p, tr)
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
  1730.51-1401.55im   1062.13-456.86im   …  -933.447-506.518im
 -941.469+1587.93im   968.095+109.594im      1217.84-381.425im
 -106.618-790.995im  -586.872-600.95im       -178.45-2630.04im
  123.857+1550.66im  -1703.82-306.442im     -955.495+476.402im
  548.116-1668.96im   63.8556-1033.34im      453.909-1153.28im
  662.111-2074.47im  -2747.51-477.537im  …  -561.385+537.565im
  -10.469+1877.94im  -625.796-127.713im     -1453.56-509.269im
  1247.48-1074.13im   251.784+1003.77im      111.856+467.224im
  246.913+61.3439im   214.395+1128.35im     -75.8665+1568.52im
  13.3036-1590.66im   1223.77-1783.53im     -28.5231+1629.33im
         ⋮                               ⋱
 -1031.12-568.437im  -793.158-570.475im      1146.01-2028.31im
  1140.73+34.5026im   322.405+2135.21im     -2438.59-1312.94im
   1985.3-300.237im   1177.74+1649.01im  …   2371.06+1195.72im
  1392.31-438.592im   74.8741-624.228im      878.585+407.952im
  975.943-183.759im  -129.389-2819.62im      324.353+1201.24im
 -1409.85+472.723im  -379.862+179.77im       880.157+872.769im
  1175.07+435.927im  -242.617-604.283im      456.229+449.378im
  341.516+317.086im   1266.79-573.395im  …    308.95-921.569im
 -737.968+1727.44im  -435.403+1192.29im     -1168.47+257.874im

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