using ducc0_jll
const libducc = ducc0_jll.libducc_julia

function ducc_nu2u(coord::Array{Cdouble,2}, data::Vector{Complex{Cdouble}}, shape; epsilon::AbstractFloat, nthreads::Int=1, verbosity::Int=0, periodicity::AbstractFloat=1., forward::Int=0)
  res=Array{Complex{Cdouble}}(undef, shape)
  shp = Array{Csize_t}([x for x in shape])
  ccall((:nufft_nu2u_julia_double,libducc),
    Cvoid, (Csize_t,Csize_t,Ptr{Csize_t},Ptr{Cdouble},Ptr{Cdouble},Cint,Cdouble,Csize_t,Ptr{Cdouble},Csize_t,Cdouble,Cdouble,Cdouble,Cint),
    size(coord)[1], size(coord)[2], pointer(shp), pointer(data), pointer(coord),
    forward, epsilon, nthreads, pointer(res), verbosity, 1.1, 2.6,
    periodicity, 0)
  return res
end

function ducc_u2nu(coord::Array{Cdouble,2}, data::Array{Complex{Cdouble}}; epsilon::AbstractFloat, nthreads::Int=1, verbosity::Int=0, periodicity::AbstractFloat=1., forward::Int=1)
  shape = size(data)
  res=Array{Complex{Cdouble}}(undef, size(coord)[2])
  shp = Array{Csize_t}([x for x in shape])
  ccall((:nufft_u2nu_julia_double,libducc),
    Cvoid, (Csize_t,Csize_t,Ptr{Csize_t},Ptr{Cdouble},Ptr{Cdouble},Cint,Cdouble,Csize_t,Ptr{Cdouble},Csize_t,Cdouble,Cdouble,Cdouble,Cint),
    size(coord)[1], size(coord)[2], pointer(shp), pointer(data), pointer(coord),
    forward, epsilon, nthreads, pointer(res), verbosity, 1.1, 2.6,
    periodicity, 0)
  return res
end
