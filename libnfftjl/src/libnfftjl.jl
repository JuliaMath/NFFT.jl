# C library of NFFT.jl, compiled with `juliac --trim`: NFFT and adjoint NFFT in 1–3 dimensions,
# double and single precision, parameters chosen from NFFT.jl's kernel table. Interface in
# include/nfftjl.h.
#
# Trimming needs every call to be resolved at compile time. The plans of NFFT.jl choose the window
# width W at run time, and the type of the core depends on it; this file is the boundary that turns
# the run-time (T, D, W) into one compiled branch per combination and keeps the cores in registries
# of concrete type.

using NFFT
using NFFT: NFFTCore, SymKernel, KaiserBessel, FFTWBackend, kb_window, es_window, grid_size,
            choose_kernels, make_kernel, trafo_core!, adjoint_core!

# FFTW.jl destroys plans queued by finalizers through a call on its abstract plan type, which
# trimming cannot resolve; this version calls the C functions directly.
using NFFT: FFTW
@eval FFTW function destroy_deferred()
  lock(deferred_destroy_lock)
  try
    if !isempty(deferred_destroy_plans) && trylock(fftwlock)
      try
        for p in deferred_destroy_plans
          ptr = getfield(p, :plan)::PlanPtr
          if p isa FFTWPlan{<:fftwDouble}
            ccall((:fftw_destroy_plan, libfftw3), Cvoid, (PlanPtr,), ptr)
          else
            ccall((:fftwf_destroy_plan, libfftw3f), Cvoid, (PlanPtr,), ptr)
          end
        end
        empty!(deferred_destroy_plans)
      finally
        unlock(fftwlock)
      end
    end
  finally
    unlock(deferred_destroy_lock)
  end
end

# FFTW.jl loads libfftw3 on first use through a method that looks the library up by name and calls
# an untyped callback; this version is resolved at compile time. `load_fftw` makes it part of the
# library.
@eval FFTW function dlopen(lib::FakeLazyLibrary)
  h = @atomic :monotonic lib.h
  h != C_NULL && return h
  @lock fftwlock begin
    h = @atomic :monotonic lib.h
    h != C_NULL && return h
    h = dlopen(lib.reallibrary === :libfftw3_no_init ? libfftw3_no_init : libfftw3f_no_init)
    fftw_init_threads()
    @atomic :release lib.h = h
  end
  h
end
# FFTW's threads are Julia tasks started by this callback; `@sync` waits on an untyped channel.
@eval FFTW function spawnloop(f::Ptr{Cvoid}, fdata::Ptr{Cvoid}, elsize::Csize_t, num::Cint, callback_data::Ptr{Cvoid})
  ts = Vector{Task}(undef, num)
  for i = 0:num-1
    ts[i+1] = Threads.@spawn ccall(f, Ptr{Cvoid}, (Ptr{Cvoid},), fdata + elsize*i)
  end
  foreach(wait, ts)
end
load_fftw() = (NFFT.FFTW.dlopen(NFFT.FFTW.libfftw3); NFFT.FFTW.dlopen(NFFT.FFTW.libfftw3f); nothing)

# The variants of the library, one compiled branch each: element type × dimension × window width.
# A handle is index << 24 | t << 16 | D << 8 | W, t the position of the element type in TS.
const TS = (Float64, Float32)
const DS = 1:3
const WS = 4:16

# The FFT passes of the FFTW backend: in-place complex plans along one dimension.
fftw_pass(T, K, D) = NFFT.PassFFT{D,FFTW.cFFTWPlan{Complex{T},K,true,D,Int}}
core_type(T, D, W) = NFFTCore{D,SymKernel{W,cld(W, 2),cld(W + 4, 2),(W + 4) ÷ 2,T},T,Float64,
                              fftw_pass(T, FFTW.FORWARD, D),fftw_pass(T, FFTW.BACKWARD, D)}

# One registry of plans per variant.
for T in TS, D in DS, W in WS
  @eval const $(Symbol(:plans_, T, :_, D, :_, W)) = Union{Nothing,$(core_type(T, D, W))}[]
end
@generated registry(::Type{T}, ::Val{D}, ::Val{W}) where {T,D,W} = Symbol(:plans_, T, :_, D, :_, W)

tcode(::Type{T}) where {T} = findfirst(==(T), TS)
handle(T, D, W, i) = Int64(i) << 24 | Int64(tcode(T)) << 16 | Int64(D) << 8 | Int64(W)
unhandle(h) = (Int((h >> 16) & 0xff), Int((h >> 8) & 0xff), Int(h & 0xff), Int(h >> 24))

# f(TS[t]), f(Val(D)), f(Val(W)) for run-time values, one branch per supported value.
for (name, values, arg, err) in ((:with_type, 1:length(TS), t -> :($(TS[t])), "unsupported element type"),
                                 (:with_dim, DS, D -> :(Val($D)), "unsupported dimension"),
                                 (:with_width, WS, W -> :(Val($W)), "unsupported window width"))
  body = :(throw(ArgumentError($err)))
  for v in reverse(values)
    body = :(x == $v ? f($(arg(v))) : $body)
  end
  @eval $name(f, x) = $body
end

# Plans can be created and destroyed from any thread; a plan must not run on two threads at once.
# Planning runs one at a time: FFTW's planner thread count is global state.
const REGISTRY_LOCK = ReentrantLock()
const PLAN_LOCK = ReentrantLock()

function create(::Type{T}, ::Val{D}, Np::Ptr{Int64}, J::Int, kp::Ptr{Float64}, reltol::Float64,
                nthreads::Int) where {T,D}
  load_fftw()
  N = ntuple(d -> Int(unsafe_load(Np, d)), Val(D))
  k = copy(unsafe_wrap(Array, kp, (D, J)))
  e = first(choose_kernels(reltol, N, J, T)[1])
  W, β, e0 = e.W, e.β, e.e0
  φ = e.window isa KaiserBessel ? (z -> kb_window(z, W, β)) : (z -> es_window(z, W, β, e0))
  Ñ = grid_size(e.σ, N, W)
  with_width(W) do w
    c = NFFTCore(k, N, Ñ, W, φ, T, FFTWBackend(), nothing, nthreads; kernel=make_kernel(φ, w, T))
    reg = registry(T, Val(D), w)
    @lock REGISTRY_LOCK begin
      push!(reg, c)
      handle(T, D, W, length(reg))
    end
  end
end

# f(registry, index) for the registry of plan h
function with_registry(f, h)
  t, D, W, i = unhandle(h)
  with_type(t) do T
    with_dim(D) do d
      with_width(W) do w
        f(registry(T, d, w), i)
      end
    end
  end
end

# f(core) for plan h of element type T; -1 for an unknown handle or another element type
lookup(f, ::Type{T}, h) where {T} = with_registry(h) do reg, i
  c = @lock REGISTRY_LOCK (1 <= i <= length(reg) ? reg[i] : nothing)
  c isa NFFTCore{<:Any,<:Any,T} ? f(c) : Cint(-1)
end

function plan_create(::Type{T}, D, N, J, k, reltol, nthreads) where {T}
  try
    @lock PLAN_LOCK with_dim(Int(D)) do d
      create(T, d, N, Int(J), k, Float64(reltol), nthreads > 0 ? min(Int(nthreads), Threads.nthreads()) : default_threads())
    end
  catch e
    Core.println(Core.stderr, e)
    Int64(-1)
  end
end

function trafo(::Type{T}, h, f, fhat) where {T}
  try
    lookup(T, h) do c
      trafo_core!(unsafe_wrap(Array, fhat, length(c.binned.perm)), c, unsafe_wrap(Array, f, c.N))
      Cint(0)
    end
  catch
    Cint(-1)
  end
end

function adjoint(::Type{T}, h, fhat, f) where {T}
  try
    lookup(T, h) do c
      adjoint_core!(unsafe_wrap(Array, f, c.N), c, unsafe_wrap(Array, fhat, length(c.binned.perm)))
      Cint(0)
    end
  catch
    Cint(-1)
  end
end

function plan_destroy(h)
  try
    with_registry(h) do reg, i
      @lock REGISTRY_LOCK begin
        (1 <= i <= length(reg) && reg[i] !== nothing) || return Cint(-1)
        reg[i] = nothing
        Cint(0)
      end
    end
  catch
    Cint(-1)
  end
end

# The C functions: nfftjl_… in double precision, nfftjlf_… in single precision (nodes and
# reltol stay double).
for (T, prefix) in ((Float64, "nfftjl_"), (Float32, "nfftjlf_"))
  C = Complex{T}
  @eval begin
    Base.@ccallable $(Symbol(prefix, :plan_create))(D::Cint, N::Ptr{Int64}, J::Int64, k::Ptr{Float64},
      reltol::Cdouble, nthreads::Cint)::Int64 = plan_create($T, D, N, J, k, reltol, nthreads)
    Base.@ccallable $(Symbol(prefix, :trafo))(h::Int64, f::Ptr{$C}, fhat::Ptr{$C})::Cint = trafo($T, h, f, fhat)
    Base.@ccallable $(Symbol(prefix, :adjoint))(h::Int64, fhat::Ptr{$C}, f::Ptr{$C})::Cint = adjoint($T, h, fhat, f)
    Base.@ccallable $(Symbol(prefix, :plan_destroy))(h::Int64)::Cint = plan_destroy(h)
  end
end

# The library starts one Julia thread per CPU thread; plans use `nthreads` of them, by default
# NFFTJL_NUM_THREADS if set, otherwise all.
function default_threads()
  n = tryparse(Int, get(ENV, "NFFTJL_NUM_THREADS", ""))
  n === nothing ? Threads.nthreads() : clamp(n, 1, Threads.nthreads())
end
Base.@ccallable nfftjl_num_threads()::Cint = Cint(default_threads())
