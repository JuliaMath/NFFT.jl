# C library of NFFT.jl, compiled with `juliac --trim`: NFFT and adjoint NFFT in 1–3 dimensions,
# with batch dimensions, double and single precision, parameters chosen from NFFT.jl's kernel table. Interface in
# include/nfftjl.h.
#
# Trimming needs every call to be resolved at compile time. The plans of NFFT.jl choose the window
# width W at run time, and the type of the core depends on it; this file is the boundary that turns
# the run-time (T, D, W, batch strategy) into one compiled branch per combination and keeps the
# cores in registries of concrete type.

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

# The variants of the library, one compiled branch each: element type × dimension × window width
# × batch strategy (as in NFFT.jl: no batch, slices one after the other on a threaded core, slices
# in parallel on single-threaded cores, the batch kernels for batches before or after the
# transformed dimensions). A handle is index << 28 | s << 24 | t << 16 | D << 8 | W, with s and t
# the positions in STRATEGIES and TS.
const TS = (Float64, Float32)
const DS = 1:3
const WS = 4:16
const STRATEGIES = (:single, :loop, :slices, :lead, :trail)

# The FFT passes of the FFTW backend: in-place complex plans along one dimension.
fftw_pass(T, K, D) = NFFT.PassFFT{D,FFTW.cFFTWPlan{Complex{T},K,true,D,Int}}
kernel_type(T, W) = SymKernel{W,cld(W, 2),cld(W + 4, 2),(W + 4) ÷ 2,T}
core_type(T, D, W) = NFFTCore{D,kernel_type(T, W),T,Float64,fftw_pass(T, FFTW.FORWARD, D),fftw_pass(T, FFTW.BACKWARD, D),
                              FFTWBackend}
# the batch kernels with chunks of 4 (batches of at least 4)
batch_type(T, D, W) = NFFT.BatchCore{D,kernel_type(T, W),T,Float64,D + 1,4,fftw_pass(T, FFTW.FORWARD, D + 1),
                                     fftw_pass(T, FFTW.BACKWARD, D + 1),FFTWBackend}
engine_core(S, T, D, W) =
  S in (:single, :loop) ? core_type(T, D, W) :
  S === :slices ? Vector{core_type(T, D, W)} :
  D > 2 ? Nothing :                                        # batch kernels in 1D and 2D only
  S === :lead ? batch_type(T, D, W) :
  NFFT.TrailCore{batch_type(T, D, W),Array{Complex{T},D + 1},Matrix{Complex{T}}}

# A plan of the library: its core, the batch sizes before (L1) and after (L3) the transformed
# dimensions, and the number of nodes. Data are (L1, N..., L3) and (L1, J, L3), column major.
struct Engine{S,C}
  core::C
  L1::Int
  L3::Int
  J::Int
end

# One registry of plans per variant; Nothing[] for combinations that do not exist.
for S in STRATEGIES, T in TS, D in DS, W in WS
  C = engine_core(S, T, D, W)
  E = C === Nothing ? Nothing : Union{Nothing,Engine{S,C}}
  @eval const $(Symbol(:plans_, S, :_, T, :_, D, :_, W)) = $E[]
end
@generated registry(::Val{S}, ::Type{T}, ::Val{D}, ::Val{W}) where {S,T,D,W} = Symbol(:plans_, S, :_, T, :_, D, :_, W)

index(x, xs) = findfirst(==(x), xs)
name(::Val{S}) where {S} = S
handle(S, T, D, W, i) = Int64(i) << 28 | Int64(index(S, STRATEGIES)) << 24 | Int64(index(T, TS)) << 16 |
                        Int64(D) << 8 | Int64(W)
unhandle(h) = (Int((h >> 24) & 0xf), Int((h >> 16) & 0xff), Int((h >> 8) & 0xff), Int(h & 0xff), Int(h >> 28))

# f(Val(STRATEGIES[s])), f(TS[t]), f(Val(D)), f(Val(W)) for run-time values, one branch per value.
for (name, values, arg, err) in ((:with_strategy, 1:length(STRATEGIES), s -> :(Val($(QuoteNode(STRATEGIES[s])))),
                                  "unsupported strategy"),
                                 (:with_type, 1:length(TS), t -> :($(TS[t])), "unsupported element type"),
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

# The batch strategy of NFFT.jl (`make_engine`), with the batch kernels for batches of at least 4.
function strategy(D, N, J, L1, L3, nthreads)
  L = L1 * L3
  L == 1 && return 1
  NFFT.use_slices(L, J + prod(N), nthreads) && return 3
  D <= 2 && L3 == 1 && L1 >= 4 && return 4
  D <= 2 && L1 == 1 && L3 >= 4 && return 5
  2
end

function make_engine(::Val{S}, ::Type{T}, ::Val{D}, k, N, Ñ, W, φ::F, kernel, nthreads, L1, L3, J) where {S,T,D,F}
  fft = FFTWBackend()
  core = if S === :single || S === :loop
    NFFTCore(k, N, Ñ, W, φ, T, fft, nothing, nthreads; kernel)
  elseif S === :slices
    NFFT.single_cores(NFFTCore(k, N, Ñ, W, φ, T, fft, nothing, nthreads; kernel), L1 * L3)
  elseif D > 2
    throw(ArgumentError("no batch kernels in $D dimensions"))
  elseif S === :lead
    NFFT.BatchCore(k, N, Ñ, W, φ, T, fft, L1, nothing, nthreads; kernel, chunk=Val(4))
  else
    NFFT.TrailCore(NFFT.BatchCore(k, N, Ñ, W, φ, T, fft, L3, nothing, nthreads; kernel, chunk=Val(4)),
                   Array{Complex{T}}(undef, L3, N...), Matrix{Complex{T}}(undef, L3, J))
  end
  Engine{S,typeof(core)}(core, L1, L3, J)
end

function create(::Type{T}, ::Val{D}, Np::Ptr{Int64}, J::Int, kp::Ptr{Float64}, reltol::Float64,
                nthreads::Int, L1::Int, L3::Int) where {T,D}
  load_fftw()
  (L1 >= 1 && L3 >= 1) || throw(ArgumentError("batch sizes must be positive"))
  N = ntuple(d -> Int(unsafe_load(Np, d)), Val(D))
  k = copy(unsafe_wrap(Array, kp, (D, J)))
  e = first(choose_kernels(reltol, N, J, T)[1])
  W, β, e0 = e.W, e.β, e.e0
  φ = e.window isa KaiserBessel ? (z -> kb_window(z, W, β)) : (z -> es_window(z, W, β, e0))
  Ñ = grid_size(e.σ, N, W)
  with_width(W) do w
    kernel = make_kernel(φ, w, T)
    with_strategy(strategy(D, N, J, L1, L3, nthreads)) do s
      eng = make_engine(s, T, Val(D), k, N, Ñ, W, φ, kernel, nthreads, L1, L3, J)
      reg = registry(s, T, Val(D), w)
      @lock REGISTRY_LOCK begin
        push!(reg, eng)
        handle(name(s), T, D, W, length(reg))
      end
    end
  end
end

# f(registry, index, element type) for the registry of plan h
function with_registry(f, h)
  s, t, D, W, i = unhandle(h)
  with_strategy(s) do sv
    with_type(t) do T
      with_dim(D) do d
        with_width(W) do w
          f(registry(sv, T, d, w), i, T)
        end
      end
    end
  end
end

# f(engine) for plan h of element type T; -1 for an unknown handle or another element type
lookup(f, ::Type{T}, h) where {T} = with_registry(h) do reg, i, Th
  Th === T || return Cint(-1)
  e = @lock REGISTRY_LOCK (1 <= i <= length(reg) ? reg[i] : nothing)
  e === nothing ? Cint(-1) : f(e)
end

core_size(c::NFFTCore) = c.N
core_size(c::Vector) = c[1].N
core_size(c::NFFT.BatchCore) = c.N
core_size(t::NFFT.TrailCore) = t.core.N

# The transforms on the data of the plan, (L1, N..., L3) and (L1, J, L3); the batch strategies of
# NFFT.jl read only the full size and the transformed dimensions of their plan.
function data(e::Engine, fp, gp)
  N = core_size(e.core)
  shape = (N=(e.L1, N..., e.L3), dims=2:length(N)+1)
  shape, unsafe_wrap(Array, fp, shape.N), unsafe_wrap(Array, gp, (e.L1, e.J, e.L3))
end
function run_trafo(e::Engine{S}, fp, gp) where {S}
  shape, f, fhat = data(e, fp, gp)
  S === :single ? trafo_core!(vec(fhat), e.core, reshape(f, core_size(e.core))) :
                  NFFT.transform!(Val(S), shape, e.core, fhat, f)
  Cint(0)
end
function run_adjoint(e::Engine{S}, gp, fp) where {S}
  shape, f, fhat = data(e, fp, gp)
  S === :single ? adjoint_core!(reshape(f, core_size(e.core)), e.core, vec(fhat)) :
                  NFFT.transform_adjoint!(Val(S), shape, e.core, f, fhat)
  Cint(0)
end

function plan_create(::Type{T}, D, N, J, k, reltol, nthreads, L1=1, L3=1) where {T}
  try
    @lock PLAN_LOCK with_dim(Int(D)) do d
      create(T, d, N, Int(J), k, Float64(reltol),
             nthreads > 0 ? min(Int(nthreads), Threads.nthreads()) : default_threads(), Int(L1), Int(L3))
    end
  catch e
    Core.println(Core.stderr, e)
    Int64(-1)
  end
end

function trafo(::Type{T}, h, f, fhat) where {T}
  try
    lookup(e -> run_trafo(e, f, fhat), T, h)
  catch
    Cint(-1)
  end
end

function adjoint(::Type{T}, h, fhat, f) where {T}
  try
    lookup(e -> run_adjoint(e, fhat, f), T, h)
  catch
    Cint(-1)
  end
end

function plan_destroy(h)
  try
    with_registry(h) do reg, i, _
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
    Base.@ccallable $(Symbol(prefix, :plan_create_many))(D::Cint, N::Ptr{Int64}, J::Int64, k::Ptr{Float64},
      L1::Int64, L3::Int64, reltol::Cdouble, nthreads::Cint)::Int64 =
      plan_create($T, D, N, J, k, reltol, nthreads, L1, L3)
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
