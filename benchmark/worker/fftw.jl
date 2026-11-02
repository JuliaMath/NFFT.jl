# FFTW state of a worker process, for every FFTW the process has loaded (FFTW.jl's FFTW_jll and a
# library's own build, e.g. FINUFFT's with OpenMP threads in a separate libfftw3_omp).
#
# FFTW keeps the plans it measured as wisdom per process; later plans of any flag reuse it. Every
# plan of a task starts without it.
#
# FFTW requires `fftw_init_threads` before any other FFTW call: `fftw_forget_wisdom` creates and
# configures the planner, and a planner configured before the threads are initialised never gets
# the threaded Cooley-Tukey solvers (only the threaded loops over vector dimensions), so a threaded
# 1D plan is up to 10× slower. FFTW.jl and FINUFFT initialise the threads lazily, at their first
# plan; hence the threads are initialised here before the first forget. `fftw_init_threads` is
# idempotent, so the libraries' own later calls are no-ops.

using Libdl

const FFTW_PRECISIONS = (("fftw", r"^libfftw3\."), ("fftwf", r"^libfftw3f\."))

# the library that holds `<prefix>_init_threads` for the FFTW `lib`: `lib` itself (FFTW_jll is
# built with combined threads) or the one loaded `_threads` / `_omp` library next to it
function fftw_threads_lib(lib, prefix)
  Libdl.dlsym(Libdl.dlopen(lib), Symbol(prefix, "_init_threads"); throw_error=false) === nothing || return lib
  base = replace(basename(lib), r"\..*$" => "")
  cand = filter(l -> occursin(Regex("^" * base * "_(threads|omp)\\."), basename(l)) && dirname(l) == dirname(lib),
                Libdl.dllist())
  length(cand) <= 1 || error("several FFTW thread libraries for $lib: $cand")
  isempty(cand) ? nothing : only(cand)
end

function forget_fftw_wisdom()
  for lib in Libdl.dllist(), (prefix, pat) in FFTW_PRECISIONS
    occursin(pat, basename(lib)) || continue
    tlib = fftw_threads_lib(lib, prefix)
    if tlib !== nothing
      ccall(Libdl.dlsym(Libdl.dlopen(tlib), Symbol(prefix, "_init_threads")), Cint, ()) == 0 &&
        error("$(prefix)_init_threads failed in $tlib")
    end
    ccall(Libdl.dlsym(Libdl.dlopen(lib), Symbol(prefix, "_forget_wisdom")), Cvoid, ())
  end
end
