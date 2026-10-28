# libnfftjl (experimental)

NFFT.jl as a C library, compiled with `juliac --trim`: NFFT and adjoint NFFT in 1–3 dimensions,
double and single precision, parameters chosen from NFFT.jl's kernel table for a requested relative error. The
library carries its Julia runtime; no Julia installation is needed to use it. Bindings for other
languages (`../bindings/`) are thin wrappers around `include/nfftjl.h`.

```
./build.sh                   # → build/lib/libnfftjl.{so,dylib} with its runtime, the C test and examples
./test/test 8                # accuracy against the direct sum, timings, 8 threads
./examples/getting_started   # the examples of the documentation
./examples/batch
```

Documentation: the pages "C" and "Python" of the NFFT.jl documentation; their examples are the
files in `examples/` and `../bindings/python/examples/`, run by the CI job `libnfftjl`.

- `src/libnfftjl.jl`: the C entry points and the boundary that turns the run-time dimension D and
  window width W into one compiled branch per combination (D ∈ 1:3, W ∈ 4:16); plans live in
  registries of concrete type, a handle encodes (D, W, index).
- `include/nfftjl.h`: the interface, including the threading rules.
- `jc/`: the environment of JuliaC.jl.

Built with one Julia thread per CPU thread (`threads=auto`; plans use `nthreads` of them, by
default `NFFTJL_NUM_THREADS` or all), Julia's signal handling (the GC stops threads through a signal;
without the handler the process dies once more than one thread runs) and one CPU target
(`JULIA_CPU_TARGET`, default `apple-m1`, `x86-64-v3` or `generic` by machine): a library for several
targets hangs at the first garbage collection triggered by a second host thread
([JuliaC.jl#190](https://github.com/JuliaLang/JuliaC.jl/issues/190)).

Known limitations:
- the Julia runtime starts on the first call: make it from one thread before others call (two
  threads making it at once abort the process);
- FFTW.jl is not trimmable: `src/libnfftjl.jl` replaces three of its methods for the build;
- only the NFFT (no RNFFT, NFCT, NFST, NNFFT), no GPU.
