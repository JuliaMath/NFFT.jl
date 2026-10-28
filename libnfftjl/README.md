# libnfftjl (experimental)

NFFT.jl as a C library, compiled with `juliac --trim`: NFFT and adjoint NFFT in 1–3 dimensions,
Float64, parameters chosen from NFFT.jl's kernel table for a requested relative error. The
library carries its Julia runtime; no Julia installation is needed to use it. Bindings for other
languages (`../bindings/`) are thin wrappers around `include/nfftjl.h`.

```
./build.sh          # → build/lib/libnfftjl.{so,dylib} with its runtime, and the C test, ≈ 2 min
./test/test 8       # accuracy against the direct sum, timings, 8 threads
```

- `src/libnfftjl.jl`: the C entry points and the boundary that turns the run-time dimension D and
  window width W into one compiled branch per combination (D ∈ 1:3, W ∈ 4:16); plans live in
  registries of concrete type, a handle encodes (D, W, index).
- `include/nfftjl.h`: the interface, including the threading rules.
- `jc/`: the environment of JuliaC.jl.

Built with 8 Julia threads and Julia's signal handling (the GC stops threads through a signal;
without the handler the process dies once more than one thread runs).

Known limitations:
- several caller threads with threaded plans (`nthreads > 1`) can hang after a garbage
  collection (Julia runtime); several callers need `nthreads = 1`;
- the Julia runtime starts on the first call: make it from one thread before others call;
- FFTW.jl is not trimmable: `src/libnfftjl.jl` replaces three of its methods for the build;
- only NFFT, Float64, no batch dimensions, no GPU.
