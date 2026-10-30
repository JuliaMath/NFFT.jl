# Benchmarks of NFFT.jl

Compares NFFT.jl with FINUFFT, ducc, NonuniformFFTs.jl and NFFT.jl 0.14, and NFFT.jl with any of
its own git revisions. The results in `results/` are the data of the
[performance page](https://JuliaMath.github.io/NFFT.jl/dev/performance/) of the documentation.

```
julia benchmark/bench.jl list                       # suites and libraries
julia benchmark/bench.jl run fast                   # NFFT.jl alone, ≈ 15 s
julia benchmark/bench.jl run quick                  # NFFT vs FINUFFT vs ducc, ≈ 1 min
julia benchmark/bench.jl run quick --libs NFFT --dims 2 --tols 1e-8
julia benchmark/bench.jl ab HEAD~1                  # working tree against a revision (suite fast), ≈ 1 min
julia benchmark/bench.jl compare out/a.csv out/b.csv
julia benchmark/bench.jl run all --save             # all suites of the documentation, ≈ 15 min
julia benchmark/bench.jl report                     # figures and explorer data for the docs
```

Run from the repository root with any Julia (the orchestrator uses the standard library only).
Options: `--libs`, `--dims`, `--tols`, `--T Float32`, `--threads 1,8`, `--budget` (seconds of
repeated calls per measurement, default 0.1), `--rounds` (`ab`; `run`: repeat all tasks, each round
in a new random order of cases and of the tasks within a case, with the round and its load average
in the CSV), `--save`.

## How it works

Each library runs in a worker process with its own environment (`envs/<lib>/Project.toml`,
instantiated on first use): competitors never become dependencies of NFFT.jl, libraries with
conflicting dependencies (NonuniformFFTs.jl needs AbstractNFFTs 0.9, NFFT.jl 0.14) can be compared,
and each library gets exactly the requested number of threads. `NFFT@<rev>` is NFFT.jl at a git
revision, checked out as a worktree in `.cache/`.

- `src/suites.jl`: the suites, each a list of tasks (library, transform, size, nodes, batch,
  precision, threads, tolerance).
- `worker/problems.jl`: nodes, data and an exact reference (NDFT on 64 sampled outputs), generated
  deterministically from the task, so every process sees the same problem.
- `worker/worker.jl`: plan (3 times, minimum), direct and adjoint transform (minimum and median of
  repeated calls within the budget), relative error against the reference, allocations, the
  parameters the library chose.
- `worker/adapters/<lib>.jl`: maps the convention of NFFT.jl (nodes in [-1/2, 1/2), modes in
  increasing order, e^{-2πi l·k} for the direct transform) onto each library. A wrong mapping shows
  up as an error of order one.
- `src/Benchmarks.jl`: orchestration, CSV files, `compare` and `ab`.
- `src/report.jl`: figures (`docs/src/assets/benchmarks/*.svg`) and `data.js` for the explorer on
  the performance page.

Results go to `out/<time>-<suite>.csv` (not versioned); `--save` also writes
`results/<machine>/<suite>.csv`, the data of the documentation. The CSV header records machine,
Julia version, NFFT.jl commit and command; each row the version and library file of the library.

Every library uses its defaults apart from the tolerance and the number of threads; FFTW plans use
`ESTIMATE`. The libraries interpret the tolerance differently, so compare time against the
measured error (`pareto`), not against the tolerance. NonuniformFFTs.jl has no tolerance
parameter; its half support comes from the tolerance as in NFFT.jl 0.14. ducc has no batched
transform; batches loop over the slices.

## Apple GPUs

`run metal` compares NFFT.jl on Metal (`storage = MtlArray`) with
[mlx-nufft](https://github.com/martinlachaine/mlx-nufft) (Python, MLX), both in Float32 with the
data on the GPU. mlx-nufft runs through PythonCall in a Python of your choice:

```
python3 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx-nufft
MLX_NUFFT_PYTHON=$PWD/.venv-mlx/bin/python julia benchmark/bench.jl run metal
```

Without `MLX_NUFFT_PYTHON` its tasks are skipped. GPU timings include a synchronisation after
every call.

## Native builds of the competitors

By default FINUFFT and ducc come from their release binaries (JLL packages). To benchmark builds
with native CPU flags, point the JLL packages to them in `envs/finufft/LocalPreferences.toml` and
`envs/ducc/LocalPreferences.toml` (not versioned):

```toml
[finufft_jll]
libfinufft_path = "/path/to/finufft/build/src/libfinufft.dylib"
```

```toml
[ducc0_jll]
libducc_julia_path = "/path/to/libducc_julia.dylib"
```

The library that was loaded is recorded in each row (`lib_library`).

"FINUFFT 2.5" (`envs/finufft25`) is a second FINUFFT for the suite `decomposition`: a build of
v2.5.1 next to the current one, with its path in `envs/finufft25/LocalPreferences.toml`. Its
CMake links the shared library with the OpenMP flags as one argument, which clang rejects; link
the objects by hand:

```
cmake -S finufft-2.5.1 -B build -DCMAKE_BUILD_TYPE=Release -DFINUFFT_ARCH_FLAGS=-mcpu=native -DFINUFFT_STATIC_LINKING=OFF
cmake --build build                    # fails at the last link step
cd build/src && c++ -dynamiclib -o libfinufft.dylib -install_name @rpath/libfinufft.dylib \
  CMakeFiles/finufft.dir/*.o common/CMakeFiles/finufft_common.dir/*.o \
  -L/opt/homebrew/opt/fftw/lib -lfftw3 -lfftw3f -lfftw3_omp -lfftw3f_omp -L/opt/homebrew/opt/libomp/lib -lomp
```

## Suite `decomposition`

Splits a speed ratio NFFT.jl : X into a parameter factor T_jl(rule X)/T_jl(own) and an
implementation factor T_X/T_jl(rule X). `src/rules.jl` gives, for FINUFFT 2.5.1 and 2.6-dev,
cuFINUFFT 2.5.1, ducc 0.29, NonuniformFFTs.jl 0.9.8, NFFT3 and mlx-nufft 0.3.3, the window, W and
grid the library would choose (each rule cites its source); NFFT.jl runs with them through a
`CustomWindow`. A task with `rule` records the rule's parameters (`rule_*`), what the library
reports (`rep_*`: the debug output of FINUFFT, ducc's verbose report, the parameters of
NonuniformFFTs.jl, or the grid NFFT.jl built) and their differences (`rule_dev`). A grid with
factors 7 or 11 (ducc) rounds up to the next one from {2, 3, 5} in NFFT.jl (`grid_rounded`).
`tol_used` is the tolerance a library got after its adapter raised or clamped it. NFFT.jl with its
own choice runs on half decades of tolerance, so that its time can be interpolated at the error
each original delivers.

```
julia benchmark/bench.jl run decomposition --rounds 5     # 675 tasks per round
```
