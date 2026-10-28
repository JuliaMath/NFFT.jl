#!/bin/bash
# Build build/lib/libnfftjl with juliac --trim (JuliaC.jl in its own environment jc/), the C
# test test/test and the examples against it. Needs Julia ≥ 1.12.
set -e
cd "$(dirname "$0")"
julia --project=jc -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.instantiate()'
rm -rf build
julia --project=jc -m JuliaC --output-lib build/libnfftjl --project . --bundle build \
  --trim=safe --compile-ccallable --experimental \
  --jl-option threads=8 --jl-option handle-signals=yes "$@" src/libnfftjl.jl
cc -O2 -o test/test test/test.c -Iinclude -Lbuild/lib -lnfftjl -lm -Wl,-rpath,"$(pwd)/build/lib"
for src in examples/*.c; do
  cc -O2 -o "${src%.c}" "$src" -Iinclude -Lbuild/lib -lnfftjl -lm -Wl,-rpath,"$(pwd)/build/lib"
done
