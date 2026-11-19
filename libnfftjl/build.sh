#!/bin/bash
# Build build/lib/libnfftjl with juliac --trim (JuliaC.jl in its own environment jc/), the C
# test test/test and the examples against it. Needs Julia ≥ 1.12.
set -e
cd "$(dirname "$0")"
julia --project=jc -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.instantiate()'
# One CPU target: a library built for several targets (Julia's default) hangs at the first garbage
# collection triggered by a second host thread, https://github.com/JuliaLang/JuliaC.jl/issues/190.
if [ -z "$JULIA_CPU_TARGET" ]; then
  case "$(uname -m)" in
    arm64|aarch64) [ "$(uname -s)" = Darwin ] && export JULIA_CPU_TARGET=apple-m1 || export JULIA_CPU_TARGET=generic ;;
    x86_64) export JULIA_CPU_TARGET=x86-64-v3 ;;
    *) export JULIA_CPU_TARGET=generic ;;
  esac
fi
# FFTL picks its vector width from the build host. The x86 target is x86-64-v3 (AVX2): on a host
# with AVX-512, LLVM cannot lower FFTL's 512-bit kernels for it, so they are limited to 256 bits.
case "$(uname -m)" in
  x86_64) printf '[FFTL]\nsimd_bytes = 32\n' > LocalPreferences.toml ;;
  *) rm -f LocalPreferences.toml ;;
esac
rm -rf build
julia --project=jc -m JuliaC --output-lib build/libnfftjl --project . --bundle build \
  --trim=safe --compile-ccallable --experimental \
  --jl-option threads=auto --jl-option handle-signals=yes "$@" src/libnfftjl.jl
# Libraries of the Julia installation that the library never loads (checked with
# DYLD_PRINT_LIBRARIES / LD_DEBUG=libs): BLAS (see src/libnfftjl.jl), networking, git, C++.
for lib in openblas gfortran quadmath gomp crypto ssl git2 ssh2 curl nghttp2 stdc++ ssp gmpxx dSFMT atomic; do
  rm -f build/lib/julia/lib${lib}*.{so,dylib,dll} build/lib/julia/lib${lib}*.so.* build/bin/lib${lib}*.dll
done
cc -O2 -o test/test test/test.c -Iinclude -Lbuild/lib -lnfftjl -lm -Wl,-rpath,"$(pwd)/build/lib"
for src in examples/*.c; do
  cc -O2 -o "${src%.c}" "$src" -Iinclude -Lbuild/lib -lnfftjl -lm -Wl,-rpath,"$(pwd)/build/lib"
done
