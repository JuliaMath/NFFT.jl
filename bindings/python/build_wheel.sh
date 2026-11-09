#!/bin/bash
# Wheel with the library of ../../libnfftjl/build (run ../../libnfftjl/build.sh first) → dist/.
set -e
cd "$(dirname "$0")"
rm -rf src/nfftjl/_lib build dist
cp -R ../../libnfftjl/build src/nfftjl/_lib
"${PYTHON:-python3}" -m build --wheel --no-isolation
