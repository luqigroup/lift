#!/usr/bin/env bash
# Type-check ONE Lean file against the prebuilt mathlib, without taking lake's
# build lock -- so several agents can check their own files in parallel.
#   ./check.sh Foo.lean
# Exit 0 = the Lean kernel accepted the file.
set -uo pipefail
cd "$(dirname "$0")"
P="$PWD/.lake/packages"
export LEAN_PATH="$P/Cli/.lake/build/lib/lean:$P/batteries/.lake/build/lib/lean:$P/Qq/.lake/build/lib/lean:$P/aesop/.lake/build/lib/lean:$P/proofwidgets/.lake/build/lib/lean:$P/importGraph/.lake/build/lib/lean:$P/LeanSearchClient/.lake/build/lib/lean:$P/plausible/.lake/build/lib/lean:$P/mathlib/.lake/build/lib/lean:$PWD/.lake/build/lib/lean"
exec "$HOME/.elan/toolchains/leanprover--lean4---v4.31.0/bin/lean" "$@"
