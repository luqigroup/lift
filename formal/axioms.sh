#!/usr/bin/env bash
# Print the axiom dependencies of every theorem in a file.  A clean result lists only
# propext, Classical.choice, Quot.sound -- never sorryAx.
#   ./axioms.sh Foo.lean
set -uo pipefail
cd "$(dirname "$0")"
f="$1"; base="${f%.lean}"
ns=$(grep -m1 '^namespace ' "$f" | awk '{print $2}')
tmp=$(mktemp /tmp/axcheck_XXXX.lean)
{ echo "import $base"; [ -n "$ns" ] && echo "open $ns"
  grep -oP '^(theorem|lemma)\s+\K[A-Za-z_][A-Za-z0-9_'"'"'.]*' "$f" | while read -r t; do echo "#print axioms $t"; done
} > "$tmp"
cp "$tmp" "./$(basename "$tmp")"
./check.sh "$(basename "$tmp")"
rm -f "./$(basename "$tmp")" "$tmp"
