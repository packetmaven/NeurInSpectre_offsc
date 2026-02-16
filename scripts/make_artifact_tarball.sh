#!/bin/bash
set -euo pipefail

# Create a deterministic source tarball from the current git commit and print SHA256.
# Intended for paper/artifact metadata (do not commit the generated tarball).

OUT_DIR="${OUT_DIR:-dist}"
mkdir -p "$OUT_DIR"

COMMIT_FULL="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short=12 HEAD)"

TARBALL="$OUT_DIR/NeurInSpectre_${COMMIT_SHORT}.tar.gz"

git archive \
  --format=tar.gz \
  --prefix="NeurInSpectre-${COMMIT_SHORT}/" \
  "$COMMIT_FULL" \
  -o "$TARBALL"

if command -v sha256sum >/dev/null 2>&1; then
  SHA="$(sha256sum "$TARBALL" | awk '{print $1}')"
else
  SHA="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
fi

echo "${SHA}  $(basename "$TARBALL")" | tee "${TARBALL}.sha256" >/dev/null
echo "$TARBALL"
echo "SHA256: $SHA"

