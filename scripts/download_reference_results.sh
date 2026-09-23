#!/usr/bin/env bash
# Download and extract the Beyond PINNs reference-results archive
# (the complete trusted numerical outputs underlying the manuscript's tables
# and figures) into results/reference/, so that
# ./scripts/reproduce_tables.sh and ./scripts/reproduce_figures.sh work.
#
# The archive is a separate, versioned research-data deposit on Zenodo (see
# docs/REPRODUCIBILITY.md), DOI 10.5281/zenodo.22904196 -- it is NOT part of
# this code repository. By default this script downloads the v1.0 archive
# directly from its Zenodo file URL. Override with:
#
#   BEYOND_PINNS_RESULTS_URL=https://example.org/other-archive.tar.gz \
#       ./scripts/download_reference_results.sh
#   (a local path also works, e.g. BEYOND_PINNS_RESULTS_URL=/path/to/archive.tar.gz)
#
# or extract the archive yourself into results/reference/ (see the archive's
# own README.md for the exact layout) and skip this script.
set -euo pipefail
cd "$(dirname "$0")/.."

DEFAULT_URL="https://zenodo.org/records/22904196/files/beyond-pinns-results-v1.0.tar.gz?download=1"

DEST="results/reference"
URL="${BEYOND_PINNS_RESULTS_URL:-$DEFAULT_URL}"
SHA256_EXPECTED="${BEYOND_PINNS_RESULTS_SHA256:-}"

mkdir -p "$DEST"
if [ -n "$(find "$DEST" -mindepth 1 -not -name 'README.md' -print -quit 2>/dev/null)" ]; then
  echo "Refusing to overwrite: $DEST already contains data other than its own README.md."
  echo "Remove or back up its contents first if you want to re-extract."
  exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
ARCHIVE="$WORKDIR/archive"

if [ -f "$URL" ]; then
  echo "Using local archive: $URL"
  cp "$URL" "$ARCHIVE"
elif command -v curl >/dev/null 2>&1; then
  echo "Downloading via curl: $URL"
  curl -fL --progress-bar -o "$ARCHIVE" "$URL"
elif command -v wget >/dev/null 2>&1; then
  echo "Downloading via wget: $URL"
  wget -O "$ARCHIVE" "$URL"
else
  echo "Neither curl nor wget is available, and '$URL' is not a local file. Aborting."
  exit 1
fi

if [ -n "$SHA256_EXPECTED" ]; then
  echo "Verifying SHA256..."
  ACTUAL="$(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
  if [ "$ACTUAL" != "$SHA256_EXPECTED" ]; then
    echo "SHA256 MISMATCH: expected $SHA256_EXPECTED, got $ACTUAL. Aborting -- not extracting."
    exit 1
  fi
  echo "SHA256 OK."
else
  echo "BEYOND_PINNS_RESULTS_SHA256 not set -- skipping checksum verification."
fi

echo "Extracting into $DEST ..."
EXTRACT_DIR="$WORKDIR/extracted"
mkdir -p "$EXTRACT_DIR"
case "$ARCHIVE" in
  *) tar xf "$ARCHIVE" -C "$EXTRACT_DIR" ;;
esac

# The archive's top-level directory name may carry a version suffix
# (e.g. beyond-pinns-results-v1.0/); copy its data subdirectories in by
# name rather than assuming the archive extracts flat.
TOP="$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | head -1)"
SRC="${TOP:-$EXTRACT_DIR}"
for d in smooth general fem timings; do
  if [ -d "$SRC/$d" ]; then
    cp -r "$SRC/$d" "$DEST/"
  fi
done
[ -f "$SRC/MANIFEST.json" ] && cp "$SRC/MANIFEST.json" "$DEST/"

echo "Done. results/reference/ now contains:"
ls "$DEST"
echo
echo "Verify with: ./scripts/reproduce_tables.sh && ./scripts/reproduce_figures.sh"
