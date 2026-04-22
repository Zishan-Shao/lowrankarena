#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <repo_root>" >&2
  exit 1
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT="$1"
mkdir -p "$REPO_ROOT"
cp -a "$SCRIPT_DIR/files/." "$REPO_ROOT/"
echo "Copied bundled files into $REPO_ROOT"
