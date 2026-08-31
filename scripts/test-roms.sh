#!/usr/bin/env bash
set -euo pipefail

fixture_dir="$(mktemp -d)"
trap 'rm -rf "$fixture_dir"' EXIT

git clone --depth 1 https://github.com/retrio/gb-test-roms "$fixture_dir/gb-test-roms"
for rom in "$fixture_dir"/gb-test-roms/cpu_instrs/individual/*.gb; do
  cargo run --quiet --release --bin gbsml-headless -- "$rom" --frames 1800
done
