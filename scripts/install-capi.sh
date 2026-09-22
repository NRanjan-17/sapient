#!/usr/bin/env bash
# Build and install the SAPIENT C library: libsapient + sapient.h + pkg-config + CMake.
#
#   scripts/install-capi.sh [--prefix DIR] [--features wgpu]
#
# Default prefix is ./dist/capi so nothing touches the system without being asked.
# Install system-wide with: scripts/install-capi.sh --prefix /usr/local
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="$ROOT/dist/capi"
FEATURES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)   PREFIX="$2"; shift 2 ;;
        --features) FEATURES=(--features "$2"); shift 2 ;;
        -h|--help)  sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"

case "$(uname -s)" in
    Darwin) LIBS='-framework CoreFoundation -framework SystemConfiguration -framework Security -framework Metal -framework QuartzCore -lc++ -liconv'
            SHLIB=libsapient.dylib ;;
    *)      LIBS='-lpthread -ldl -lm'
            SHLIB=libsapient.so ;;
esac

echo "building sapient-capi ${VERSION} ${FEATURES[*]:-}"
(cd "$ROOT" && cargo build --release -p sapient-capi "${FEATURES[@]}")

mkdir -p "$PREFIX/lib/pkgconfig" "$PREFIX/include" "$PREFIX/lib/cmake/sapient"

install -m 0644 "$ROOT/crates/sapient-capi/include/sapient.h" "$PREFIX/include/"
install -m 0644 "$ROOT/target/release/libsapient.a"           "$PREFIX/lib/"
[[ -f "$ROOT/target/release/$SHLIB" ]] && install -m 0755 "$ROOT/target/release/$SHLIB" "$PREFIX/lib/"

sed -e "s|@PREFIX@|$PREFIX|g" -e "s|@VERSION@|$VERSION|g" -e "s|@PRIVATE_LIBS@|$LIBS|g" \
    "$ROOT/crates/sapient-capi/sapient.pc.in" > "$PREFIX/lib/pkgconfig/sapient.pc"

sed -e "s|@PREFIX@|$PREFIX|g" -e "s|@VERSION@|$VERSION|g" \
    "$ROOT/crates/sapient-capi/cmake/sapient-config.cmake.in" > "$PREFIX/lib/cmake/sapient/sapient-config.cmake"

cat <<EOF

installed to $PREFIX

  pkg-config:  export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:\$PKG_CONFIG_PATH"
               cc app.c \$(pkg-config --cflags --libs sapient)
  cmake:       find_package(sapient REQUIRED PATHS "$PREFIX/lib/cmake/sapient")
               target_link_libraries(app PRIVATE sapient::sapient)
EOF
