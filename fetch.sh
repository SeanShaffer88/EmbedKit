#!/bin/sh
# Download the pinned fixtures into fixtures/ and verify SHA-256:
# the ONNX Runtime package for this host, plus the e2e model + vocab.
# Nothing fetched here is ever committed; fixtures/ is gitignored, and
# ONNX Runtime is a runtime component the DLL loads dynamically — the
# fetch exists for the e2e proof and as a known-good source for users.
set -eu
cd "$(dirname "$0")"
# shellcheck disable=SC1091
. ./pins.txt
mkdir -p fixtures

sha_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

fetch() {
    url="$1"; file="$2"; want="$3"
    if [ ! -f "fixtures/$file" ]; then
        echo "downloading $url"
        curl -fL "$url" -o "fixtures/$file"
    fi
    got=$(sha_of "fixtures/$file")
    if [ "$got" != "$want" ]; then
        echo "SHA-256 MISMATCH for $file" >&2
        echo "  pinned: $want" >&2
        echo "  actual: $got" >&2
        exit 1
    fi
    echo "verified $file ($got)"
}

case "$(uname -s)" in
    Linux)  pkg_file="$ORT_LINUX_FILE"; pkg_url="$ORT_LINUX_URL"; pkg_sha="$ORT_LINUX_SHA256" ;;
    *)
        echo "fetch.sh supports Linux hosts; on Windows use fetch.ps1." >&2
        echo "Other platforms: download ONNX Runtime $ORT_VERSION from" >&2
        echo "  https://github.com/microsoft/onnxruntime/releases/tag/v$ORT_VERSION" >&2
        exit 1
        ;;
esac

fetch "$pkg_url" "$pkg_file" "$pkg_sha"
fetch "$TEST_MODEL_URL" "$TEST_MODEL_NAME" "$TEST_MODEL_SHA256"
fetch "$TEST_VOCAB_URL" "$TEST_VOCAB_NAME" "$TEST_VOCAB_SHA256"

# Extract the runtime package into fixtures/ort (stable layout for the
# e2e test: fixtures/ort/lib/libonnxruntime.so.$ORT_VERSION on Linux).
rm -rf fixtures/ort
mkdir -p fixtures/ort
tar --no-same-owner -xzf "fixtures/$pkg_file" -C fixtures/ort --strip-components=1
echo "runtime ready under fixtures/ort; model at fixtures/$TEST_MODEL_NAME; vocab at fixtures/$TEST_VOCAB_NAME"
echo "e2e: EK_ORT_LIB=fixtures/ort/lib/libonnxruntime.so.$ORT_VERSION EK_MODEL=fixtures/$TEST_MODEL_NAME EK_VOCAB=fixtures/$TEST_VOCAB_NAME cargo test --test e2e -- --ignored --nocapture"
