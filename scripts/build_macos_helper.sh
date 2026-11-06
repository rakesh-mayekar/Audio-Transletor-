#!/usr/bin/env bash
# Builds the ScreenCaptureKit helper. Requires Xcode command line tools (xcode-select --install).
set -euo pipefail
cd "$(dirname "$0")/.."
swiftc -O oslayer/macos/helper/AudioTap.swift -o oslayer/macos/olt-audio-tap
echo "built oslayer/macos/olt-audio-tap"
