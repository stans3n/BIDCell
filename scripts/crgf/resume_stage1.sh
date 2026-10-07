#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The runner rechecks flock, host/boot/start-ticks child identity, source/config/inputs,
# completed receipt hashes and checkpoint dependencies before accepting resume.
exec "$HERE/launch_stage1.sh" "${1:?Usage: resume_stage1.sh /absolute/run.json}"
