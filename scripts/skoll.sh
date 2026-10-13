#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
for config in "$@"; do
    log="logs/$(basename "$config" .yaml)-$(date +%Y%m%d-%H%M%S).log"
    nohup .venv/bin/madcal run "$config" > "$log" 2>&1 &
    echo "$config -> pid $! log $log"
done
