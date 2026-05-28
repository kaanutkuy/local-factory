#!/usr/bin/env bash
# Factory Stop hook — idempotently ensure today's date marker exists in
# factory/memory/log.md. Running it twice in one day must not add two markers.

set -u

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
LOG="$REPO_ROOT/factory/memory/log.md"

if [[ ! -f "$LOG" ]]; then
  # No log file yet — bail quietly, do not invent factory state from a hook.
  exit 0
fi

today="$(date +%Y-%m-%d)"
marker="## ${today}"

if ! grep -qxF "$marker" "$LOG" 2>/dev/null; then
  {
    echo ""
    echo "$marker"
    echo ""
  } >> "$LOG"
fi

exit 0
