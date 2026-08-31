#!/bin/sh
# Restore hermes-claude-auth loader if missing from the venv after hermes update.
# Runs via Hermes cron (every 15m). Idempotent: only acts if loader is missing
# or has been replaced by a non-claude-auth sitecustomize.py.

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) LOCAL="${LOCALAPPDATA:-/c/Users/ronal/AppData/Local}" ;;
  *) LOCAL="${XDG_DATA_HOME:-$HOME/.local/share}" ;;
esac

LOADERDST="$LOCAL/Hermes/hermes-agent/venv/Lib/site-packages/sitecustomize.py"
LOADERSRC="$LOCAL/hermes/patches/sitecustomize.py"
LOG="$LOCAL/hermes/logs/restore_loader.log"

if [ ! -f "$LOADERDST" ] || ! grep -q "hermes-claude-auth managed" "$LOADERDST" 2>/dev/null; then
  if [ -f "$LOADERSRC" ]; then
    cp -f "$LOADERSRC" "$LOADERDST" 2>/dev/null \
      && echo "[restore_loader] $(date) loader restored (was missing/replaced)" >> "$LOG" 2>/dev/null
  fi
fi
exit 0
