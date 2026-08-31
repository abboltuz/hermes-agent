#!/bin/sh
# post-merge hook — re-install the hermes-claude-auth sitecustomize loader
# after every `hermes update` (which runs `git merge --ff-only` + `uv pip
# install`).  The loader lives in the venv's site-packages and is NOT part
# of the hermes-agent repo, so a plain `git pull` would leave it untouched —
# but `uv pip install -e .` can recreate the venv and wipe it.  This hook
# restores it from a stable location outside the git checkout.
#
# Idempotent and never fatal: any failure is logged to stderr and ignored so
# it can never break the update itself.

set -u

# --- resolve the Hermes data root (patches dir holds the canonical loader) ---
if [ -n "${HERMES_HOME:-}" ]; then
  HERMES_ROOT="$HERMES_HOME"
  # HERMES_HOME may point at a *profile* dir (.../hermes/profiles/<name>);
  # the patches live at the data root, two levels up.
  case "$(basename "$(dirname "$HERMES_HOME")")" in
    profiles) HERMES_ROOT="$(dirname "$(dirname "$HERMES_HOME")")" ;;
  esac
elif [ -n "${LOCALAPPDATA:-}" ]; then
  HERMES_ROOT="$LOCALAPPDATA/hermes"
else
  HERMES_ROOT="$HOME/.hermes"
fi

SRC="$HERMES_ROOT/patches/sitecustomize.py"
echo "[post-merge] HERMES_ROOT resolved to: $HERMES_ROOT" >&2

# --- resolve the venv that Hermes actually uses -------------------------------
# With core.hooksPath pointing outside the repo, $0 is no longer inside the
# hermes-agent tree, so derive REPO_DIR from HERMES_HOME / known locations.
if [ -n "${HERMES_HOME:-}" ]; then
  case "$(basename "$(dirname "$HERMES_HOME")")" in
    profiles) AGENT_DIR="$(dirname "$(dirname "$HERMES_HOME")")/hermes-agent" ;;
    *) AGENT_DIR="$HERMES_HOME/hermes-agent" ;;
  esac
elif [ -n "${LOCALAPPDATA:-}" ]; then
  AGENT_DIR="$LOCALAPPDATA/hermes/hermes-agent"
else
  AGENT_DIR="$HOME/.hermes/hermes-agent"
fi

REPO_DIR="$AGENT_DIR"
VENV_DIR=""
for cand in "$REPO_DIR/venv" "$REPO_DIR/.venv"; do
  if [ -d "$cand" ]; then
    VENV_DIR="$cand"
    break
  fi
done

if [ ! -f "$SRC" ]; then
  echo "[post-merge] hermes-claude-auth loader not found at $SRC — skipping" >&2
  exit 0
fi

if [ -z "$VENV_DIR" ]; then
  echo "[post-merge] no hermes venv found (looked in $REPO_DIR/venv, .venv) — skipping" >&2
  exit 0
fi

DEST="$VENV_DIR/Lib/site-packages/sitecustomize.py"

# Only overwrite if it is (or would be) the claude-auth loader.
if [ -f "$DEST" ] && ! grep -q "hermes-claude-auth managed" "$DEST" 2>/dev/null; then
  echo "[post-merge] $DEST is not the claude-auth loader — refusing to overwrite" >&2
  exit 0
fi

if cp -f "$SRC" "$DEST" 2>/dev/null; then
  echo "[post-merge] hermes-claude-auth loader restored to $DEST" >&2
else
  echo "[post-merge] failed to copy loader to $DEST — ignored" >&2
fi

exit 0
