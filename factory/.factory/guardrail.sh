#!/usr/bin/env bash
# Factory guardrail — PreToolUse hook.
#
# Receives a JSON event on stdin describing the tool call about to happen.
# Exit 2 + stderr message = block the call and surface the reason to Claude.
# Exit 0 = allow.
#
# Rules (apply in this order):
#   1. Any phase: deny writes/edits/notebook-edits whose file_path is under
#      factory/scenarios/holdout/** or factory/.factory/**.
#   2. Any phase: deny Bash commands that look destructive (rm -rf, --force,
#      git push --force / -f, history-rewriting git, filesystem nukes).
#   3. FACTORY_PHASE=builder: also deny Read/Glob/Grep against
#      factory/scenarios/holdout/**, and deny Bash that reads holdout paths.
#      The judge phase MAY read holdout but still cannot write there.
#
# Anything we cannot parse: allow + warn on stderr. The other defenses
# (tool-permission denies, the loop's own logic) still apply.

set -u

FACTORY_HOOK_INPUT="$(cat)"
export FACTORY_HOOK_INPUT
export FACTORY_PHASE="${FACTORY_PHASE:-unspecified}"

decision="$(python3 - <<'PY'
import json, os, re, sys

raw = os.environ.get("FACTORY_HOOK_INPUT", "")
try:
    ev = json.loads(raw)
except Exception as e:
    print(f"ALLOW\tguardrail: could not parse hook input ({e})", end="")
    sys.exit(0)

phase = os.environ.get("FACTORY_PHASE", "unspecified")
tool  = ev.get("tool_name", "")
ti    = ev.get("tool_input") or {}

HOLDOUT_RE = re.compile(r"(^|/)factory/scenarios/holdout(/|$)")
DOTFAC_RE  = re.compile(r"(^|/)factory/\.factory(/|$)")

# Bootstrap carve-out: when `factory/.bootstrap_unlock` exists, the bootstrap
# session is allowed to seed factory/scenarios/holdout/. .factory/ is NEVER
# unlocked, nor are destructive commands. The sentinel must be removed once
# seeding is done.
def bootstrap_unlocked():
    try:
        cwd = ev.get("cwd") or os.getcwd()
        return os.path.exists(os.path.join(cwd, "factory", ".bootstrap_unlock"))
    except Exception:
        return False

def block(msg):
    print(f"BLOCK\t{msg}", end="")
    sys.exit(0)

def allow():
    print("ALLOW", end="")
    sys.exit(0)

def path_blocked_for_write(p):
    if not isinstance(p, str):
        return None
    if HOLDOUT_RE.search(p):
        return "writes under factory/scenarios/holdout/ are sealed — judge-only territory"
    if DOTFAC_RE.search(p):
        return "writes under factory/.factory/ are off-limits to task work"
    return None

# --- Rule 1: write-like tools ---
if tool in ("Write", "Edit", "NotebookEdit"):
    fp = ti.get("file_path", "")
    why = path_blocked_for_write(fp)
    if why:
        # Bootstrap may seed holdout scenarios; .factory/ stays sealed always.
        if "holdout" in why and bootstrap_unlocked():
            sys.stderr.write(
                "[factory guardrail] holdout write allowed via .bootstrap_unlock sentinel\n"
            )
        else:
            block(f"{tool} to {fp!r} blocked: {why}")

# --- Rule 3a (builder only): read-like tools against holdout or .factory ---
# .factory/ holds VALIDATE_CMD and other loop wiring; reading it would leak
# the holdout's probe inputs to the builder.
if phase == "builder" and tool in ("Read", "Glob", "Grep"):
    target = ti.get("file_path") or ti.get("path") or ti.get("pattern") or ""
    if isinstance(target, str):
        if HOLDOUT_RE.search(target):
            block(f"{tool} against {target!r} blocked: holdout is sealed from the builder")
        if DOTFAC_RE.search(target):
            block(f"{tool} against {target!r} blocked: factory/.factory/ is sealed from the builder (would leak holdout probe inputs)")

# --- Rule 2 + 3b: Bash command inspection ---
if tool == "Bash":
    cmd = ti.get("command", "") or ""
    c = " " + re.sub(r"\s+", " ", cmd).strip() + " "

    # 2a. Destructive filesystem.
    if re.search(r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r|-rf|-fr)\b", c):
        block("destructive command blocked: `rm -rf`-style deletion is not allowed from the loop")

    # 2b. Force / history rewrites.
    if re.search(r"\bgit\s+push\b.*(--force(-with-lease)?|\s-f\b)", c):
        block("destructive command blocked: force-push is not allowed from the loop")
    if re.search(r"\bgit\s+reset\s+--hard\b", c):
        block("destructive command blocked: `git reset --hard` is not allowed from the loop")
    if re.search(r"\bgit\s+(filter-branch|filter-repo)\b", c):
        block("destructive command blocked: history rewrite is not allowed from the loop")
    if re.search(r"\bgit\s+rebase\b.*\s-i\b", c) or re.search(r"\bgit\s+rebase\s+-i\b", c):
        block("destructive command blocked: interactive rebase is not allowed from the loop")
    if re.search(r"\bgit\s+commit\b.*--amend\b", c):
        block("destructive command blocked: `git commit --amend` is not allowed from the loop")
    if re.search(r"\bgit\s+clean\b.*\s-[a-zA-Z]*f", c):
        block("destructive command blocked: `git clean -f` is not allowed from the loop")

    # 2c. Writes via shell into protected dirs.
    write_verbs = r"(?:>{1,2}|\btee\b|\bcp\b|\bmv\b|\bmkdir\b|\brmdir\b|\btouch\b|\bsed\s+-i\b|\bln\s+-s\b)"
    if re.search(write_verbs + r"[^|]*factory/scenarios/holdout(/|\b)", c):
        if bootstrap_unlocked():
            sys.stderr.write(
                "[factory guardrail] holdout write allowed via .bootstrap_unlock sentinel\n"
            )
        else:
            block("Bash write into factory/scenarios/holdout/ blocked: holdout is sealed")
    if re.search(write_verbs + r"[^|]*factory/\.factory(/|\b)", c):
        block("Bash write into factory/.factory/ blocked: not editable as part of task work")

    # 3b. Builder phase: also block shell reads of holdout AND .factory/.
    if phase == "builder":
        read_verbs = r"(?:\bcat\b|\bhead\b|\btail\b|\bless\b|\bmore\b|\bnl\b|\bgrep\b|\bawk\b|\bsed\b|\bls\b|\bfind\b|\brg\b|\bxxd\b|\bod\b|\bstrings\b)"
        if re.search(read_verbs + r"[^|]*factory/scenarios/holdout(/|\b)", c):
            block("Bash read of factory/scenarios/holdout/ blocked: builder phase may not observe the holdout")
        if re.search(read_verbs + r"[^|]*factory/\.factory(/|\b)", c):
            block("Bash read of factory/.factory/ blocked: would leak holdout probe inputs to the builder")

allow()
PY
)"

if [[ -z "$decision" ]]; then
  echo "guardrail: helper returned empty decision; allowing by default" >&2
  exit 0
fi

verdict="${decision%%$'\t'*}"
reason="${decision#*$'\t'}"

if [[ "$verdict" == "BLOCK" ]]; then
  echo "[factory guardrail | phase=$FACTORY_PHASE] $reason" >&2
  exit 2
fi

exit 0
