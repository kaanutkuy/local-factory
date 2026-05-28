#!/usr/bin/env bash
# factory/loop.sh — scenario-first build/judge loop.
#
# Usage:
#   factory/loop.sh <task-name>            # task at factory/tasks/<task-name>.md
#   factory/loop.sh <task-name> --one-iter # one iteration then exit (for gating)
#
# Posture (non-negotiable):
#   - Builder runs with FACTORY_PHASE=builder; the guardrail blocks any read of
#     factory/scenarios/holdout/** for it.
#   - Judge runs with FACTORY_PHASE=judge; it may read holdout, never write there.
#   - The loop never commits. It stages and shows a diff summary; human reviews.
#   - Rate/billing halts: detected from claude's JSON envelope, state is saved,
#     loop exits cleanly (never fakes a scenario failure for a billing event).
#   - State is the single source of truth: factory/runs/state.json.

set -u
set -o pipefail

# ---------- paths & config ----------

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

FACTORY_DIR="factory"
CONF="$FACTORY_DIR/.factory/config"
STATE="$FACTORY_DIR/runs/state.json"
SUMMARY_DIR="$FACTORY_DIR/runs/summaries"
LOG="$FACTORY_DIR/memory/log.md"

if [[ ! -f "$CONF" ]]; then
  echo "loop: missing $CONF" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONF"
: "${MAX_ITERS:=8}"
: "${THRESHOLD:=0.9}"
: "${VALIDATE_CMD:=}"

ONE_ITER=0
TASK_NAME=""
for arg in "$@"; do
  case "$arg" in
    --one-iter) ONE_ITER=1 ;;
    --*)        echo "loop: unknown flag $arg" >&2; exit 1 ;;
    *)          TASK_NAME="$arg" ;;
  esac
done
if [[ -z "$TASK_NAME" ]]; then
  echo "loop: usage: $0 <task-name> [--one-iter]" >&2
  exit 1
fi
TASK_FILE="$FACTORY_DIR/tasks/${TASK_NAME}.md"
if [[ ! -f "$TASK_FILE" ]]; then
  echo "loop: no task file at $TASK_FILE" >&2
  exit 1
fi

mkdir -p "$SUMMARY_DIR" "$(dirname "$STATE")"

# ---------- helpers ----------

today() { date +%Y-%m-%d; }

log_append() {
  # log_append <markdown-block>
  printf '\n%s\n' "$1" >> "$LOG"
}

state_get() {
  # state_get <key>  — empty if state file or key missing
  if [[ ! -f "$STATE" ]]; then echo ""; return 0; fi
  python3 - "$STATE" "$1" <<'PY' 2>/dev/null
import json, sys
try:
    with open(sys.argv[1]) as f: s = json.load(f)
    v = s.get(sys.argv[2], "")
    print("" if v is None else v)
except Exception:
    print("")
PY
}

state_write() {
  # state_write iter last_model halted_reason last_score task
  python3 - "$STATE" "$1" "$2" "$3" "$4" "$5" <<'PY'
import json, sys
path = sys.argv[1]
state = {
    "iter": int(sys.argv[2]),
    "last_model": sys.argv[3],
    "halted_reason": sys.argv[4],
    "last_score": (None if sys.argv[5] == "" else float(sys.argv[5])),
    "task": sys.argv[6],
}
with open(path, "w") as f: json.dump(state, f, indent=2)
PY
}

# Concatenate visible scenarios for the builder prompt.
read_visible_scenarios() {
  shopt -s nullglob
  local f any=0
  for f in "$FACTORY_DIR"/scenarios/*.md "$FACTORY_DIR"/scenarios/*.txt; do
    any=1
    printf '\n----- %s -----\n' "$f"
    cat "$f"
  done
  if [[ "$any" -eq 0 ]]; then
    printf '(no visible scenarios)\n'
  fi
  shopt -u nullglob
}

# Concatenate holdout scenarios (judge only).
read_holdout_scenarios() {
  shopt -s nullglob
  local f any=0
  for f in "$FACTORY_DIR"/scenarios/holdout/*.md "$FACTORY_DIR"/scenarios/holdout/*.txt; do
    any=1
    printf '\n----- %s -----\n' "$f"
    cat "$f"
  done
  if [[ "$any" -eq 0 ]]; then
    printf '(no holdout scenarios)\n'
  fi
  shopt -u nullglob
}

# Last ~80 lines of memory/log.md for recent context.
recent_log() {
  tail -n 80 "$LOG" 2>/dev/null || true
}

# Extract halt reason from a claude -p JSON envelope. Echoes one of:
#   ""               OK
#   "rate_limit"     halted, save state and exit cleanly
#   "billing_error"  halted, save state and exit cleanly
#   "other_error"    parse/runtime error — loop should treat as iteration failure
extract_halt() {
  python3 - "$1" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path) as f: env = json.load(f)
except Exception:
    print("other_error"); sys.exit(0)
if not env.get("is_error", False):
    print(""); sys.exit(0)
sub = (env.get("subtype") or "").lower()
api = (env.get("api_error_status") or "").lower()
blob = (sub + " " + api + " " + json.dumps(env)).lower()
if "rate" in sub or "rate_limit" in blob or "rate-limit" in blob or "429" in blob:
    print("rate_limit"); sys.exit(0)
if "billing" in sub or "credit" in blob or "billing" in blob or "402" in blob or "insufficient" in blob:
    print("billing_error"); sys.exit(0)
print("other_error")
PY
}

extract_result() {
  python3 - "$1" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f: env = json.load(f)
    print(env.get("result", ""))
except Exception:
    pass
PY
}

# Pull the first valid {min_satisfaction,...} JSON object out of a result text.
# Echoes the canonical JSON line (one line). Empty echo = parse failed.
parse_judge_result() {
  local text
  text="$(cat)"
  FACTORY_JUDGE_TEXT="$text" python3 - <<'PY'
import json, re, sys, os
txt = os.environ.get("FACTORY_JUDGE_TEXT", "")
# Strip code fences and extract the largest balanced {...} block.
txt = re.sub(r"```[a-zA-Z]*", "", txt)
txt = txt.replace("```", "")
start_idx = txt.find("{")
if start_idx < 0:
    sys.exit(0)
depth = 0
end = -1
in_str = False
esc = False
for i in range(start_idx, len(txt)):
    ch = txt[i]
    if in_str:
        if esc: esc = False
        elif ch == "\\": esc = True
        elif ch == '"': in_str = False
        continue
    if ch == '"':
        in_str = True
    elif ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0:
            end = i + 1
            break
if end < 0:
    sys.exit(0)
try:
    obj = json.loads(txt[start_idx:end])
except Exception:
    sys.exit(0)
ms = obj.get("min_satisfaction")
if not isinstance(ms, (int, float)):
    sys.exit(0)
ms = float(ms)
if not (0.0 <= ms <= 1.0):
    sys.exit(0)
failing = obj.get("failing") or []
if not isinstance(failing, list):
    failing = []
notes = obj.get("notes") or ""
if not isinstance(notes, str):
    notes = str(notes)
print(json.dumps({"min_satisfaction": ms, "failing": failing, "notes": notes}))
PY
}

last_model() {
  python3 - "$1" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f: env = json.load(f)
    mu = env.get("modelUsage") or {}
    if not mu: print(""); sys.exit(0)
    print(sorted(mu.keys())[0])
except Exception:
    print("")
PY
}

# ---------- iteration logic ----------

ITER="$(state_get iter)"; ITER="${ITER:-0}"
LAST_REASON="$(state_get halted_reason)"; LAST_REASON="${LAST_REASON:-}"
LAST_SCORE_PREV="$(state_get last_score)"; LAST_SCORE_PREV="${LAST_SCORE_PREV:-}"

# If the previous run halted on billing/rate, the human is expected to clear
# that condition (or accept the pause) before re-running. We resume regardless,
# but we note it.
if [[ "$LAST_REASON" == "rate_limit" || "$LAST_REASON" == "billing_error" ]]; then
  echo "loop: resuming after prior halt ($LAST_REASON), iter=$ITER" >&2
fi

# Build a stable input bundle the builder & judge will both read from disk.
BUNDLE_DIR="$(mktemp -d)"
trap 'rm -rf "$BUNDLE_DIR"' EXIT

TASK_TEXT="$(cat "$TASK_FILE")"
VIS_SCEN="$(read_visible_scenarios)"
HOLD_SCEN="$(read_holdout_scenarios)"
RECENT="$(recent_log)"

# Run one iteration. Exits the loop (via state + return code) on terminal
# conditions; returns 0 to continue.
run_iteration() {
  local i="$1"
  echo "===== iteration $i ====="

  # ---- BUILDER ----
  local builder_prompt builder_out builder_envelope halt
  builder_prompt="$(cat <<EOF
You are the builder for this iteration of a scenario-first factory.

Hard rules (also in factory/CLAUDE.md):
- You may NOT read, list, or otherwise observe factory/scenarios/holdout/.
  The guardrail will block any attempt; do not try.
- You may NOT edit factory/.factory/ or factory/runs/.
- You may NOT commit. Just edit files. The loop will stage and the human will review.
- Make the SMALLEST change that plausibly moves the failing holdout scenarios forward.

Task (factory/tasks/${TASK_NAME}.md):
${TASK_TEXT}

Visible scenarios (builder may read):
${VIS_SCEN}

Recent factory log (carries judge feedback from prior iterations):
${RECENT}

When you are done editing, write ONE LINE to factory/memory/log.md describing
what you changed and the reasoning, then stop.
EOF
)"

  builder_envelope="$BUNDLE_DIR/builder.json"
  FACTORY_PHASE=builder claude -p "$builder_prompt" \
      --output-format json \
      --permission-mode acceptEdits \
      --allowedTools Read,Edit,Write,Bash \
      > "$builder_envelope" 2>/dev/null || true

  halt="$(extract_halt "$builder_envelope")"
  if [[ "$halt" == "rate_limit" || "$halt" == "billing_error" ]]; then
    log_append "- iter ${i} PAUSED: builder halted with reason \"$halt\". State saved."
    state_write "$i" "$(last_model "$builder_envelope")" "$halt" "$LAST_SCORE_PREV" "$TASK_NAME"
    echo "loop: PAUSED ($halt). State saved at $STATE." >&2
    exit 0
  fi
  if [[ "$halt" == "other_error" ]]; then
    log_append "- iter ${i}: builder call errored (not a halt). Will retry next iteration. Envelope at $builder_envelope (gone after exit)."
    state_write "$i" "$(last_model "$builder_envelope")" "builder_error" "$LAST_SCORE_PREV" "$TASK_NAME"
    echo "loop: builder errored; advancing iteration with no edits applied" >&2
    return 0
  fi

  # ---- STAGE & SUMMARIZE (no commit) ----
  git -C "$REPO_ROOT" add -A
  local diff_stat
  diff_stat="$(git -C "$REPO_ROOT" diff --cached --stat || true)"
  echo "----- staged diff (no commit) -----"
  echo "${diff_stat:-(no changes staged)}"
  echo "-----------------------------------"

  # ---- VALIDATE_CMD (judge gets this output) ----
  local validate_out=""
  if [[ -n "$VALIDATE_CMD" ]]; then
    validate_out="$(bash -c "$VALIDATE_CMD" 2>&1 || true)"
    # Cap to keep prompt size sane.
    validate_out="$(printf '%s' "$validate_out" | head -c 16000)"
  else
    validate_out="(no VALIDATE_CMD configured — judge will rely on file reads)"
  fi

  # ---- JUDGE ----
  local judge_prompt judge_envelope judge_text parsed
  judge_prompt="$(cat <<EOF
You are the JUDGE in a scenario-first factory. You score observable behavior
against the holdout scenarios. You may read holdout/, but you must NOT edit
any files, and you must NOT commit anything.

Task (factory/tasks/${TASK_NAME}.md):
${TASK_TEXT}

Holdout scenarios (judge-only):
${HOLD_SCEN}

Captured output of VALIDATE_CMD = "${VALIDATE_CMD}":
\`\`\`
${validate_out}
\`\`\`

You may also use Read/Bash to inspect the working tree as needed.

For each holdout scenario, judge how well the observed behavior satisfies the
scenario's "satisfied when". Then output ONLY a single JSON object, no prose,
no markdown fences, with this exact shape:

{"min_satisfaction": <number 0.0..1.0>,
 "failing": [<filenames of scenarios that did not fully pass>],
 "notes": "<one paragraph explaining the score>"}

\`min_satisfaction\` is the MINIMUM across all holdout scenarios (not the
average) — if one fully fails, this is near 0.
EOF
)"

  judge_envelope="$BUNDLE_DIR/judge.json"
  FACTORY_PHASE=judge claude -p "$judge_prompt" \
      --output-format json \
      --permission-mode acceptEdits \
      --allowedTools Read,Bash \
      > "$judge_envelope" 2>/dev/null || true

  halt="$(extract_halt "$judge_envelope")"
  if [[ "$halt" == "rate_limit" || "$halt" == "billing_error" ]]; then
    log_append "- iter ${i} PAUSED: judge halted with reason \"$halt\". State saved."
    state_write "$i" "$(last_model "$judge_envelope")" "$halt" "$LAST_SCORE_PREV" "$TASK_NAME"
    echo "loop: PAUSED ($halt). State saved at $STATE." >&2
    exit 0
  fi

  judge_text="$(extract_result "$judge_envelope")"
  parsed="$(printf '%s' "$judge_text" | parse_judge_result)"
  if [[ -z "$parsed" ]]; then
    # Retry once with a stricter reminder.
    local retry_prompt
    retry_prompt="$(printf '%s\n\nYour previous reply was not parseable JSON. Reply with ONLY the JSON object and nothing else.' "$judge_prompt")"
    FACTORY_PHASE=judge claude -p "$retry_prompt" \
        --output-format json \
        --permission-mode acceptEdits \
        --allowedTools Read,Bash \
        > "$judge_envelope" 2>/dev/null || true
    judge_text="$(extract_result "$judge_envelope")"
    parsed="$(printf '%s' "$judge_text" | parse_judge_result)"
  fi
  if [[ -z "$parsed" ]]; then
    log_append "- iter ${i}: judge output did not parse twice. Recording as failed iteration."
    state_write "$i" "$(last_model "$judge_envelope")" "judge_parse_error" "$LAST_SCORE_PREV" "$TASK_NAME"
    echo "loop: judge output unparseable after retry; state saved" >&2
    return 0
  fi

  local score failing notes
  score="$(  python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["min_satisfaction"])' <<<"$parsed")"
  failing="$(python3 -c 'import json,sys; o=json.loads(sys.stdin.read()); print(", ".join(o.get("failing") or []))' <<<"$parsed")"
  notes="$(  python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("notes",""))' <<<"$parsed")"

  echo "judge: min_satisfaction=$score   failing=[$failing]"
  LAST_SCORE_PREV="$score"

  # Decide.
  local at_or_above
  at_or_above="$(python3 -c "print(1 if float('$score') >= float('$THRESHOLD') else 0)")"
  if [[ "$at_or_above" == "1" ]]; then
    {
      echo "# Summary: $TASK_NAME"
      echo
      echo "- Converged at iter $i with min_satisfaction = $score"
      echo "- Threshold = $THRESHOLD"
      echo "- Judge notes: $notes"
      echo
      echo "## Staged diff"
      echo '```'
      git diff --cached --stat
      echo '```'
      echo
      echo "The loop did NOT commit. Human reviews and commits manually."
    } > "$SUMMARY_DIR/${TASK_NAME}.md"
    state_write "$i" "$(last_model "$judge_envelope")" "approved_stop" "$score" "$TASK_NAME"
    log_append "- iter ${i} CONVERGED: score=$score >= $THRESHOLD. Summary at $SUMMARY_DIR/${TASK_NAME}.md. Awaiting human approval to commit."
    echo "loop: CONVERGED. See $SUMMARY_DIR/${TASK_NAME}.md. No commit happened."
    exit 0
  fi

  # Below threshold — append feedback for next iteration.
  log_append "$(cat <<EOF
- iter ${i} feedback (judge): score=$score
  failing: [$failing]
  notes: $notes
EOF
)"
  state_write "$i" "$(last_model "$judge_envelope")" "" "$score" "$TASK_NAME"
}

# ---------- driver ----------

START_ITER=$((ITER + 1))
END_ITER="$MAX_ITERS"
if [[ "$ONE_ITER" == "1" ]]; then
  END_ITER="$START_ITER"
fi

for ((i=START_ITER; i<=END_ITER; i++)); do
  run_iteration "$i"
done

# If we got here without exiting, we either ran --one-iter or hit MAX_ITERS.
if [[ "$ONE_ITER" == "1" ]]; then
  echo "loop: --one-iter complete. State: $STATE"
  exit 0
fi

# Reached MAX_ITERS without converging.
state_write "$MAX_ITERS" "$(state_get last_model)" "max_iters" "$LAST_SCORE_PREV" "$TASK_NAME"
log_append "- run END: hit MAX_ITERS=$MAX_ITERS without converging. last_score=$LAST_SCORE_PREV. Human review needed."
echo "loop: hit MAX_ITERS without converging. No commit happened."
exit 0
