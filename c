#!/usr/bin/env bash
# Human-only commit.  ./c <TASK> "<area>: <what> - <checkpoint> green" <path> [path ...]
set -euo pipefail
[ $# -ge 3 ] || { echo 'usage: ./c A2 "env: tax table + midpoint - KAT-1 green" env tests/env' >&2; exit 2; }
task="$1"; msg="$2"; shift 2
cd "$(dirname "$0")"
git add -A -- "$@"
if git diff --cached --quiet; then echo "c: nothing to commit under: $*" >&2; exit 1; fi
git diff --cached --stat | tail -n 15
HUMAN_COMMIT=1 git commit -q -m "$task $msg"
git log --oneline -1
