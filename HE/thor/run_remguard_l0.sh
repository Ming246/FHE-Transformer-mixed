#!/usr/bin/env bash
# Durable L0 plan_mock + rem_guard (double-fork; survives Cursor teardown).
# Usage: bash HE/thor/run_remguard_l0.sh [tag]
set -u
cd /workspace
TAG="${1:-v13}"
export REMGUARD_TAG="$TAG"
export REMGUARD_LOG="/tmp/thor_logs/l0_remguard_${TAG}.log"
export REMGUARD_PIDF="/tmp/thor_logs/l0_remguard_${TAG}.pid"
mkdir -p /tmp/thor_logs

if [[ -f "$REMGUARD_PIDF" ]] && kill -0 "$(cat "$REMGUARD_PIDF")" 2>/dev/null; then
  echo "already running pid=$(cat "$REMGUARD_PIDF") log=$REMGUARD_LOG"
  exit 0
fi

python3 -c '
import os, sys, time
log, pidf = os.environ["REMGUARD_LOG"], os.environ["REMGUARD_PIDF"]
if os.fork() > 0:
    time.sleep(0.5)
    print("started pid=" + open(pidf).read().strip() + " log=" + log)
    raise SystemExit(0)
os.setsid()
if os.fork() > 0:
    raise SystemExit(0)
os.chdir("/workspace")
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.close(fd)
try:
    os.close(0)
except OSError:
    pass
open(pidf, "w").write(str(os.getpid()))
os.execvp("python3", ["python3", "-u", "HE/thor/smoke_thor_repro.py",
    "--layer", "0", "--bootstrap", "plan_mock", "--probe-levels", "--rest-div", "1"])
'
