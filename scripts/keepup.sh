#!/usr/bin/env bash
# Keep botc-ai services alive. Vite/API/vision die when started as agent
# background jobs; this daemon is independent of any chat session.
#
#   ./scripts/keepup.sh start    # start watchdog (idempotent)
#   ./scripts/keepup.sh stop     # stop everything
#   ./scripts/keepup.sh status   # port + pid check
#   ./scripts/keepup.sh once     # one ensure pass (no daemon)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/.run"
LOG="$RUN/logs"
WATCH_PID="$RUN/watchdog.pid"
INTERVAL="${KEEPUP_INTERVAL:-5}"

mkdir -p "$RUN" "$LOG"

# Prefer project venv / nvm node if present
export PATH="${ROOT}/vision/.venv/bin:${HOME}/.nvm/versions/node/v22.22.0/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

services=(
  # name|port|health_url|start_cmd  (env VAR=val prefix is supported)
  "vision|8766|http://127.0.0.1:8766/api/state|env VISION_FACE_BACKEND=opencv \"${ROOT}/vision/.venv/bin/python\" \"${ROOT}/vision/server.py\""
  "api|3001|http://127.0.0.1:3001/api/state|node \"${ROOT}/server/index.js\""
  "ui|5181|http://127.0.0.1:5181/|npx --yes vite --port 5181 --host 127.0.0.1 --strictPort"
)

alive() {
  local url="$1"
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 1 --max-time 2 "$url" 2>/dev/null || echo 0)
  [[ "$code" =~ ^[23][0-9][0-9]$ ]]
}

port_pids() {
  lsof -ti ":$1" 2>/dev/null || true
}

start_svc() {
  local name="$1" port="$2" cmd="$3"
  local pidfile="$RUN/${name}.pid"
  local logfile="$LOG/${name}.log"

  # Already healthy?
  if alive "$4"; then
    return 0
  fi

  # Clear stale listeners on the port
  local p
  for p in $(port_pids "$port"); do
    kill "$p" 2>/dev/null || true
  done
  sleep 0.3

  echo "[keepup $(date '+%H:%M:%S')] starting $name on :$port" | tee -a "$LOG/watchdog.log"
  # Detach fully: new session, stdin from /dev/null, no job control
  # shellcheck disable=SC2086
  nohup bash -c "cd \"$ROOT\" && exec $cmd" </dev/null >>"$logfile" 2>&1 &
  local pid=$!
  echo "$pid" >"$pidfile"
  disown "$pid" 2>/dev/null || true

  # Wait briefly for health
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 0.5
    if alive "$4"; then
      echo "[keepup $(date '+%H:%M:%S')] $name up (pid $pid)" >>"$LOG/watchdog.log"
      return 0
    fi
  done
  echo "[keepup $(date '+%H:%M:%S')] WARN $name not healthy yet — see $logfile" | tee -a "$LOG/watchdog.log"
  return 1
}

ensure_all() {
  local entry name port url cmd
  for entry in "${services[@]}"; do
    IFS='|' read -r name port url cmd <<<"$entry"
    if alive "$url"; then
      continue
    fi
    start_svc "$name" "$port" "$cmd" "$url" || true
  done
}

status() {
  local entry name port url cmd code
  printf "%-8s %-6s %-6s %s\n" "NAME" "PORT" "HTTP" "PIDS"
  for entry in "${services[@]}"; do
    IFS='|' read -r name port url cmd <<<"$entry"
    code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 1 --max-time 2 "$url" 2>/dev/null || echo down)
    printf "%-8s %-6s %-6s %s\n" "$name" "$port" "$code" "$(port_pids "$port" | tr '\n' ' ')"
  done
  if [[ -f "$WATCH_PID" ]] && kill -0 "$(cat "$WATCH_PID")" 2>/dev/null; then
    echo "watchdog: running (pid $(cat "$WATCH_PID"), every ${INTERVAL}s)"
  else
    echo "watchdog: not running"
  fi
}

stop_all() {
  # Stop watchdog first
  if [[ -f "$WATCH_PID" ]]; then
    local wp
    wp=$(cat "$WATCH_PID" 2>/dev/null || true)
    if [[ -n "$wp" ]]; then
      kill "$wp" 2>/dev/null || true
    fi
    rm -f "$WATCH_PID"
  fi
  local entry name port url cmd p
  for entry in "${services[@]}"; do
    IFS='|' read -r name port url cmd <<<"$entry"
    for p in $(port_pids "$port"); do
      echo "[keepup] killing $name pid $p"
      kill "$p" 2>/dev/null || true
    done
    rm -f "$RUN/${name}.pid"
  done
  sleep 0.5
  # force
  for entry in "${services[@]}"; do
    IFS='|' read -r name port url cmd <<<"$entry"
    for p in $(port_pids "$port"); do
      kill -9 "$p" 2>/dev/null || true
    done
  done
  echo "[keepup] stopped"
}

watch_loop() {
  echo "[keepup $(date '+%H:%M:%S')] watchdog started (interval=${INTERVAL}s)" >>"$LOG/watchdog.log"
  while true; do
    ensure_all
    sleep "$INTERVAL"
  done
}

start_watchdog() {
  if [[ -f "$WATCH_PID" ]] && kill -0 "$(cat "$WATCH_PID")" 2>/dev/null; then
    echo "[keepup] watchdog already running (pid $(cat "$WATCH_PID"))"
    ensure_all
    status
    return 0
  fi
  ensure_all
  nohup bash -c "cd \"$ROOT\" && exec \"$ROOT/scripts/keepup.sh\" _watch" \
    </dev/null >>"$LOG/watchdog.log" 2>&1 &
  echo $! >"$WATCH_PID"
  disown 2>/dev/null || true
  sleep 1
  echo "[keepup] watchdog pid $(cat "$WATCH_PID")"
  status
}

cmd="${1:-start}"
case "$cmd" in
  start) start_watchdog ;;
  stop) stop_all ;;
  status) status ;;
  once) ensure_all; status ;;
  _watch) watch_loop ;;
  *)
    echo "usage: $0 {start|stop|status|once}"
    exit 1
    ;;
esac
