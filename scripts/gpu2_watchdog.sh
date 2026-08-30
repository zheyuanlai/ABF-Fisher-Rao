#!/usr/bin/env bash
# GPU-2 co-tenancy watchdog (uniform-FR campaign, olefin/CHA stage).
# Policy: GPU 2 may be used while its other tenants are user 'yesom' or nobody.
# If ANY other user's compute process appears on GPU 2, kill OUR OWN GPU-2 jobs
# (by pid, matched to our scripts -- never a device-wide kill) and exit.
set -u
UUID2=$(nvidia-smi -L | sed -n 's/GPU 2: .*UUID: \(GPU-[0-9a-f-]*\)).*/\1/p')
while true; do
  foreign=0
  while read -r uuid pid; do
    pid=${pid// /}; uuid=${uuid%,}
    [ "$uuid" = "$UUID2" ] || continue
    user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -z "$user" ] && continue
    if [ "$user" != "yesom" ] && [ "$user" != "$(whoami)" ]; then
      foreign=1
      echo "$(date -u +%H:%M:%S) foreign user '$user' (pid $pid) on GPU 2"
    fi
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
  if [ "$foreign" = "1" ]; then
    echo "$(date -u +%H:%M:%S) stopping OUR GPU-2 jobs (pid-targeted)"
    for pid in $(pgrep -u "$(whoami)" -f "run_cha_reference.py"); do
      env=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=')
      if [ "$env" = "CUDA_VISIBLE_DEVICES=2" ]; then
        echo "  kill $pid ($env)"; kill -TERM "$pid"
      fi
    done
    exit 0
  fi
  sleep 60
done
