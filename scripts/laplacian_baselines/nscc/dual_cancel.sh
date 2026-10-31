#!/bin/bash
# Dual submission helper: given two job ids of the SAME idempotent task (e.g. gdev + g1 copies), poll until one is
# running and qdel the other while it is still queued.  Usage: dual_cancel.sh <jobA> <jobB> [poll_s]
A=$1; B=$2; P=${3:-60}
state() { qstat -f "$1" 2>/dev/null | awk '/job_state/ {print $3}'; }
while true; do
  sa=$(state "$A"); sb=$(state "$B")
  if [ -z "$sa" ] && [ -z "$sb" ]; then echo "both gone"; exit 0; fi
  if [ "$sa" = R ] && [ "$sb" = Q ]; then qdel "$B" && echo "$(date -u +%FT%TZ) $A running -> cancelled $B"; exit 0; fi
  if [ "$sb" = R ] && [ "$sa" = Q ]; then qdel "$A" && echo "$(date -u +%FT%TZ) $B running -> cancelled $A"; exit 0; fi
  if [ "$sa" = R ] && [ -z "$sb" ]; then echo "$A running, $B gone"; exit 0; fi
  if [ "$sb" = R ] && [ -z "$sa" ]; then echo "$B running, $A gone"; exit 0; fi
  if [ "$sa" = R ] && [ "$sb" = R ]; then echo "BOTH running ($A, $B): nothing cancelled (idempotent tasks)"; exit 0; fi
  sleep "$P"
done
