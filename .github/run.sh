#!/usr/bin/env bash
# Run a CI command; on failure, surface errors, failed tests and the tail of the output
# as one annotation, so the cause is visible on the run page without downloading logs.
set -o pipefail
out=$(mktemp)
"$@" 2>&1 | tee "$out"
status=$?
if [ $status -ne 0 ]; then
  { grep -E 'FAILED|Error|error:|assert |Exception|would reformat' "$out" | head -n 40; echo '--- tail ---'; tail -n 25 "$out"; } \
    | sed -e 's/%/%25/g' | awk '{printf "%s%%0A", $0}' > "$out.msg"
  echo "::error title=$1 failed (exit $status)::$(cat "$out.msg")"
fi
exit $status
