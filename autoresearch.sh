#!/bin/bash
set -euo pipefail

cd /home/hashb/workspace/ruckig

# Rebuild the C++ extension
source .venv/bin/activate
pip install -e . 2>&1 | tail -1

# Run the comparison benchmark
OUTPUT=$(python examples/18_waypoints_backend_compare.py 2>&1)

# Extract metrics from the output
LOCAL_DUR=$(echo "$OUTPUT" | grep "^local  duration" | sed 's/.*= //' | sed 's/ s//')
CLOUD_DUR=$(echo "$OUTPUT" | grep "^cloud  duration" | sed 's/.*= //' | sed 's/ s//')

# Deviation lines look like: "  local   max=0.265393  mean=0.096482  ..."
LOCAL_MAX_DEV=$(echo "$OUTPUT" | grep '^  local ' | head -1 | grep -oP 'max=\K[0-9.]+')
CLOUD_MAX_DEV=$(echo "$OUTPUT" | grep '^  cloud ' | head -1 | grep -oP 'max=\K[0-9.]+')

LOCAL_MEAN_DEV=$(echo "$OUTPUT" | grep '^  local ' | head -1 | grep -oP 'mean=\K[0-9.]+')
CLOUD_MEAN_DEV=$(echo "$OUTPUT" | grep '^  cloud ' | head -1 | grep -oP 'mean=\K[0-9.]+')

# Check kinematic pass/fail
KINEMATIC_PASS="true"
if echo "$OUTPUT" | grep -q "FAIL"; then
    KINEMATIC_PASS="false"
fi

# Calculate duration gap percentage
if [ -n "$CLOUD_DUR" ] && [ -n "$LOCAL_DUR" ]; then
    DURATION_GAP=$(python3 -c "print(f'{(($LOCAL_DUR - $CLOUD_DUR) / $CLOUD_DUR * 100):.2f}')")
else
    DURATION_GAP="999"
fi

echo "METRIC duration_gap_pct=$DURATION_GAP"
echo "METRIC local_duration=$LOCAL_DUR"
echo "METRIC cloud_duration=$CLOUD_DUR"
echo "METRIC local_max_dev=$LOCAL_MAX_DEV"
echo "METRIC cloud_max_dev=$CLOUD_MAX_DEV"
echo "METRIC local_mean_dev=$LOCAL_MEAN_DEV"
echo "METRIC cloud_mean_dev=$CLOUD_MEAN_DEV"
echo "METRIC kinematic_pass=$KINEMATIC_PASS"
