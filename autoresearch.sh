#!/bin/bash
set -euo pipefail

cd /home/hashb/workspace/ruckig

# Rebuild the C++ extension
source .venv/bin/activate
pip install -e . 2>&1 | tail -1

# Run the comparison benchmark
OUTPUT=$(python examples/18_waypoints_backend_compare.py 2>&1)

# Extract metrics from the output
LOCAL_DUR=$(echo "$OUTPUT" | grep "^local  duration" | awk '{print $4}')
CLOUD_DUR=$(echo "$OUTPUT" | grep "^cloud  duration" | awk '{print $4}')

LOCAL_MAX_DEV=$(echo "$OUTPUT" | grep "^\s*local " | head -1 | awk '{print $2}' | sed 's/max=//')
CLOUD_MAX_DEV=$(echo "$OUTPUT" | grep "^\s*cloud " | head -1 | awk '{print $2}' | sed 's/max=//')

LOCAL_MEAN_DEV=$(echo "$OUTPUT" | grep "^\s*local " | head -1 | awk '{print $3}' | sed 's/mean=//')
CLOUD_MEAN_DEV=$(echo "$OUTPUT" | grep "^\s*cloud " | head -1 | awk '{print $3}' | sed 's/mean=//')

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
