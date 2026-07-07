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

# Check kinematic pass/fail AND end state accuracy
KINEMATIC_PASS="true"
if echo "$OUTPUT" | grep -q "FAIL"; then
    KINEMATIC_PASS="false"
fi

# Also verify end state accuracy
END_STATE_OK=$(python3 -c "
import sys
sys.path.insert(0, '/home/hashb/workspace/ruckig/.venv/lib/python3.11/site-packages')
try:
    from ruckig import InputParameter, OutputParameter, Result, Ruckig, WaypointsBackend
    import numpy as np
    inp = InputParameter(3)
    inp.current_position = [0.2, 0, -0.3]
    inp.current_velocity = [0, 0.2, 0]
    inp.current_acceleration = [0, 0.6, 0]
    inp.intermediate_positions = [[1.4,-1.6,1.0],[-0.6,-0.5,0.4],[-0.4,-0.35,0.0],[0.8,1.8,-0.1]]
    inp.target_position = [0.5, 1, 0]
    inp.target_velocity = [0.2, 0, 0.3]
    inp.target_acceleration = [0, 0.1, -0.1]
    inp.max_velocity = [1, 2, 1]; inp.max_acceleration = [3, 2, 2]; inp.max_jerk = [6, 10, 20]
    otg = Ruckig(3, 0.01, 10)
    otg.set_waypoints_backend(WaypointsBackend.Local)
    out = OutputParameter(3, 10)
    while True:
        res = otg.update(inp, out)
        out.pass_to_input(inp)
        if res != 1: break
    traj = out.trajectory
    p, v, a = traj.at_time(traj.duration)
    p_err = max(abs(p[i] - [0.5,1,0][i]) for i in range(3))
    v_err = max(abs(v[i] - [0.2,0,0.3][i]) for i in range(3))
    ok = p_err < 0.05 and v_err < 0.1
    print('true' if ok else 'false')
except Exception as e:
    print('false')
" 2>/dev/null)
if [ "$END_STATE_OK" != "true" ]; then
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
