We recently fixed issues with path deviation and that worked a bit but not totally there yet. 
I want you to focus on closing the gap between total moving time from local calculator 
compared to the output of cloud api while keeping the kinematic limits of the trajectory 
valid. I also want you to bring the deiviation to be similar to or lower than what the cloud
api has, according to the paper, this algorithm can achieve that.

local  duration = 9.8108 s
[ruckig] calculate trajectory via cloud API.
cloud  duration = 7.8705 s

LocalWaypointsCalculator class implements the trajectory generation algorithm described in 
doc/paper-tex latex source. I believe the implementation is incorrect. Verify whats broken 
and implement fixes to bring the local implementation to be in parity with the paper and 
verify by running test example 18

use the .venv/bin/python

to build and run test use the following command

```
source .venv/bin/activate && pip install -e . && python examples/18_waypoints_backend_compare.py
```
