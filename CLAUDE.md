# Robotics Autoresearch

Read `program.md` for full instructions. This is an autoresearch loop where you iteratively improve `mpc.py` to maximize forward speed of a Unitree H1 humanoid robot in MuJoCo.

## Quick Reference

- **Edit only**: `mpc.py`
- **Do not edit**: `prepare.py`
- **Run**: `uv run mpc.py > run.log 2>&1`
- **Check**: `cat run.log | grep -E "avg_speed_mps|Error|Traceback"`
- **Log results**: append to `results.tsv`
- **Metric**: `avg_speed_mps` (higher is better)
- **Visualize**: `uv run mpc.py --viz` (opens MuJoCo viewer, not for automated runs)
