"""Minimal smoke test: run a few random-action steps against the FMU-backed
liquid-cooling environment used throughout this repo (see frontier_env.py)."""
from frontier_env import SmallFrontierModel

# Create and reset the environment (loads and initializes the FMU)
env = SmallFrontierModel()
obs = env.reset()
print(f"Initial observation keys: {list(obs.keys())}")

# Simple 5-step control loop
for step in range(5):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(
        f"Step {step}: "
        f"cdu-cabinet-1 reward={reward['cdu-cabinet-1']:.3f}, "
        f"cooling-tower-1 reward={reward['cooling-tower-1']:.3f}, "
        f"cabinet-1 boundary temps (K)={info['cdu-cabinet-1'][:3]}"
    )
    if all(done.values()):
        break

env.close()
