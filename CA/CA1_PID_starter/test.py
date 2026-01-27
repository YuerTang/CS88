import numpy as np
import robosuite as suite
import time
from scipy.spatial.transform import Rotation as R
from policies2 import *


# create environment instance
env = suite.make(
    env_name="NutAssembly", # replace with other tasks "NutAssembly" and "Door"
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# RENDER_DELAY = 0.05  # Slow down rendering (50ms per frame)

# reset the environment
for episode in range(10):
    obs = env.reset()
    policy = NutAssemblyPolicy(obs)
    step = 0
    prev_phase = -1

    while True:
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)  # take action in the environment

        env.render()  # render on display
        # time.sleep(RENDER_DELAY)  # Slow down for observation

        # Debug: print EEF position, orientation, and action
        eef_pos = obs['robot0_eef_pos']
        eef_quat = obs['robot0_eef_quat']
        eef_euler = R.from_quat(eef_quat).as_euler('xyz', degrees=True)

        phase_changed = (policy.phase != prev_phase)
        if phase_changed or step % 30 == 0:
            print(f"Step {step:4d} | Phase {policy.phase:2d} | "
                  f"EEF:[{eef_pos[0]:6.3f},{eef_pos[1]:6.3f},{eef_pos[2]:6.3f}] | "
                  f"Euler:[{eef_euler[0]:6.1f},{eef_euler[1]:6.1f},{eef_euler[2]:6.1f}] | "
                  f"Ori:[{action[3]:6.3f},{action[4]:6.3f},{action[5]:6.3f}]")
            prev_phase = policy.phase
        step += 1

        if reward == 1.0 or done: break

    print(f"Episode {episode+1} ended at step {step}")