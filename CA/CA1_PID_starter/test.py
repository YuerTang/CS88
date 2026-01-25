import numpy as np
import robosuite as suite
import time
from policies import *


# create environment instance
env = suite.make(
    env_name="Door", # replace with other tasks "NutAssembly" and "Door"
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# RENDER_DELAY = 0.05  # Slow down rendering (50ms per frame)

# reset the environment
for episode in range(5):
    obs = env.reset()
    policy = DoorPolicy(obs)
    step = 0

    while True:
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)  # take action in the environment

        env.render()  # render on display
        # time.sleep(RENDER_DELAY)  # Slow down for observation

        # Debug: print phase, handle rotation, and door opening
        handle_qpos = obs.get('handle_qpos', 0)
        if hasattr(handle_qpos, '__len__'):
            handle_qpos = handle_qpos[0]
        hinge_qpos = obs.get('hinge_qpos', 0)
        if hasattr(hinge_qpos, '__len__'):
            hinge_qpos = hinge_qpos[0]

        if step % 30 == 0:
            print(f"Step {step}: Phase {policy.phase}, handle={handle_qpos:.3f}, hinge={hinge_qpos:.3f} (need >0.3)")
        step += 1

        if reward == 1.0 or done: break

    print(f"Episode {episode+1} ended at step {step}")