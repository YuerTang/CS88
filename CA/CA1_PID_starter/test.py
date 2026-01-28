import numpy as np
import robosuite as suite
import time
from scipy.spatial.transform import Rotation as R
from policies2 import NutAssemblyPolicy


# create environment instance
env = suite.make(
    env_name="NutAssembly",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# Track success rate
num_episodes = 10
successes = 0

for episode in range(num_episodes):
    obs = env.reset()
    policy = NutAssemblyPolicy(obs)
    step = 0
    prev_phase = -1

    while True:
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)

        env.render()

        # Debug: print phase changes
        eef_pos = obs['robot0_eef_pos']
        eef_quat = obs['robot0_eef_quat']
        eef_euler = R.from_quat(eef_quat).as_euler('xyz', degrees=True)

        phase_changed = (policy.phase != prev_phase)
        if phase_changed or step % 50 == 0:
            print(f"Step {step:4d} | Phase {policy.phase:2d} | "
                  f"EEF:[{eef_pos[0]:6.3f},{eef_pos[1]:6.3f},{eef_pos[2]:6.3f}] | "
                  f"Yaw:{eef_euler[2]:6.1f} deg")
            prev_phase = policy.phase
        step += 1

        if reward == 1.0:
            successes += 1
            print(f"Episode {episode+1}: SUCCESS at step {step}")
            break
        elif done:
            print(f"Episode {episode+1}: FAILED at step {step}, phase {policy.phase}")
            break

print(f"\n{'='*50}")
print(f"Results: {successes}/{num_episodes} success ({100*successes/num_episodes:.0f}%)")
print(f"{'='*50}")

env.close()