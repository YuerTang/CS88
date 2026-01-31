import numpy as np
from scipy.spatial.transform import Rotation as R
import robosuite as suite
from CA.CA1_PID_starter.policies import NutAssemblyPolicy

env = suite.make(env_name='NutAssembly', robots='Panda', has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, horizon=1500)

# Run multiple episodes to check success rate
num_episodes = 10
successes = 0
partial = 0

for episode in range(num_episodes):
    obs = env.reset()
    policy = NutAssemblyPolicy(obs)

    for step in range(2000):
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)

        if reward == 1.0:
            successes += 1
            print(f'Episode {episode+1}: SUCCESS at step {step}')
            break
        elif done:
            if reward >= 0.5:
                partial += 1
            print(f'Episode {episode+1}: DONE at step {step}, phase {policy.phase}, reward={reward:.1f}')
            break
    else:
        if reward >= 0.5:
            partial += 1
        print(f'Episode {episode+1}: TIMEOUT at step 1500, phase {policy.phase}, reward={reward:.1f}')

print(f'\nResults: {successes}/{num_episodes} full success ({100*successes/num_episodes:.0f}%)')
print(f'Partial success (≥0.5 reward): {partial+successes}/{num_episodes}')
env.close()
