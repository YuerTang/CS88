import numpy as np
import robosuite as suite

from robosuite.utils.placement_samplers import UniformRandomSampler

import os
import cv2


# Import your custom modules
from policies import LiftPolicy
from camera_utils import (
    run_hand_eye_calibration, 
    merge_point_clouds, 
    get_intrinsics,
    get_real_depth_map,
    detect_red_blob,
    pixel_to_camera3d,
    VIEWPOINTS, 
    get_calibration_offset,
    visualize_point_cloud
)

"""
Main Test Script with Calibration Check
"""

# --------------------------------------------------
# 1. Check & Load Calibration
# --------------------------------------------------
transforms = {}
intrinsics = None
need_calibration = False

# Check if a file exists for every requested viewpoint
for cam in VIEWPOINTS:
    filename = f"T_{cam}.npy"
    if not os.path.exists(filename):
        print(f"Missing calibration file: {filename}")
        need_calibration = True
        break

if need_calibration:
    print(">>> Calibration files missing. Running FULL calibration sequence...")
    # This runs the robot, collects data, and saves the .npy files
    transforms, intrinsics = run_hand_eye_calibration()
else:
    print(">>> Found existing calibration files. Loading...")
    for cam in VIEWPOINTS:
        filename = f"T_{cam}.npy"
        transforms[cam] = np.load(filename)
        print(f"    Loaded {filename}")

# Verify loaded transforms
print("\n=== LOADED TRANSFORMS ===")
for cam_name in VIEWPOINTS:
    if cam_name in transforms:
        T = transforms[cam_name]
        print(f"\n{cam_name}:")
        print(f"  Translation: {T[:3, 3]}")
        print(f"  Rotation determinant: {np.linalg.det(T[:3, :3]):.3f}")
        
        # Check if it's reasonable
        distance = np.linalg.norm(T[:3, 3])
        print(f"  Distance from origin: {distance:.3f}m")
        
        if distance < 0.1 or distance > 5.0:
            print(f"  ⚠️ WARNING: Suspicious camera position!")

offset = get_calibration_offset()

# --------------------------------------------------
# 2. Simulation Setup
# --------------------------------------------------
print("\n>>> Setting up Simulation Environment...")

placement_initializer = UniformRandomSampler(
        name="ObjectSampler",
        mujoco_objects=None,  # Will be set by environment
        x_range=[-0.05, 0.18],   #  X range
        y_range=[-0.1, 0.1],   #  Y range
        rotation=0.0,         # Fixed rotation (radians)
        rotation_axis='z',
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array([0, 0, 0.8]),  # Table height
        z_offset=0.01,
    )


env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=VIEWPOINTS,
    placement_initializer=placement_initializer,
    camera_depths=True
)

# --------------------------------------------------
# 3. Calculate Intrinsics (If we loaded from file)
# --------------------------------------------------
if intrinsics is None:
    print(">>> Calculating intrinsics from environment...")
    img_height, img_width = 256, 256
    intrinsics = get_intrinsics(env, VIEWPOINTS, img_height, img_width)

# --------------------------------------------------
# 4. Main Execution Loop - Using all Cameras
# --------------------------------------------------
num_trials = 10
success_rate = 0

for trial in range(num_trials):
    obs = env.reset()
    print(f"\n--- Trial {trial + 1} ---")

    # --------------------------------------------------
    # Try to detect red object from all cameras
    # --------------------------------------------------
    target_pos = None
    
    for cam_name in VIEWPOINTS:
        # Flip RGB image for display
        obs_rgb = cv2.flip(obs[cam_name+'_image'], 0)

        # Convert and flip depth image
        obs_depth_buffer = obs[cam_name+'_depth']
        obs_depth = get_real_depth_map(env.sim, obs_depth_buffer, cam_name)
        obs_depth = cv2.flip(obs_depth, 0)

        # Detect the red object in the image
        red_target = detect_red_blob(obs_rgb)
        print(f"{cam_name} red target: {red_target}")

        # Visualize detected red blob
        if red_target:
            cv2.circle(obs_rgb, red_target, 5, (0, 255, 0), -1)

        cv2.imshow(f'{cam_name} view', obs_rgb)
        cv2.waitKey(1)

        # Skip this camera if no red blob was found
        if red_target is None:
            print(f'  {cam_name}: No target detected')
            continue

        # --------------------------------------------------
        # Convert detected red pixel to robot frame coordinate
        # --------------------------------------------------
        cx, cy = red_target
        target_3d_cam = pixel_to_camera3d(cx, cy, obs_depth, intrinsics[cam_name])
        
        if target_3d_cam is None:
            print(f'  {cam_name}: Invalid depth at target pixel')
            continue
            
        # Transform to robot base frame
        target_3d_homo = np.array([target_3d_cam[0], target_3d_cam[1], target_3d_cam[2], 1.0])
        T = transforms[cam_name]
        target_in_base_homo = T @ target_3d_homo
        target_in_base = target_in_base_homo[:3]
        
        print(f"  {cam_name} detected target at: {target_in_base}")
        
        # Use the first successful detection
        if target_pos is None:
            target_pos = target_in_base
            print(f"  Using {cam_name} detection")
            break

    # If no camera detected the target, skip this trial
    if target_pos is None:
        print('ERROR: No target detected from any camera!')
        continue

    # Visualize point cloud for first trial [press 'q' to continue]
    if trial == 0:
        cloud = merge_point_clouds(obs, intrinsics, transforms, env.sim)
        if cloud is not None and len(cloud) > 0:
            print("Visualizing first trial point cloud...")
            visualize_point_cloud(cloud, target_pos)

    # Execute Policy
    goal_position = target_pos + offset
    print(f"Goal position: {goal_position}")
    policy = LiftPolicy(goal_position)

    step_count = 0
    while True:
        action = policy.get_action_proprio(obs['robot0_eef_pos'])
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        # Visualize agentview
        obs_rgb = cv2.cvtColor(obs['agentview_image'], cv2.COLOR_BGR2RGB)
        obs_rgb = cv2.flip(obs_rgb, 0) 
        cv2.imshow('Test Run', obs_rgb)
        cv2.waitKey(1)

        if reward == 1.0:
            print("Success!")
            success_rate += 1
            break
        if done or step_count > 200:
            print("Failed or Timed out.")
            break

cv2.destroyAllWindows()
print(f"\nFinal Success Rate: {success_rate / num_trials}")