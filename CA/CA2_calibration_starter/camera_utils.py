import numpy as np
import robosuite as suite
import cv2
import robosuite.utils.camera_utils as cu
from scipy.spatial.transform import Rotation as R
from robosuite.utils.placement_samplers import UniformRandomSampler

# Import policies for robot control
try:
    from policies import LiftPolicy, MoveToPolicy
except ImportError:
    print("Warning: 'policies' module not found.")

# Camera viewpoints used for calibration
VIEWPOINTS = ["frontview", "agentview", "sideview"]

"""
CS 188: Camera Calibration Assignment
Complete the functions marked with TODO to implement hand-eye calibration.
"""

# ==============================================================================
# TASK 1: Detect Red Cube in Image 
# ==============================================================================

def detect_red_blob(input_image):
    """
    Detects the largest red-colored blob in an RGB image using HSV thresholding.
    
    Args:
        input_image: RGB image as numpy array (H x W x 3), values in range [0, 255]
    
    Returns:
        (cx, cy): Tuple of integers representing the center pixel coordinates,
                  or None if no red blob is found
    
    """
    if input_image is None:
        return None

    # TODO: Implement red blob detection here
    # Your code here...
            
    return None


# ==============================================================================
# TASK 2: Convert Pixel to 3D Camera Coordinates 
# ==============================================================================

def pixel_to_camera3d(x, y, depth_image, cam_intrinsics):
    """
    Converts a 2D pixel coordinate to a 3D point in the camera coordinate frame.
    
    Args:
        x: Pixel x-coordinate (column index, integer)
        y: Pixel y-coordinate (row index, integer)
        depth_image: Depth map as numpy array (H x W), values in meters
        cam_intrinsics: Dictionary with keys 'fx', 'fy', 'cx', 'cy'
                        fx, fy: focal lengths in pixels
                        cx, cy: principal point coordinates in pixels
    
    Returns:
        np.array([X, Y, Z]): 3D point in camera frame (meters), or None if invalid
    
    Camera Coordinate System:
        - Origin at camera optical center
        - X-axis: right
        - Y-axis: down
        - Z-axis: forward (into the scene)
    
    Pinhole Camera Model:
        x = (X * fx / Z) + cx  -->  X = (x - cx) * Z / fx
        y = (Y * fy / Z) + cy  -->  Y = (y - cy) * Z / fy
        Z = depth value at pixel (x, y)
    
    Hints:
        - Access depth as: depth_image[y, x] (row-major indexing)
        - Handle the case where z is a numpy array: use z.item()
        - Return None for invalid pixels or depth values
    """
    
    # TODO: Implement pixel-to-3D conversion here
    # Your code here...

    return None


# ==============================================================================
# TASK 3: Generate Calibration Waypoints 
# ==============================================================================

def generate_calibration_waypoints(eef_quat, num_x=3, num_y=3, num_z=3, num_rot=2):
    """
    Generates waypoints for hand-eye calibration by sampling positions and orientations.
    
    Args:
        eef_quat: Current end-effector orientation as quaternion [x, y, z, w]
        num_x, num_y, num_z: Number of position samples in each dimension
        num_rot: Number of orientation samples (rotations around Z-axis)
    
    Returns:
        waypoints: List of waypoint dictionaries, each containing:
                   {'pos': [x, y, z], 'ori': [qx, qy, qz, qw]}
    
    Sampling Strategy:
        - Positions: Sample a 3D grid around the cube's position
          - X: [-0.10, 0.10] relative to cube
          - Y: [-0.15, 0.15] relative to cube
          - Z: [cube_z + 0.08, cube_z + 0.15] (above cube)
        - Orientations: Apply delta rotations to current end-effector orientation
          - Rotate around Z-axis: 0°, 90°, 180°, 270° (evenly spaced by num_rot)
        
    Hints:
        - Total waypoints = num_x * num_y * num_z * num_rot
        - Use scipy.spatial.transform.Rotation for rotation math
        - Quaternion format: [x, y, z, w] (scipy convention)
    
    Note: You can modify this function signature as long as it's compatible
          with the run_hand_eye_calibration() function below.
    """
    waypoints = []
    
    # TODO: Implement waypoint generation here
    # Your code here...
    
    return waypoints


# ==============================================================================
# TASK 4: Kabsch Algorithm for Rigid Transformation 
# ==============================================================================

def solve_for_rigid_transformation(inpts, outpts):
    """
    Computes the optimal rigid transformation T such that outpts ≈ T @ inpts.
    Uses the Kabsch algorithm to find rotation and translation.
    
    Args:
        inpts: Source points as numpy array (N x 3) - points in camera frame
        outpts: Target points as numpy array (N x 3) - points in robot frame
    
    Returns:
        T: 4x4 homogeneous transformation matrix such that:
           outpts[i] ≈ (T @ [inpts[i], 1])[0:3] for all i
    
    Implementation Hints:
        - Use np.linalg.svd() for SVD (returns U, S, Vt where Vt = V^T)
        - Use np.linalg.det() to check determinant
        - Initialize T as np.eye(4) and fill in R and t
        - Handle edge case: if len(inpts) != len(outpts), return identity
    
    Expected Calibration Error:
        - Mean error: < 0.03 meters (excellent)
        - Max error:  < 0.10 meters
    """
    
    # TODO: Implement Kabsch algorithm here
    # Your code here...
    
    return None


# ==============================================================================
# TASK 5: Fine-Tune Calibration Offset (10 points)
# ==============================================================================

def get_calibration_offset():
    """
    Returns a 3D offset vector to compensate for systematic calibration errors.
    
    This offset is added to the detected target position before sending to the
    robot controller. Tune this empirically by observing robot behavior.
    
    Returns:
        offset: numpy array [dx, dy, dz] in meters (robot base frame)
    """
    
    # TODO: Tune this offset based on your calibration results
    # Default: no offset
    return np.array([0.0, 0.0, 0.0])


# ==============================================================================
# HELPER FUNCTIONS (Already Implemented - Do Not Modify)
# ==============================================================================

def get_real_depth_map(sim, depth_buffer):
    """
    Converts normalized depth buffer to metric depth using Robosuite's official function.
    
    Args:
        sim: MuJoCo simulation object
        depth_buffer: Normalized depth array from observation (values in [0, 1])
    
    Returns:
        Real depth in meters
    """
    return cu.get_real_depth_map(sim, depth_buffer)


def depth_to_point_cloud_vectorized(depth_image, cam_intrinsics):
    """
    Vectorized conversion of entire depth map to 3D point cloud in camera frame.
    Uses your pixel_to_camera3d logic but operates on all pixels at once.
    """
    if depth_image.ndim == 3:
        depth_image = depth_image.squeeze()
    
    h, w = depth_image.shape
    
    fx, fy = cam_intrinsics['fx'], cam_intrinsics['fy']
    cx, cy = cam_intrinsics['cx'], cam_intrinsics['cy']

    # Generate pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    u = u.flatten()
    v = v.flatten()
    z = depth_image.flatten()
    
    # Filter invalid depths
    valid_mask = (z > 0.1) & (z < 10.0) & np.isfinite(z)
    u, v, z = u[valid_mask], v[valid_mask], z[valid_mask]
    
    if len(z) == 0:
        return np.array([]).reshape(0, 3)
    
    # Back-project to 3D
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z
    
    points = np.stack([x_cam, y_cam, z_cam], axis=1)
    
    # Remove inf/nan
    valid_points = np.isfinite(points).all(axis=1)
    points = points[valid_points]
    
    return points


def merge_point_clouds(obs, cam_intrinsics_dict, transforms_dict, sim):
    """
    Merges point clouds from all calibrated cameras into robot base frame.
    Used by test.py to create a unified 3D scene representation.
    """
    merged_points = []

    for cam_name in transforms_dict.keys():
        depth_key = cam_name + '_depth'
        if depth_key not in obs:
            continue
            
        depth_buffer = obs[depth_key].copy()
        depth_img = get_real_depth_map(sim, depth_buffer)
        depth_img = cv2.flip(depth_img, 0)
        
        if depth_img.ndim == 3: 
            depth_img = depth_img.squeeze()
        
        valid_depths = depth_img[(depth_img > 0.1) & (depth_img < 10.0)]
        if len(valid_depths) < 100:
            continue
        
        points_cam = depth_to_point_cloud_vectorized(depth_img, cam_intrinsics_dict[cam_name])
        
        if len(points_cam) == 0:
            continue
        
        if not np.isfinite(points_cam).all():
            valid_mask = np.isfinite(points_cam).all(axis=1)
            points_cam = points_cam[valid_mask]
            
        if len(points_cam) == 0:
            continue
            
        # Transform to robot frame using calibrated transform
        ones = np.ones((points_cam.shape[0], 1))
        points_homo = np.hstack([points_cam, ones])
        
        T = transforms_dict[cam_name]
        points_robot_homo = points_homo @ T.T
        points_robot = points_robot_homo[:, :3]
        
        if not np.isfinite(points_robot).all():
            continue
        
        merged_points.append(points_robot)
        
    if merged_points:
        cloud = np.vstack(merged_points)
        
        # Remove outliers
        valid_mask = (
            (np.abs(cloud[:, 0]) < 2.0) &
            (np.abs(cloud[:, 1]) < 2.0) &
            (cloud[:, 2] > 0.0) &
            (cloud[:, 2] < 2.0)
        )
        cloud = cloud[valid_mask]
        
        return cloud
        
    return None


def get_intrinsics(env, camera_names, width=256, height=256):
    """
    Extracts camera intrinsic parameters from the simulation.
    Returns dictionary mapping camera names to intrinsic parameters.
    """
    intrinsics = {}
    for name in camera_names:
        K = cu.get_camera_intrinsic_matrix(
            sim=env.sim, 
            camera_name=name, 
            camera_height=height, 
            camera_width=width
        )
        
        intrinsics[name] = {
            "fx": K[0, 0],
            "fy": K[1, 1],
            "cx": K[0, 2],
            "cy": K[1, 2]
        }
    return intrinsics


def visualize_point_cloud(points, target=None):
    """
    Visualizes 3D point cloud using Open3D.
    Useful for debugging calibration results.
    """
    import open3d as o3d
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    pts = np.asarray(pcd.points)
    z_vals = pts[:, 2]
    
    z_min, z_max = 0.75, 1.2
    z_norm = np.clip((z_vals - z_min) / (z_max - z_min), 0, 1)
    
    colors = np.zeros((len(pts), 3))
    colors[:, 0] = z_norm
    colors[:, 2] = 1 - z_norm
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]
    
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )
    geometries.append(base_frame)

    if target is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        sphere.translate(target)
        sphere.paint_uniform_color([0.0, 1.0, 0.0])
        geometries.append(sphere)

    o3d.visualization.draw_geometries(geometries)


# ==============================================================================
# MAIN CALIBRATION ROUTINE 
# ==============================================================================

def run_hand_eye_calibration(verbose=True):
    """
    Main calibration routine that orchestrates the entire process.
    
    Process:
        1. Create simulation environment with fixed cube placement
        2. Calculate camera intrinsics
        3. Pick up the cube using LiftPolicy
        4. Move cube to calibration waypoints
        5. Detect red blob and record correspondences
        6. Solve for camera-to-robot transforms using Kabsch
        7. Verify calibration quality
        8. Save transforms to .npy files
    
    Returns:
        transforms_dict: Camera-to-robot transformation matrices
        cam_intrinsics_dict: Camera intrinsic parameters
    """
    # Setup environment with fixed cube placement for reproducibility
    placement_initializer = UniformRandomSampler(
        name="ObjectSampler",
        mujoco_objects=None,
        x_range=[0.0, 0.0],   # Fixed X position
        y_range=[0.0, 0.0],   # Fixed Y position
        rotation=0.0,
        rotation_axis='z',
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array([0, 0, 0.8]),
        z_offset=0.01,
    )

    env = suite.make(
        env_name="Lift", 
        robots="Panda",  
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=VIEWPOINTS, 
        camera_depths=True,
        placement_initializer=placement_initializer,
        ignore_done=True
    )

    img_height, img_width = 256, 256
    cam_intrinsics_dict = get_intrinsics(env, VIEWPOINTS, img_height, img_width)

    if verbose:
        print("Camera intrinsics calculated.")

    # Phase 1: Pick up the cube
    obs = env.reset()
    policy = LiftPolicy(obs['cube_pos'])
    
    if verbose:
        print("Lifting cube for calibration...")
    
    while True:
        action = policy.get_action_lowdim(obs)
        obs, reward, _, _ = env.step(action)
        if reward == 1.0:
            break

    # Phase 2: Data collection at waypoints
    data_store = {name: {'cam': [], 'rob': []} for name in VIEWPOINTS}
    
    waypoints = generate_calibration_waypoints(obs["robot0_eef_quat"])

    print(f"Collecting data at {len(waypoints)} waypoints...")

    for i, waypoint in enumerate(waypoints):
        movto_policy = MoveToPolicy(waypoint)
        
        # Move to waypoint
        for a_i in range(50):
            eef_pos = obs["robot0_eef_pos"]
            eef_quat = obs["robot0_eef_quat"]
            action = movto_policy.get_action(eef_pos, eef_quat)
            obs, _, _, _ = env.step(action)
        
        # Detect cube in each camera view
        for cam_name in VIEWPOINTS:
            rgb = obs[cam_name+'_image'].copy()
            rgb = cv2.flip(rgb, 0)
            rgb_display = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            blob_uv = detect_red_blob(rgb)
            
            if blob_uv:
                cv2.circle(rgb_display, blob_uv, 10, (0, 255, 0), 2)
                
                depth_buffer = obs[cam_name+'_depth'].copy()
                depth = get_real_depth_map(env.sim, depth_buffer)
                depth = cv2.flip(depth, 0)
                
                if depth.ndim == 3:
                    depth = depth.squeeze()
                
                target_cam = pixel_to_camera3d(
                    blob_uv[0], blob_uv[1], depth, cam_intrinsics_dict[cam_name]
                )
                
                if target_cam is not None:
                    data_store[cam_name]['cam'].append(target_cam)
                    data_store[cam_name]['rob'].append(obs["robot0_eef_pos"])
                    
                    if verbose:
                        depth_val = depth[blob_uv[1], blob_uv[0]]
                        print(f"{cam_name} point {len(data_store[cam_name]['cam'])}: "
                              f"pixel={blob_uv}, depth={depth_val:.3f}m")
                else:
                    if verbose:
                        print(f"{cam_name}: Invalid depth at {blob_uv}")
            else:
                if verbose:
                    print(f"{cam_name}: No blob at waypoint {i+1}")

            if verbose:
                cv2.imshow(f'Calibration: {cam_name}', rgb_display)
                cv2.waitKey(50)

    if verbose:
        cv2.destroyAllWindows()

    # Phase 3: Solve for transformations
    transforms_dict = {}
    for cam_name in VIEWPOINTS:
        cam_pts = np.array(data_store[cam_name]['cam'])
        rob_pts = np.array(data_store[cam_name]['rob'])
        
        if len(cam_pts) >= 3:
            if verbose:
                print(f"\nSolving transform for {cam_name} using {len(cam_pts)} points...")
            
            T = solve_for_rigid_transformation(cam_pts, rob_pts)
            transforms_dict[cam_name] = T
            np.save(f"T_{cam_name}.npy", T)
            
            if verbose:
                print(f"  Translation: {T[:3, 3]}")
        else:
            print(f"Failed to calibrate {cam_name} (insufficient points)")

    # Phase 4: Verification
    if verbose:
        print("\n=== CALIBRATION VERIFICATION ===")
    
    for cam_name in VIEWPOINTS:
        if cam_name not in transforms_dict:
            continue
            
        cam_pts = np.array(data_store[cam_name]['cam'])
        rob_pts = np.array(data_store[cam_name]['rob'])
        T = transforms_dict[cam_name]
        
        # Transform camera points to robot frame
        cam_pts_homo = np.hstack([cam_pts, np.ones((len(cam_pts), 1))])
        cam_pts_in_robot = cam_pts_homo @ T.T
        cam_pts_in_robot = cam_pts_in_robot[:, :3]
        
        # Calculate reprojection error
        errors = np.linalg.norm(cam_pts_in_robot - rob_pts, axis=1)
        
        if verbose:
            print(f"{cam_name}:")
            print(f"  Mean error: {np.mean(errors):.4f}m")
            print(f"  Max error:  {np.max(errors):.4f}m")
            print(f"  Num points: {len(cam_pts)}")
            
            if np.mean(errors) > 0.05:
                print(f"  ⚠️  WARNING: High calibration error!")

    return transforms_dict, cam_intrinsics_dict


if __name__ == "__main__":
    run_hand_eye_calibration(verbose=True)