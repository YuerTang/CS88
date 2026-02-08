"""Quick debug: save a camera image to check what color space robosuite gives us."""
import numpy as np
import robosuite as suite
import cv2
from robosuite.utils.placement_samplers import UniformRandomSampler

placement_initializer = UniformRandomSampler(
    name="ObjectSampler",
    mujoco_objects=None,
    x_range=[0.0, 0.0],
    y_range=[0.0, 0.0],
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
    camera_names=["frontview", "agentview"],
    camera_depths=True,
    placement_initializer=placement_initializer,
    ignore_done=True,
)

obs = env.reset()

for cam in ["frontview", "agentview"]:
    img = obs[cam + "_image"].copy()
    img_flipped = cv2.flip(img, 0)

    # Save raw (as-is from robosuite, flipped)
    cv2.imwrite(f"/Users/yuertang/Desktop/2026Winter/CS 188/CS88/tmp/{cam}_raw.png", img_flipped)

    # Print pixel values at center to check channel order
    h, w = img_flipped.shape[:2]
    center = img_flipped[h // 2, w // 2]
    print(f"{cam} center pixel (raw): {center}  shape={img_flipped.shape}")

    # Also check what the red cube pixels look like
    # Try HSV with RGB2HSV
    hsv_rgb = cv2.cvtColor(img_flipped, cv2.COLOR_RGB2HSV)
    # Try HSV with BGR2HSV
    hsv_bgr = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2HSV)

    # Check mask with both
    m1_rgb = cv2.inRange(hsv_rgb, np.array([0, 100, 100]), np.array([10, 255, 255]))
    m2_rgb = cv2.inRange(hsv_rgb, np.array([170, 100, 100]), np.array([180, 255, 255]))
    mask_rgb = cv2.bitwise_or(m1_rgb, m2_rgb)

    m1_bgr = cv2.inRange(hsv_bgr, np.array([0, 100, 100]), np.array([10, 255, 255]))
    m2_bgr = cv2.inRange(hsv_bgr, np.array([170, 100, 100]), np.array([180, 255, 255]))
    mask_bgr = cv2.bitwise_or(m1_bgr, m2_bgr)

    print(f"  RGB2HSV red pixels: {np.count_nonzero(mask_rgb)}")
    print(f"  BGR2HSV red pixels: {np.count_nonzero(mask_bgr)}")

    cv2.imwrite(f"/Users/yuertang/Desktop/2026Winter/CS 188/CS88/tmp/{cam}_mask_rgb.png", mask_rgb)
    cv2.imwrite(f"/Users/yuertang/Desktop/2026Winter/CS 188/CS88/tmp/{cam}_mask_bgr.png", mask_bgr)
