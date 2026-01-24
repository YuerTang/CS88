"""
DoorPolicy - Final Implementation
Based on Lab 8 findings

Copy this class to policies.py
"""
import numpy as np
from pid import PID


class DoorPolicy(object):
    """
    Policy for the Door Opening task.

    Phase Sequence:
        0: APPROACH     - Move to offset position above/beside handle (gripper open)
        1: REACH        - Move to handle grip position (gripper open)
        2: GRIP_WAIT    - Close gripper and wait for grip
        3: PULL         - Pull toward fixed target to open door
    """

    KP = 3.0
    KI = 0.0
    KD = 0.2
    DT = 0.05

    THRESHOLD = 0.05
    WAIT_STEPS = 30

    GRIPPER_OPEN = -1
    GRIPPER_CLOSE = 1

    # Offsets discovered through experimentation
    APPROACH_OFFSET = np.array([0.13, 0.0, 0.07])
    GRIP_OFFSET = np.array([0.0, 0.0, -0.02])
    PULL_DISTANCE = 0.5

    def __init__(self, obs):
        """
        Initialize the door opening policy.

        Args:
            obs (dict): Initial observation containing:
                - obs['handle_pos']: [x, y, z] position of door handle
                - obs['robot0_eef_pos']: [x, y, z] robot end-effector position
        """
        handle_pos = np.array(obs['handle_pos'])
        eef_pos = np.array(obs['robot0_eef_pos'])

        # Calculate pull direction (from handle toward robot, in XY plane)
        pull_direction = eef_pos - handle_pos
        pull_direction[2] = 0  # Keep Z level
        pull_direction = pull_direction / np.linalg.norm(pull_direction)
        self.pull_direction = pull_direction

        # Store initial handle position for approach waypoint
        self.initial_handle_pos = handle_pos.copy()

        # Waypoints (some are dynamic, computed in get_action)
        self.approach_target = handle_pos + self.APPROACH_OFFSET

        # Pull target will be set when entering pull phase
        self.pull_target = None

        # State machine
        self.phase = 0
        self.wait_counter = 0

        # Initialize PID controller
        self.pid = PID(
            kp=self.KP,
            ki=self.KI,
            kd=self.KD,
            target=self.approach_target
        )

    def get_action(self, obs):
        """
        Compute action for current timestep.

        Args:
            obs (dict): Current observation.

        Returns:
            np.ndarray: 7D action [dx, dy, dz, ax, ay, az, gripper]
        """
        current_pos = np.array(obs['robot0_eef_pos'])
        handle_pos = np.array(obs['handle_pos'])

        # Determine target and gripper state based on phase
        if self.phase == 0:  # APPROACH
            target = self.approach_target
            gripper = self.GRIPPER_OPEN

        elif self.phase == 1:  # REACH (with grip offset)
            target = handle_pos + self.GRIP_OFFSET
            gripper = self.GRIPPER_OPEN

        elif self.phase == 2:  # GRIP_WAIT
            target = handle_pos + self.GRIP_OFFSET
            gripper = self.GRIPPER_CLOSE

        elif self.phase == 3:  # PULL
            target = self.pull_target  # Fixed target!
            gripper = self.GRIPPER_CLOSE

        else:
            # Stay at pull target if somehow past phase 3
            target = self.pull_target if self.pull_target is not None else current_pos
            gripper = self.GRIPPER_CLOSE

        # Update PID target if changed
        self.pid.reset(target=target)

        # Compute control signal
        control = self.pid.update(current_pos, self.DT)
        error = self.pid.get_error()

        # Check phase transitions
        if error < self.THRESHOLD:
            if self.phase == 0:
                self._advance_phase()
            elif self.phase == 1:
                self._advance_phase()
            elif self.phase == 2:
                self.wait_counter += 1
                if self.wait_counter >= self.WAIT_STEPS:
                    # Set fixed pull target before advancing
                    self.pull_target = current_pos + self.pull_direction * self.PULL_DISTANCE
                    self._advance_phase()
            # Phase 3: keep pulling, no transition needed

        # Build action vector
        action = np.zeros(7)
        action[0:3] = control
        action[3:6] = 0  # No rotation control needed
        action[6] = gripper

        return action

    def _advance_phase(self):
        """Move to next phase."""
        self.phase += 1
        self.wait_counter = 0


# Test code - remove when copying to policies.py
if __name__ == "__main__":
    import robosuite as suite

    env = suite.make(
        env_name="Door",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_latch=False,
    )

    # Test with multiple seeds
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        obs = env.reset()
        policy = DoorPolicy(obs)

        print(f"\n{'='*50}")
        print(f"Testing seed {seed}")
        print(f"{'='*50}")

        for step in range(600):
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            env.render()

            if reward == 1.0:
                print(f"SUCCESS at step {step}!")
                break

        if reward != 1.0:
            print(f"FAILED - hinge={obs['hinge_qpos']:.4f}")

    env.close()
