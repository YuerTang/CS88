import numpy as np
from pid import PID
from scipy.spatial.transform import Rotation as R

class StackPolicy(object):
    """
    Claude assists me to write this low-level code.
    Policy for the Block Stacking task.

    This policy uses a STATE MACHINE approach:
    - We define a sequence of PHASES (states)
    - Each phase has a TARGET POSITION (waypoint) and GRIPPER STATE
    - We use PID control to smoothly move toward each waypoint
    - When we reach a waypoint (error < threshold), we transition to the next phase

    Phase Sequence:
        0: HOVER_ABOVE_A  - Move above Cube A (gripper open)
        1: DESCEND_TO_A   - Lower to Cube A grasp position (gripper open)
        2: GRASP_WAIT     - Stay at Cube A, wait for gripper to close
        3: LIFT_UP        - Lift Cube A up (gripper closed)
        4: MOVE_ABOVE_B   - Move above Cube B (gripper closed)
        5: DESCEND_TO_B   - Lower onto Cube B (gripper closed)
        6: RELEASE_WAIT   - Stay at Cube B, wait for gripper to open
        7: RETRACT        - Move up to clear block (gripper open)
    """

    KP = 3.0   # Proportional: higher = faster but may overshoot
    KI = 0.0   # Integral: helps eliminate steady-state error (start with 0)
    KD = 0.2   # Derivative: damping to reduce oscillation

    HOVER_HEIGHT = 0.08    # How high above objects to hover (safe clearance)
    CUBE_HEIGHT = 0.015     # Approximate height of cube (for grasp/stack offset)

    # Adaptive thresholds: tight for grasping, loose for stacking
    THRESHOLD_TIGHT = 0.015  # Phases 0-4: free-space motion (1.5cm)
    THRESHOLD_LOOSE = 0.03   # Phases 5-7: stacking with physical constraints (3cm)
    WAIT_STEPS = 30          # How many timesteps to wait for gripper actuation
    DT = 0.05              # Time step (20Hz control rate)

    GRIPPER_OPEN = -1
    GRIPPER_CLOSE = 1
    

    def __init__(self, obs):
        """
        Called ONCE at the start of each episode.

        Args:
            obs (dict): Initial observation containing object positions.
                - obs['cubeA_pos']: [x, y, z] position of Cube A (to pick up)
                - obs['cubeB_pos']: [x, y, z] position of Cube B (to stack on)
                - obs['robot0_eef_pos']: [x, y, z] robot end-effector position
        """

        cube_a_pos = np.array(obs['cubeA_pos']) 
        cube_b_pos = np.array(obs['cubeB_pos']) 

        hover_offset = np.array([0, 0, self.HOVER_HEIGHT])  
        grasp_offset = np.array([0, 0, -0.01]) #slightly down
        stack_offset = np.array([0, 0, self.CUBE_HEIGHT])

        self.waypoints = [
            cube_a_pos + hover_offset,                                      # 0: HOVER_A
            cube_a_pos + grasp_offset,                                      # 1: DESCEND_A
            cube_a_pos + grasp_offset,                                      # 2: GRASP
            cube_a_pos + hover_offset,                                      # 3: LIFT
            cube_b_pos + hover_offset + np.array([0, 0, self.CUBE_HEIGHT]), # 4: MOVE_B
            cube_b_pos + stack_offset,                                      # 5: DESCEND_B
            cube_b_pos + stack_offset,                                      # 6: RELEASE
            cube_b_pos + stack_offset + hover_offset,                       # 7: RETRACT
        ]

        self.phase = 0           
        self.wait_counter = 0    

        self.pid = PID(
            kp=self.KP,
            ki=self.KI,
            kd=self.KD,
            target=self.waypoints[0]
        )

    def get_action(self, obs):
        """
        Claude assists me to write this low-level code.
        Called EVERY TIMESTEP during the episode.

        This function:
        1. Gets current robot position
        2. Uses PID to compute control signal toward current waypoint
        3. Checks if we should transition to the next phase
        4. Builds and returns the 7D action vector

        Args:
            obs (dict): Current observation with robot and object states.

        Returns:
            np.ndarray: 7D action vector [dx, dy, dz, ax, ay, az, gripper]
                - [0:3] Position delta from PID controller
                - [3:6] Rotation delta (we use zeros - no rotation control)
                - [6]   Gripper command (-1=open, +1=close)
        """
        current_pos = np.array(obs['robot0_eef_pos'])
        control = self.pid.update(current_pos, self.DT)

        error = self.pid.get_error()
        threshold = self._get_threshold()

        if error < threshold:
            if self.phase in [2, 6]:  # Takes time for gripper to open / closes
                self.wait_counter += 1
                if self.wait_counter >= self.WAIT_STEPS:
                    self._advance_phase()
            else:
                self._advance_phase()
        action = np.zeros(7)

        action[0] = control[0] 
        action[1] = control[1]  
        action[2] = control[2]  

        action[3] = 0  
        action[4] = 0  
        action[5] = 0 

        action[6] = self._get_gripper_state()

        return action

    def _advance_phase(self):
        """
        Move to the next phase in the state machine.

        This function:
        1. Increments the phase counter
        2. Resets the PID controller with the new target waypoint
        3. Resets the wait counter
        """
        if self.phase < len(self.waypoints) - 1:
            self.phase += 1
            self.pid.reset(target=self.waypoints[self.phase])
            self.wait_counter = 0

    def _get_threshold(self):
        """
        Get adaptive threshold based on current phase.

        Phases 0-4 (free-space motion): Use tight threshold for precision grasping.
        Phases 5-7 (stacking): Use loose threshold due to physical constraints
                               when placing block on top of another.
        """
        if self.phase <= 4:
            return self.THRESHOLD_TIGHT
        else:
            return self.THRESHOLD_LOOSE

    def _get_gripper_state(self):
        """
        Determine gripper state based on current phase.

        Gripper Logic:
        - Phases 0-1: OPEN (approaching cube A)
        - Phases 2-5: CLOSED (grasping and carrying cube A)
        - Phases 6-7: OPEN (releasing and retracting)

        Returns:
            float: -1 for open, +1 for closed
        """
        if self.phase < 2:

            return self.GRIPPER_OPEN
        elif self.phase < 6:
            return self.GRIPPER_CLOSE
        else:
            return self.GRIPPER_OPEN

class NutAssemblyPolicy(object):
    """
    Policy for the Nut Assembly task.

    [AI Declaration]: Generated using Claude with the prompt:
    "Implement NutAssemblyPolicy to fit square nut on square peg and round nut on round peg"

    Goal: Fit square nut on square peg and round nut on round peg.

    This policy uses a STATE MACHINE approach with 16 phases:
    - Phases 0-7: Pick up square nut and place on square peg (peg1)
    - Phases 8-15: Pick up round nut and place on round peg (peg2)

    Phase Sequence (repeated for each nut):
        0/8:  HOVER_ABOVE      - Move above nut, outside radius (gripper open)
        1/9:  DESCEND_OUTSIDE  - Lower while staying outside radius (gripper open)
        2/10: MOVE_TO_HANDLE   - Move inward to handle position (gripper open)
        3/11: GRASP_WAIT       - Wait for gripper to close
        4/12: LIFT_NUT         - Lift nut up (gripper closed)
        5/13: MOVE_TO_PEG      - Move above target peg (gripper closed)
        6/14: INSERT_PEG       - Lower nut onto peg (gripper closed)
        7/15: RELEASE_WAIT     - Wait for gripper to open
    """

    # PID gains (same as StackPolicy)
    KP = 3.0
    KI = 0.0
    KD = 0.2
    DT = 0.05

    # Height parameters
    HOVER_HEIGHT = 0.08       # Height above nut to hover (same as StackPolicy)
    GRASP_OFFSET_Z = 0.0      # Grasp at nut level (can't go lower - table in the way)
    HANDLE_OFFSET_SQUARE = 0.054  # From square-nut.xml: handle_site pos="0.054 0 0"
    HANDLE_OFFSET_ROUND = 0.06    # From round-nut.xml: handle_site pos="0.06 0 0" (graspable part)
    APPROACH_EXTRA = 0.03     # Extra distance beyond handle for approach (avoid collision)
    PEG_HOVER_HEIGHT = 0.15   # Height above peg for clearance (higher to avoid collision)
    PEG_INSERT_HEIGHT = 0.04  # Height above peg base to release

    # Thresholds
    THRESHOLD_TIGHT = 0.008   # For precise positioning - must be very close!
    THRESHOLD_LOOSE = 0.06    # For movement phases (allow physical constraint tolerance)
    WAIT_STEPS = 60           # Steps to wait for gripper actuation (more time to close)

    # Yaw rotation control (to align gripper with handle direction)
    YAW_GAIN = 0.2            # Proportional gain for yaw rotation (using relative quaternion method)

    # Gripper states
    GRIPPER_OPEN = -1
    GRIPPER_CLOSE = 1

    # FIXED peg positions (discovered from MuJoCo simulation)
    PEG1_POS = np.array([0.23, 0.10, 0.85])   # Square peg
    PEG2_POS = np.array([0.23, -0.10, 0.85])  # Round peg

    def __init__(self, obs):
        """
        Initialize the NutAssembly policy.

        Args:
            obs (dict): Initial observation containing:
                - obs['SquareNut_pos']: [x, y, z] position of square nut
                - obs['SquareNut_quat']: [w, x, y, z] quaternion orientation
                - obs['RoundNut_pos']: [x, y, z] position of round nut
                - obs['RoundNut_quat']: [w, x, y, z] quaternion orientation
                - obs['robot0_eef_pos']: [x, y, z] robot end-effector position
        """
        # Get nut positions and orientations (randomized each episode)
        square_nut_pos = np.array(obs['SquareNut_pos'])
        round_nut_pos = np.array(obs['RoundNut_pos'])
        square_nut_quat = obs['SquareNut_quat']
        round_nut_quat = obs['RoundNut_quat']

        # Compute initial handle offsets and target yaw based on nut rotation
        # These will be updated dynamically during grasp phases
        self.sq_handle_offset, self.sq_target_yaw = self._get_handle_offset_and_yaw(
            square_nut_quat, self.HANDLE_OFFSET_SQUARE
        )
        self.rd_handle_offset, self.rd_target_yaw = self._get_handle_offset_and_yaw(
            round_nut_quat, self.HANDLE_OFFSET_ROUND
        )

        # Build waypoints (will be overridden by dynamic tracking for grasp phases)
        # For peg phases, we'll update waypoints when we know the actual grasp offset
        waypoints_square = self._build_waypoints(
            square_nut_pos, self.PEG1_POS, self.sq_handle_offset, self.HANDLE_OFFSET_SQUARE
        )
        waypoints_round = self._build_waypoints(
            round_nut_pos, self.PEG2_POS, self.rd_handle_offset, self.HANDLE_OFFSET_ROUND
        )
        self.waypoints = waypoints_square + waypoints_round

        # State machine
        self.phase = 0
        self.wait_counter = 0

        # Initialize PID controller
        self.pid = PID(
            kp=self.KP,
            ki=self.KI,
            kd=self.KD,
            target=self.waypoints[0]
        )

    def _get_handle_offset_and_yaw(self, nut_quat, handle_distance):
        """
        Compute world-frame offset to nut handle and target yaw angle.

        Uses 2D trigonometry since the nut only rotates around Z-axis (yaw).
        The handle is at local position (handle_distance, 0, 0) in nut's frame.

        Args:
            nut_quat: Quaternion orientation of the nut [x, y, z, w]
            handle_distance: Distance from nut center to handle (meters)

        Returns:
            tuple: (offset_array, theta)
                - offset_array: 3D offset in world frame [x, y, 0]
                - theta: Yaw angle of handle (radians)
        """
        # Extract yaw angle from quaternion
        # MuJoCo/robosuite uses [x, y, z, w] quaternion format (not [w, x, y, z])
        # For Z-axis rotation: w = cos(θ/2), z = sin(θ/2)
        # Therefore: θ = 2 * atan2(z, w)
        w = nut_quat[3]  # w is the 4th component in [x, y, z, w]
        z = nut_quat[2]  # z is the 3rd component
        theta = 2 * np.arctan2(z, w)

        # 2D rotation of handle offset [handle_distance, 0] by angle theta
        offset_x = handle_distance * np.cos(theta)
        offset_y = handle_distance * np.sin(theta)
        offset_z = 0.0  # Handle is at same height as nut center

        return np.array([offset_x, offset_y, offset_z]), theta

    def _build_waypoints(self, nut_pos, peg_pos, handle_offset, handle_distance):
        """
        Build waypoint sequence for one nut-to-peg assembly.

        Args:
            nut_pos: 3D position of the nut
            peg_pos: 3D position of the target peg
            handle_offset: 3D offset to handle [x, y, 0] based on nut rotation
            handle_distance: Distance from nut center to handle (for approach calc)

        Returns:
            List of 8 waypoints for the assembly sequence
        """
        hover_offset = np.array([0, 0, self.HOVER_HEIGHT])
        grasp_z_offset = np.array([0, 0, self.GRASP_OFFSET_Z])
        peg_hover_offset = np.array([0, 0, self.PEG_HOVER_HEIGHT])
        peg_insert_offset = np.array([0, 0, self.PEG_INSERT_HEIGHT])

        # Compute approach offset (radially outward from handle to avoid nut body)
        approach_direction = handle_offset / handle_distance  # unit vector
        approach_offset = approach_direction * self.APPROACH_EXTRA

        return [
            nut_pos + hover_offset + handle_offset + approach_offset,   # 0: HOVER_ABOVE (outside, high)
            nut_pos + grasp_z_offset + handle_offset + approach_offset, # 1: DESCEND_OUTSIDE (outside, low)
            nut_pos + grasp_z_offset + handle_offset,                   # 2: MOVE_TO_HANDLE (at handle, low)
            nut_pos + grasp_z_offset + handle_offset,                   # 3: GRASP_WAIT
            nut_pos + hover_offset + handle_offset,                     # 4: LIFT_NUT
            peg_pos + peg_hover_offset + handle_offset,                 # 5: MOVE_TO_PEG
            peg_pos + peg_insert_offset + handle_offset,                # 6: INSERT_PEG
            peg_pos + peg_insert_offset + handle_offset,                # 7: RELEASE_WAIT
        ]

    def get_action(self, obs):
        """
        Compute action for current timestep.

        Args:
            obs (dict): Current observation with robot state.

        Returns:
            np.ndarray: 7D action [dx, dy, dz, ax, ay, az, gripper]
        """
        current_pos = np.array(obs['robot0_eef_pos'])

        # DYNAMIC TARGET UPDATE: For approach/grasp phases, track current nut position
        # and rotation because nut may have moved/settled since init
        if self.phase in [0, 1, 2, 3]:  # Square nut approach/grasp phases
            current_nut = np.array(obs['SquareNut_pos'])
            nut_quat = obs['SquareNut_quat']
            # Compute handle offset and target yaw based on CURRENT rotation
            self.sq_handle_offset, self.sq_target_yaw = self._get_handle_offset_and_yaw(
                nut_quat, self.HANDLE_OFFSET_SQUARE
            )

            # Compute approach offset (radially outward)
            approach_direction = self.sq_handle_offset / self.HANDLE_OFFSET_SQUARE  # unit vector
            approach_offset = approach_direction * self.APPROACH_EXTRA

            if self.phase == 0:  # HOVER_ABOVE - outside + high
                target = current_nut + np.array([0, 0, self.HOVER_HEIGHT]) + self.sq_handle_offset + approach_offset
            elif self.phase == 1:  # DESCEND_OUTSIDE - outside + low
                target = current_nut + np.array([0, 0, self.GRASP_OFFSET_Z]) + self.sq_handle_offset + approach_offset
            else:  # Phase 2, 3: MOVE_TO_HANDLE / GRASP_WAIT - at handle + low
                target = current_nut + np.array([0, 0, self.GRASP_OFFSET_Z]) + self.sq_handle_offset
            self.pid.reset(target=target)

            # DEBUG - show gripper position, target, and error distance
            error_dist = np.linalg.norm(target - current_pos)
            print(f"Phase {self.phase}: gripper={current_pos}, target={target}, error={error_dist:.4f} (need <{self._get_threshold()})")

            # When we grasp (phase 3), update lift+peg waypoints with final handle offset
            if self.phase == 3:
                self._update_waypoints_square(obs)

        elif self.phase in [8, 9, 10, 11]:  # Round nut approach/grasp phases
            current_nut = np.array(obs['RoundNut_pos'])
            nut_quat = obs['RoundNut_quat']
            # Compute handle offset and target yaw based on CURRENT rotation
            self.rd_handle_offset, self.rd_target_yaw = self._get_handle_offset_and_yaw(
                nut_quat, self.HANDLE_OFFSET_ROUND
            )

            # Compute approach offset (radially outward)
            approach_direction = self.rd_handle_offset / self.HANDLE_OFFSET_ROUND  # unit vector
            approach_offset = approach_direction * self.APPROACH_EXTRA

            if self.phase == 8:  # HOVER_ABOVE - outside + high
                target = current_nut + np.array([0, 0, self.HOVER_HEIGHT]) + self.rd_handle_offset + approach_offset
            elif self.phase == 9:  # DESCEND_OUTSIDE - outside + low
                target = current_nut + np.array([0, 0, self.GRASP_OFFSET_Z]) + self.rd_handle_offset + approach_offset
            else:  # Phase 10, 11: MOVE_TO_HANDLE / GRASP_WAIT - at handle + low
                target = current_nut + np.array([0, 0, self.GRASP_OFFSET_Z]) + self.rd_handle_offset
            self.pid.reset(target=target)

            # When we grasp (phase 11), update lift+peg waypoints with final handle offset
            if self.phase == 11:
                self._update_waypoints_round(obs)

        # Phases 4-7 and 12-15: Use FIXED waypoints (nut moves with gripper)

        control = self.pid.update(current_pos, self.DT)
        error = self.pid.get_error()

        # Get threshold based on phase
        threshold = self._get_threshold()

        # Phase transition logic
        if error < threshold:
            if self._is_wait_phase():
                self.wait_counter += 1
                if self.wait_counter >= self.WAIT_STEPS:
                    self._advance_phase()
            else:
                self._advance_phase()

        # Build action
        action = np.zeros(7)
        action[0:3] = control          # Position control from PID
        action[6] = self._get_gripper_state()

        # Apply yaw rotation during approach phases to align gripper with handle
        # Use relative quaternion from nut to gripper - directly encodes the yaw error
        if self.phase in [0, 1, 2]:  # Square nut hover/descend/move-to-handle
            rel_quat = obs['SquareNut_to_robot0_eef_quat']
            if np.sum(np.abs(rel_quat)) > 0:  # Check quaternion is valid
                yaw_error = R.from_quat(rel_quat).as_euler('xyz')[2]
                action[5] = self.YAW_GAIN * yaw_error
        elif self.phase in [8, 9, 10]:  # Round nut hover/descend/move-to-handle
            rel_quat = obs['RoundNut_to_robot0_eef_quat']
            if np.sum(np.abs(rel_quat)) > 0:  # Check quaternion is valid
                yaw_error = R.from_quat(rel_quat).as_euler('xyz')[2]
                action[5] = self.YAW_GAIN * yaw_error
        # Other phases: no rotation (action[3:6] already 0)

        return action

    def _update_waypoints_square(self, obs):
        """Update square nut waypoints (lift + peg) with actual grasp handle offset."""
        current_nut = np.array(obs['SquareNut_pos'])
        hover = np.array([0, 0, self.HOVER_HEIGHT])
        peg_hover = np.array([0, 0, self.PEG_HOVER_HEIGHT])
        peg_insert = np.array([0, 0, self.PEG_INSERT_HEIGHT])

        # Waypoint 4: LIFT - use current nut pos + handle offset
        self.waypoints[4] = current_nut + hover + self.sq_handle_offset
        # Waypoints 5, 6, 7: PEG phases
        self.waypoints[5] = self.PEG1_POS + peg_hover + self.sq_handle_offset
        self.waypoints[6] = self.PEG1_POS + peg_insert + self.sq_handle_offset
        self.waypoints[7] = self.PEG1_POS + peg_insert + self.sq_handle_offset

    def _update_waypoints_round(self, obs):
        """Update round nut waypoints (lift + peg) with actual grasp handle offset."""
        current_nut = np.array(obs['RoundNut_pos'])
        hover = np.array([0, 0, self.HOVER_HEIGHT])
        peg_hover = np.array([0, 0, self.PEG_HOVER_HEIGHT])
        peg_insert = np.array([0, 0, self.PEG_INSERT_HEIGHT])

        # Waypoint 12: LIFT - use current nut pos + handle offset
        self.waypoints[12] = current_nut + hover + self.rd_handle_offset
        # Waypoints 13, 14, 15: PEG phases
        self.waypoints[13] = self.PEG2_POS + peg_hover + self.rd_handle_offset
        self.waypoints[14] = self.PEG2_POS + peg_insert + self.rd_handle_offset
        self.waypoints[15] = self.PEG2_POS + peg_insert + self.rd_handle_offset

    def _advance_phase(self):
        """Move to next phase and reset PID controller."""
        if self.phase < len(self.waypoints) - 1:
            self.phase += 1
            self.pid.reset(target=self.waypoints[self.phase])
            self.wait_counter = 0

    def _is_wait_phase(self):
        """Check if current phase requires waiting for gripper."""
        # Wait phases: 3 (grasp sq), 7 (release sq), 11 (grasp rd), 15 (release rd)
        return self.phase in [3, 7, 11, 15]

    def _get_threshold(self):
        """Get adaptive threshold based on current phase."""
        # TIGHT only for handle approach and grasp phases (need precise positioning)
        # LOOSE for all other movement phases (hover, lift, move, insert, release)
        if self.phase in [2, 3, 10, 11]:  # Move-to-handle and grasp phases - need precision
            return self.THRESHOLD_TIGHT
        return self.THRESHOLD_LOOSE  # All other phases - movement tolerance OK

    def _get_gripper_state(self):
        """
        Determine gripper state based on current phase.

        Gripper Logic:
        - Phases 0-2:   OPEN (approaching square nut handle)
        - Phases 3-6:   CLOSED (grasping and placing square nut)
        - Phases 7-10:  OPEN (releasing square, approaching round nut)
        - Phases 11-14: CLOSED (grasping and placing round nut)
        - Phase 15:     OPEN (releasing round nut)
        """
        if self.phase < 3:
            return self.GRIPPER_OPEN
        elif self.phase < 7:
            return self.GRIPPER_CLOSE
        elif self.phase < 11:
            return self.GRIPPER_OPEN
        elif self.phase < 15:
            return self.GRIPPER_CLOSE
        else:
            return self.GRIPPER_OPEN


class DoorPolicy(object):
    """
    Policy for the Door Opening task.

    [AI Declaration]: Generated using Claude with the prompt:
    "Fix DoorPolicy to rotate gripper 90 degrees, reach handle, rotate handle, open door"

    Phase Sequence:
        0: ORIENT       - Rotate gripper 90 degrees (roll) so fingers can wrap handle
        1: APPROACH     - Move to offset position near handle (gripper open)
        2: REACH        - Move to handle grip position (gripper open)
        3: GRIP         - Close gripper and wait for secure grip
        4: ROTATE       - Rotate handle to unlatch door
        5: PULL         - Pull door open while tracking handle
    """

    KP = 3.0
    KI = 0.0
    KD = 0.2
    DT = 0.05

    THRESHOLD = 0.05
    ORIENT_STEPS = 45  # 45 steps for ~90 degrees rotation
    WAIT_STEPS = 30
    ROTATE_STEPS = 600  # Max steps, but may transition earlier based on handle position
    HANDLE_THRESHOLD = 1.4  # Transition to pull when handle reaches this angle

    GRIPPER_OPEN = -1
    GRIPPER_CLOSE = 1

    # Offsets discovered through experimentation
    APPROACH_OFFSET = np.array([0.12, 0.0, 0.07])
    GRIP_OFFSET = np.array([0.0, -0.02, -0.02])  # Y=-0.02 to center grip on handle
    PULL_FORCE = 0.4  # Constant force to pull door open

    # Rotation parameters
    # X-axis rotation (action[3]) - rotates gripper 90 degrees
    ORIENT_ROTATION = np.array([-0.3, 0.0, 0.0])  # Rotate around X-axis
    HANDLE_ROTATION = np.array([0.0, -0.15, 0.0])  # Moderate rotation for handle

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

        # Store initial positions
        self.initial_handle_pos = handle_pos.copy()
        self.initial_eef_pos = eef_pos.copy()

        # Pull direction will be set AFTER rotation phase
        self.pull_direction = None

        # State machine
        self.phase = 0
        self.step_counter = 0

        # Initialize PID controller (target set per phase)
        self.pid = PID(
            kp=self.KP,
            ki=self.KI,
            kd=self.KD,
            target=eef_pos  # Start at current position for ORIENT phase
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

        action = np.zeros(7)

        if self.phase == 0:  # ORIENT - rotate gripper 90 degrees (pitch down)
            # Use PID to maintain position while rotating
            self.pid.reset(target=self.initial_eef_pos)
            control = self.pid.update(current_pos, self.DT)
            action[0:3] = control  # Maintain position
            action[3:6] = self.ORIENT_ROTATION  # Apply rotation
            action[6] = self.GRIPPER_OPEN
            self.step_counter += 1
            if self.step_counter >= self.ORIENT_STEPS:
                self.phase = 1
                self.step_counter = 0
            return action

        elif self.phase == 1:  # APPROACH
            target = self.initial_handle_pos + self.APPROACH_OFFSET
            self.pid.reset(target=target)
            control = self.pid.update(current_pos, self.DT)
            error = self.pid.get_error()

            action[0:3] = control
            action[6] = self.GRIPPER_OPEN

            if error < self.THRESHOLD:
                self.phase = 2
            return action

        elif self.phase == 2:  # REACH
            target = handle_pos + self.GRIP_OFFSET
            self.pid.reset(target=target)
            control = self.pid.update(current_pos, self.DT)
            error = self.pid.get_error()

            action[0:3] = control
            action[6] = self.GRIPPER_OPEN

            if error < self.THRESHOLD:
                self.phase = 3
                self.step_counter = 0
            return action

        elif self.phase == 3:  # GRIP
            target = handle_pos + self.GRIP_OFFSET
            self.pid.reset(target=target)
            control = self.pid.update(current_pos, self.DT)

            action[0:3] = control
            action[6] = self.GRIPPER_CLOSE

            self.step_counter += 1
            if self.step_counter >= self.WAIT_STEPS:
                self.phase = 4
                self.step_counter = 0
            return action

        elif self.phase == 4:  # ROTATE handle
            target = handle_pos + self.GRIP_OFFSET
            self.pid.reset(target=target)
            control = self.pid.update(current_pos, self.DT)

            action[0:3] = control
            action[3:6] = self.HANDLE_ROTATION
            action[6] = self.GRIPPER_CLOSE

            # Get handle rotation angle
            handle_qpos = obs.get('handle_qpos', 0)
            if hasattr(handle_qpos, '__len__'):
                handle_qpos = handle_qpos[0]

            self.step_counter += 1
            # Transition to pull when handle rotated enough OR max steps reached
            if handle_qpos >= self.HANDLE_THRESHOLD or self.step_counter >= self.ROTATE_STEPS:
                # Pull in +Y direction (toward robot)
                # Door hinge is on the left, pulling +Y swings door open
                self.pull_direction = np.array([0.0, 1.0, 0.0])

                self.phase = 5
                self.step_counter = 0
            return action

        elif self.phase == 5:  # PULL door open
            # Track handle position to maintain grip
            target = handle_pos + self.GRIP_OFFSET
            self.pid.reset(target=target)
            control = self.pid.update(current_pos, self.DT)

            # Add constant pull force to swing door open
            pull_force = self.pull_direction * self.PULL_FORCE

            action[0:3] = control + pull_force
            # Continue rotating to prevent handle from springing back
            action[3:6] = self.HANDLE_ROTATION
            action[6] = self.GRIPPER_CLOSE
            return action

        return action