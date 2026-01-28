import numpy as np
from pid import PID
from scipy.spatial.transform import Rotation as R

class StackPolicy(object):
    """
    [Short Description]: 8-phase state machine policy for pick-and-place cube stacking.
    [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
    "Implement StackPolicy state machine for pick and place cube stacking with adaptive thresholds"

    Phase Sequence:
        0: HOVER_ABOVE_A  - Move above Cube A (gripper open)
        1: DESCEND_TO_A   - Lower to Cube A grasp position (gripper open)
        2: GRASP_WAIT     - Stay at Cube A, wait for gripper to close
        3: LIFT_UP        - Lift Cube A up (gripper closed)
        4: MOVE_ABOVE_B   - Move above Cube B (gripper closed)
        5: DESCEND_TO_B   - Lower onto Cube B (gripper closed)
        6: RELEASE_WAIT   - Stay at Cube B, wait for gripper to open
        7: RETRACT        - Move up to clear block (gripper open)

    Notes:
        Adaptive thresholds: TIGHT (1.5cm) for phases 0-4, LOOSE (3cm) for phases 5-7.
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
        [Short Description]: Initialize policy with waypoints from cube positions.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Set up waypoints for 8-phase state machine based on cube A and B positions"

        Args:
            obs (dict): Initial observation with cubeA_pos, cubeB_pos, robot0_eef_pos.

        Returns:
            None

        Notes:
            RETRACT phase added to separate gripper from block after release.
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
        [Short Description]: Compute 7D action with PID control and phase transitions.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Implement get_action with PID control, phase transitions, and gripper logic"

        Args:
            obs (dict): Current observation with robot and object states.

        Returns:
            np.ndarray: 7D action [dx, dy, dz, ax, ay, az, gripper].

        Notes:
            Wait phases (2, 6) require WAIT_STEPS before advancing.
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
        [Short Description]: Advance to next phase and reset PID controller.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Implement phase transition with PID reset for new waypoint target"

        Args:
            None

        Returns:
            None

        Notes:
            PID reset clears integral/derivative error for clean transitions.
        """
        if self.phase < len(self.waypoints) - 1:
            self.phase += 1
            self.pid.reset(target=self.waypoints[self.phase])
            self.wait_counter = 0

    def _get_threshold(self):
        """
        [Short Description]: Get adaptive threshold based on current phase.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Add adaptive thresholds - tight for grasping, loose for stacking phases"

        Args:
            None

        Returns:
            float: Position error threshold in meters.

        Notes:
            TIGHT (1.5cm) for phases 0-4, LOOSE (3cm) for phases 5-7.
        """
        if self.phase <= 4:
            return self.THRESHOLD_TIGHT
        else:
            return self.THRESHOLD_LOOSE

    def _get_gripper_state(self):
        """
        [Short Description]: Determine gripper command based on current phase.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Map phases to gripper states for pick and place sequence"

        Args:
            None

        Returns:
            float: -1 (open) or +1 (close).

        Notes:
            OPEN: 0-1, 6-7. CLOSE: 2-5.
        """
        if self.phase < 2:

            return self.GRIPPER_OPEN
        elif self.phase < 6:
            return self.GRIPPER_CLOSE
        else:
            return self.GRIPPER_OPEN

class NutAssemblyPolicy(object):
    """
    [Short Description]: 19-phase state machine policy for assembling two nuts onto pegs.
    [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
    "Implement NutAssemblyPolicy to fit square nut on square peg and round nut on round peg
    using outside-in approach, quaternion-based handle offset, and yaw rotation control"

    Goal: Fit square nut on square peg and round nut on round peg.

    This policy uses a STATE MACHINE approach with 19 phases:
    - Phases 0-8: Pick up round nut, align, and place on round peg (peg2)
    - Phase 9: TRANSITION_WAIT - Wait between the two tasks
    - Phases 10-18: Pick up square nut, align, and place on square peg (peg1)

    Phase Sequence (repeated for each nut):
        0/10:  HOVER_ABOVE      - Move above nut, outside radius (gripper open)
        1/11:  DESCEND_OUTSIDE  - Lower while staying outside radius (gripper open)
        2/12:  MOVE_TO_HANDLE   - Move inward to handle position (gripper open)
        3/13:  GRASP_WAIT       - Wait for gripper to close
        4/14:  LIFT_NUT         - Lift nut up (gripper closed)
        5/15:  ROTATE_ALIGN     - Rotate so handle points toward robot (gripper closed)
        6/16:  MOVE_TO_PEG      - Move above target peg in straight line (gripper closed)
        7/17:  INSERT_PEG       - Lower nut onto peg (gripper closed)
        8/18:  RELEASE_WAIT     - Wait for gripper to open

    Notes:
        Handle offset uses 2D trigonometry (nuts only rotate around Z-axis).
        Yaw control uses relative quaternion during approach, absolute nut yaw during rotate-align.
    """

    # PID gains (same as StackPolicy)
    KP = 3.0
    KI = 0.0  # Keep at 0 - using target update instead of reset for convergence
    KD = 0.2
    DT = 0.05

    # Height parameters
    HOVER_HEIGHT = 0.08       # Height above nut to hover (same as StackPolicy)
    GRASP_OFFSET_Z = -0.01    # Grasp 1cm below nut center (accounts for EEF-to-finger offset)
    HANDLE_OFFSET_SQUARE = 0.054  # From square-nut.xml: handle_site pos="0.054 0 0"
    HANDLE_OFFSET_ROUND = 0.06    # From round-nut.xml: handle_site pos="0.06 0 0"
    APPROACH_EXTRA = 0.04         # Extra distance beyond handle for approach (increased to avoid collision)
    PEG_HOVER_HEIGHT = 0.20   # Height above peg for clearance (increased to avoid nut-peg collision)
    PEG_INSERT_HEIGHT = 0.0   # Height at peg base to release (lowered for proper insertion)

    # Thresholds
    THRESHOLD_TIGHT = 0.025   # For precise positioning (2.5cm - balance between convergence and accuracy)
    THRESHOLD_PEG = 0.01      # For peg alignment - must be smaller than clearance (1cm)
    THRESHOLD_LOOSE = 0.06    # For movement phases (allow physical constraint tolerance)
    WAIT_STEPS = 90           # Steps to wait for gripper actuation (increased for secure grip)
    TRANSITION_WAIT_STEPS = 60  # Steps to wait between completing round nut and starting square nut
    ROTATE_ALIGN_STEPS = 120  # Max steps to rotate (increased for safety)
    ROTATE_ALIGN_MIN_STEPS = 30  # Minimum steps before checking yaw alignment
    YAW_ALIGN_THRESHOLD = 0.1  # Yaw must be within ~5.7 degrees of target to proceed

    # Target yaw for aligned handle
    # Round nut: handle points TOWARD robot (+X from nut, so nut body is far, but gripper is close)
    # We want yaw = 0 so handle is in +X direction from nut center
    # BUT since peg is at X=0.23 and robot at X~0, we want gripper at X < 0.23
    # So handle should be in -X direction from nut = yaw = π
    ALIGNED_HANDLE_YAW = np.pi  # Handle points in -X direction (toward robot)

    # Yaw rotation control (to align gripper with handle direction)
    YAW_GAIN = 1.0            # Proportional gain for yaw rotation (high for fast alignment)
    INSERTION_YAW_GAIN = 2.0  # Higher gain during insertion to fight contact forces
    ALIGN_GAIN = 0.3          # Proportional gain for aligning nut with peg
    YAW_ALIGNED_THRESHOLD = 0.3  # Yaw must be within this (radians) to proceed from hover
    INSERTION_YAW_THRESHOLD = 0.1  # ~5.7 deg - moderate alignment requirement

    # Gripper states
    GRIPPER_OPEN = -1
    GRIPPER_CLOSE = 1

    # FIXED peg positions (discovered from MuJoCo simulation)
    PEG1_POS = np.array([0.23, 0.10, 0.85])   # Square peg
    PEG2_POS = np.array([0.23, -0.10, 0.85])  # Round peg

    def __init__(self, obs):
        """
        [Short Description]: Initialize policy with waypoints computed from nut positions and orientations.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Set up 19-phase state machine for dual nut assembly with handle offset from quaternion"

        Args:
            obs (dict): Initial observation containing nut positions, orientations, and robot state.

        Returns:
            None

        Notes:
            Peg positions are hardcoded (not in observations). Handle offsets computed via 2D trig.
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
        # Order: Round nut first (phases 0-7), then transition wait (phase 8), then square nut (phases 9-16)
        waypoints_round = self._build_waypoints(
            round_nut_pos, self.PEG2_POS, self.rd_handle_offset, self.HANDLE_OFFSET_ROUND
        )
        waypoints_square = self._build_waypoints(
            square_nut_pos, self.PEG1_POS, self.sq_handle_offset, self.HANDLE_OFFSET_SQUARE
        )
        # Transition waypoint (phase 8): hover above round peg after release
        transition_waypoint = [self.PEG2_POS + np.array([0, 0, self.PEG_HOVER_HEIGHT])]
        self.waypoints = waypoints_round + transition_waypoint + waypoints_square

        # State machine
        self.phase = 0
        self.wait_counter = 0

        # Track chosen flip direction for each nut (decided once, used consistently)
        self.sq_flip_180 = None  # Will be set on first hover
        self.rd_flip_180 = None

        # Track whether grasp target has been set (only set once per grasp phase)
        self.sq_grasp_target_set = False
        self.rd_grasp_target_set = False

        # Alignment target yaw for each nut (computed dynamically to minimize rotation)
        # Will be set in _update_waypoints_round/square() when grasp is confirmed
        self.rd_align_target_yaw = None
        self.sq_align_target_yaw = None

        # Initialize PID controller
        self.pid = PID(
            kp=self.KP,
            ki=self.KI,
            kd=self.KD,
            target=self.waypoints[0]
        )

    def _get_handle_offset_and_yaw(self, nut_quat, handle_distance):
        """
        Compute world-frame handle offset and yaw angle from quaternion.

        Args:
            nut_quat (array): Quaternion orientation [x, y, z, w].
            handle_distance (float): Distance from nut center to handle (meters).

        Returns:
            tuple: (offset_array [x, y, 0], theta in radians)
        """
        # Extract yaw using scipy - handles arbitrary rotations correctly
        # (The simplified formula 2*arctan2(z,w) only works for pure Z rotations)
        theta = R.from_quat(nut_quat).as_euler('xyz')[2]

        # 2D rotation of handle offset [handle_distance, 0] by angle theta
        offset_x = handle_distance * np.cos(theta)
        offset_y = handle_distance * np.sin(theta)
        offset_z = 0.0

        return np.array([offset_x, offset_y, offset_z]), theta

    def _get_min_rotation_target(self, current_yaw, is_square_nut=False):
        """
        Get target yaw that minimizes rotation while keeping gripper on robot's side.

        For insertion, gripper must be between robot and peg (cos(θ) < 0).
        Prefers 180° (approach from -X direction) for better robot reachability.
        Only uses 90°/-90° if rotation to 180° would exceed 90°.

        Args:
            current_yaw: Current nut yaw angle in radians.
            is_square_nut: True for square nut (4-fold symmetry), False for round.

        Returns:
            float: Target yaw requiring minimum rotation.
        """
        # Normalize to [-π, π]
        current_yaw = np.arctan2(np.sin(current_yaw), np.cos(current_yaw))

        if is_square_nut:
            # Square nut has 4-fold symmetry
            # Prefer 180° for best robot reachability (gripper approaches from -X)
            # Only use 90°/-90° if rotation to 180° would be > 90°
            diff_to_180 = np.pi - current_yaw
            diff_to_180 = np.arctan2(np.sin(diff_to_180), np.cos(diff_to_180))

            if abs(diff_to_180) <= np.pi / 2:  # <= 90° rotation needed
                return np.pi
            else:
                # Use 90° or -90° depending on which is closer
                diff_to_90 = np.pi / 2 - current_yaw
                diff_to_90 = np.arctan2(np.sin(diff_to_90), np.cos(diff_to_90))
                diff_to_neg90 = -np.pi / 2 - current_yaw
                diff_to_neg90 = np.arctan2(np.sin(diff_to_neg90), np.cos(diff_to_neg90))

                if abs(diff_to_90) < abs(diff_to_neg90):
                    return np.pi / 2
                else:
                    return -np.pi / 2
        else:
            # Round peg is rotationally symmetric
            # Prefer 180° for consistency, but any angle where cos(θ) < 0 works
            diff_to_180 = np.pi - current_yaw
            diff_to_180 = np.arctan2(np.sin(diff_to_180), np.cos(diff_to_180))

            if abs(diff_to_180) <= np.pi / 2:  # <= 90° rotation
                return np.pi
            elif np.cos(current_yaw) < 0:
                # Already in valid range and rotation to 180° is large
                return current_yaw
            else:
                # Snap to nearest edge
                if current_yaw >= 0:
                    return np.pi / 2
                else:
                    return -np.pi / 2

    def _build_waypoints(self, nut_pos, peg_pos, handle_offset, handle_distance):
        """
        [Short Description]: Build 9-waypoint sequence for one nut-to-peg assembly.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Design outside-in approach trajectory with approach_extra offset to avoid nut collision"

        Args:
            nut_pos (np.ndarray): 3D nut position [x, y, z].
            peg_pos (np.ndarray): 3D peg position [x, y, z].
            handle_offset (np.ndarray): Handle offset in world frame [x, y, 0].
            handle_distance (float): Distance from nut center to handle (meters).

        Returns:
            list: 9 waypoints for phases 0-8 or 10-18.

        Notes:
            APPROACH_EXTRA=0.04m beyond handle. Aligned offset assumes handle at -X after rotation.
        """
        hover_offset = np.array([0, 0, self.HOVER_HEIGHT])
        grasp_z_offset = np.array([0, 0, self.GRASP_OFFSET_Z])
        peg_hover_offset = np.array([0, 0, self.PEG_HOVER_HEIGHT])
        peg_insert_offset = np.array([0, 0, self.PEG_INSERT_HEIGHT])

        # Compute approach offset (radially outward from handle to avoid nut body)
        approach_direction = handle_offset / handle_distance  # unit vector
        approach_offset = approach_direction * self.APPROACH_EXTRA

        # Aligned handle offset: handle pointing toward robot (-X direction)
        aligned_handle_offset = np.array([-handle_distance, 0, 0])

        return [
            nut_pos + hover_offset + handle_offset + approach_offset,   # 0: HOVER_ABOVE (outside, high)
            nut_pos + grasp_z_offset + handle_offset + approach_offset, # 1: DESCEND_OUTSIDE (outside, low)
            nut_pos + grasp_z_offset + handle_offset,                   # 2: MOVE_TO_HANDLE (at handle, low)
            nut_pos + grasp_z_offset + handle_offset,                   # 3: GRASP_WAIT
            nut_pos + hover_offset + handle_offset,                     # 4: LIFT_NUT
            nut_pos + hover_offset + handle_offset,                     # 5: ROTATE_ALIGN (stay in place, rotate)
            peg_pos + peg_hover_offset + aligned_handle_offset,         # 6: MOVE_TO_PEG (straight line)
            peg_pos + peg_insert_offset + aligned_handle_offset,        # 7: INSERT_PEG
            peg_pos + peg_insert_offset + aligned_handle_offset,        # 8: RELEASE_WAIT
        ]

    def get_action(self, obs):
        """
        [Short Description]: Compute 7D action with PID position control, yaw rotation, and gripper.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Implement dynamic target tracking, yaw rotation using relative quaternion, and flip_180 logic"

        Args:
            obs (dict): Current observation with robot and nut states.

        Returns:
            np.ndarray: 7D action [dx, dy, dz, ax, ay, az, gripper].

        Notes:
            Dynamic tracking for approach phases (0-2, 10-12). Flip direction decided once per nut.
            Yaw control aligns gripper with handle during approach, rotates nut to -X during lift.
        """
        current_pos = np.array(obs['robot0_eef_pos'])

        # DYNAMIC TARGET UPDATE: For approach phases, track current nut position
        # and rotation because nut may have moved/settled since init
        # Order: Round nut first (phases 0-8), transition wait (phase 9), square nut (phases 10-18)
        # NOTE: Don't reset PID during WAIT phases (3, 13) - need stable hold position
        if self.phase in [0, 1, 2]:  # Round nut approach phases (NOT grasp wait)
            current_nut = np.array(obs['RoundNut_pos'])
            nut_quat = obs['RoundNut_quat']
            # Compute handle offset and target yaw based on CURRENT rotation
            self.rd_handle_offset, self.rd_target_yaw = self._get_handle_offset_and_yaw(
                nut_quat, self.HANDLE_OFFSET_ROUND
            )

            # Compute approach offset (radially outward)
            approach_direction = self.rd_handle_offset / self.HANDLE_OFFSET_ROUND  # unit vector
            approach_offset = approach_direction * self.APPROACH_EXTRA

            if self.phase == 0:  # HOVER_ABOVE - outside + high
                target = current_nut + np.array([0, 0, self.HOVER_HEIGHT]) + self.rd_handle_offset + approach_offset
                self.pid.reset(target=target)  # Reset for far tracking
            elif self.phase == 1:  # DESCEND_OUTSIDE - outside + low
                target = current_nut + np.array([0, 0, self.GRASP_OFFSET_Z]) + self.rd_handle_offset + approach_offset
                self.pid.reset(target=target)  # Reset for far tracking
            else:  # Phase 2: MOVE_TO_HANDLE - at handle + low
                target = current_nut + np.array([0, 0, self.GRASP_OFFSET_Z]) + self.rd_handle_offset
                # Update target WITHOUT resetting (allows PID to converge for final approach)
                self.pid.target = np.array(target)

        elif self.phase == 3:  # Round nut GRASP_WAIT - hold position for grasp
            # DON'T reset PID target - gripper should already be at handle from phase 2
            # Just update waypoints once for lift+peg phases
            if not self.rd_grasp_target_set:
                self._update_waypoints_round(obs)
                self.rd_grasp_target_set = True
            # PID continues with last target from phase 2 (allows stable grip)

        elif self.phase == 5:  # Round nut ROTATE_ALIGN - rotate handle toward robot
            # DON'T update PID target - gripper stays in place while nut rotates around it
            # The yaw control (applied later) will rotate the nut
            # Just keep the gripper at its current position
            pass

        elif self.phase == 9:  # TRANSITION_WAIT - pause between completing round nut and starting square nut
            # Keep gripper at current position while waiting
            pass  # PID continues with last target, wait handled in phase transition logic

        elif self.phase in [10, 11, 12]:  # Square nut approach phases (NOT grasp wait)
            current_nut = np.array(obs['SquareNut_pos'])
            nut_quat = obs['SquareNut_quat']
            # Compute handle offset and target yaw based on CURRENT rotation
            self.sq_handle_offset, self.sq_target_yaw = self._get_handle_offset_and_yaw(
                nut_quat, self.HANDLE_OFFSET_SQUARE
            )

            # Compute approach offset (radially outward)
            approach_direction = self.sq_handle_offset / self.HANDLE_OFFSET_SQUARE  # unit vector
            approach_offset = approach_direction * self.APPROACH_EXTRA

            if self.phase == 10:  # HOVER_ABOVE - outside + high
                target = current_nut + np.array([0, 0, self.HOVER_HEIGHT]) + self.sq_handle_offset + approach_offset
                self.pid.reset(target=target)  # Reset for far tracking
            elif self.phase == 11:  # DESCEND_OUTSIDE - outside + low
                target = current_nut + np.array([0, 0, self.GRASP_OFFSET_Z]) + self.sq_handle_offset + approach_offset
                self.pid.reset(target=target)  # Reset for far tracking
            else:  # Phase 12: MOVE_TO_HANDLE - at handle + low
                target = current_nut + np.array([0, 0, self.GRASP_OFFSET_Z]) + self.sq_handle_offset
                # Update target WITHOUT resetting (allows PID to converge for final approach)
                self.pid.target = np.array(target)

        elif self.phase == 13:  # Square nut GRASP_WAIT - hold position for grasp
            # DON'T reset PID target - gripper should already be at handle from phase 12
            # Just update waypoints once for lift+peg phases
            if not self.sq_grasp_target_set:
                self._update_waypoints_square(obs)
                self.sq_grasp_target_set = True
            # PID continues with last target from phase 12 (allows stable grip)

        elif self.phase == 15:  # Square nut ROTATE_ALIGN - rotate handle toward robot
            # DON'T update PID target - gripper stays in place while nut rotates around it
            # The yaw control (applied later) will rotate the nut
            # Just keep the gripper at its current position
            pass

        # Phases 4, 6-8 and 14, 16-18: Use FIXED waypoints (nut moves with gripper)

        control = self.pid.update(current_pos, self.DT)
        error = self.pid.get_error()

        # Get threshold based on phase
        threshold = self._get_threshold()

        # Phase transition logic
        # FIX OPTION C: Removed yaw alignment requirement from hover phases (0, 10)
        # The gripper will still rotate during hover, but won't block if not aligned
        if error < threshold:
            if self._is_wait_phase():
                self.wait_counter += 1
                # Use different wait time for different wait phases
                if self.phase == 9:  # Transition wait
                    wait_threshold = self.TRANSITION_WAIT_STEPS
                    if self.wait_counter >= wait_threshold:
                        self._advance_phase()
                elif self.phase in [5, 15]:  # Rotate align phases - check ACTUAL yaw alignment
                    # Get current nut yaw and compute error
                    if self.phase == 5:
                        nut_quat = obs['RoundNut_quat']
                        target_yaw = self.rd_align_target_yaw if self.rd_align_target_yaw is not None else self.ALIGNED_HANDLE_YAW
                    else:
                        nut_quat = obs['SquareNut_quat']
                        target_yaw = self.sq_align_target_yaw if self.sq_align_target_yaw is not None else self.ALIGNED_HANDLE_YAW
                    nut_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
                    current_yaw_error = abs(target_yaw - nut_yaw)
                    current_yaw_error = abs(np.arctan2(np.sin(current_yaw_error), np.cos(current_yaw_error)))

                    # Advance only if: min steps elapsed AND yaw aligned OR max steps reached
                    if (self.wait_counter >= self.ROTATE_ALIGN_MIN_STEPS and
                        current_yaw_error < self.YAW_ALIGN_THRESHOLD):
                        self._advance_phase()
                    elif self.wait_counter >= self.ROTATE_ALIGN_STEPS:
                        # Max steps reached - advance anyway to avoid getting stuck
                        self._advance_phase()
                else:  # Gripper phases
                    wait_threshold = self.WAIT_STEPS
                    if self.wait_counter >= wait_threshold:
                        self._advance_phase()
            else:
                self._advance_phase()

        # Build action
        action = np.zeros(7)
        action[0:3] = control          # Position control from PID
        action[6] = self._get_gripper_state()

        # Yaw rotation control: align gripper with handle direction
        # CRITICAL: Must rotate during ALL approach phases (0, 1, 2) not just (0, 1)
        # Use simple 360° normalization with high gain for consistent convergence
        # NEGATIVE sign: if error is positive, rotate negative to reduce it
        # Order: Round nut first (phases 0-2), square nut (phases 10-12)
        if self.phase in [0, 1, 2]:  # Round nut approach phases - align with handle
            rel_quat = obs.get('RoundNut_to_robot0_eef_quat', None)
            if rel_quat is not None and np.sum(np.abs(rel_quat)) > 0:
                raw_yaw = R.from_quat(rel_quat).as_euler('xyz')[2]
                # Normalize to [-π, π]
                yaw_error = np.arctan2(np.sin(raw_yaw), np.cos(raw_yaw))

                # Decide flip direction on first hover (Phase 0 entry)
                if self.rd_flip_180 is None:
                    # If yaw_error > 90°, it's easier to go the other way (add 180°)
                    self.rd_flip_180 = abs(yaw_error) > np.pi / 2

                # Apply flip: target the opposite alignment if needed
                if self.rd_flip_180:
                    # Flip by 180°: if error was +170°, make it -10° (closer via opposite rotation)
                    yaw_error = np.arctan2(np.sin(yaw_error + np.pi), np.cos(yaw_error + np.pi))

                action[5] = -self.YAW_GAIN * yaw_error  # Negative feedback
        elif self.phase in [10, 11, 12]:  # Square nut approach phases - align with handle
            rel_quat = obs.get('SquareNut_to_robot0_eef_quat', None)
            if rel_quat is not None and np.sum(np.abs(rel_quat)) > 0:
                raw_yaw = R.from_quat(rel_quat).as_euler('xyz')[2]
                # Normalize to [-π, π]
                yaw_error = np.arctan2(np.sin(raw_yaw), np.cos(raw_yaw))

                # Decide flip direction on first hover (Phase 10 entry)
                if self.sq_flip_180 is None:
                    self.sq_flip_180 = abs(yaw_error) > np.pi / 2

                # Apply flip if needed
                if self.sq_flip_180:
                    yaw_error = np.arctan2(np.sin(yaw_error + np.pi), np.cos(yaw_error + np.pi))

                action[5] = -self.YAW_GAIN * yaw_error  # Negative feedback
        elif self.phase == 4:  # Round nut LIFT - start rotating toward aligned yaw early
            nut_quat = obs['RoundNut_quat']
            nut_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
            # Use dynamically computed target (minimizes rotation)
            target_yaw = self.rd_align_target_yaw if self.rd_align_target_yaw is not None else self.ALIGNED_HANDLE_YAW
            yaw_error = target_yaw - nut_yaw
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            action[5] = self.YAW_GAIN * yaw_error
        elif self.phase == 5:  # Round nut ROTATE_ALIGN - continue rotating to target yaw
            nut_quat = obs['RoundNut_quat']
            nut_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
            # Use dynamically computed target (minimizes rotation)
            target_yaw = self.rd_align_target_yaw if self.rd_align_target_yaw is not None else self.ALIGNED_HANDLE_YAW
            yaw_error = target_yaw - nut_yaw
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # Normalize
            action[5] = self.YAW_GAIN * yaw_error
        elif self.phase == 14:  # Square nut LIFT - start rotating toward aligned yaw early
            nut_quat = obs['SquareNut_quat']
            nut_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
            # Use dynamically computed target (minimizes rotation for 4-fold symmetry)
            target_yaw = self.sq_align_target_yaw if self.sq_align_target_yaw is not None else self.ALIGNED_HANDLE_YAW
            yaw_error = target_yaw - nut_yaw
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            action[5] = self.YAW_GAIN * yaw_error
        elif self.phase == 15:  # Square nut ROTATE_ALIGN - continue rotating to target yaw
            nut_quat = obs['SquareNut_quat']
            nut_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
            # Use dynamically computed target (minimizes rotation for 4-fold symmetry)
            target_yaw = self.sq_align_target_yaw if self.sq_align_target_yaw is not None else self.ALIGNED_HANDLE_YAW
            yaw_error = target_yaw - nut_yaw
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # Normalize
            action[5] = self.YAW_GAIN * yaw_error
        elif self.phase in [6, 7]:  # Round nut insertion - stronger yaw correction
            nut_quat = obs['RoundNut_quat']
            nut_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
            target_yaw = self.rd_align_target_yaw if self.rd_align_target_yaw is not None else self.ALIGNED_HANDLE_YAW
            yaw_error = target_yaw - nut_yaw
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            # Use higher gain during insertion to fight contact forces
            action[5] = self.INSERTION_YAW_GAIN * yaw_error
        elif self.phase in [16, 17]:  # Square nut insertion - stronger yaw correction
            nut_quat = obs['SquareNut_quat']
            nut_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
            target_yaw = self.sq_align_target_yaw if self.sq_align_target_yaw is not None else self.ALIGNED_HANDLE_YAW
            yaw_error = target_yaw - nut_yaw
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            # Use higher gain during insertion to fight contact forces
            action[5] = self.INSERTION_YAW_GAIN * yaw_error

        return action

    def _update_waypoints_round(self, obs):
        """
        [Short Description]: Update round nut waypoints (4-8) with actual grasp position.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Update lift and peg waypoints once grasp is secured to use actual handle offset"

        Args:
            obs (dict): Current observation with RoundNut_pos.

        Returns:
            None (modifies self.waypoints in place).

        Notes:
            Called once in phase 3. Uses _get_min_rotation_target() to minimize rotation.
        """
        current_nut = np.array(obs['RoundNut_pos'])
        nut_quat = obs['RoundNut_quat']
        hover = np.array([0, 0, self.HOVER_HEIGHT])
        peg_hover = np.array([0, 0, self.PEG_HOVER_HEIGHT])
        peg_insert = np.array([0, 0, self.PEG_INSERT_HEIGHT])

        # Compute optimal target yaw that minimizes rotation
        current_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
        self.rd_align_target_yaw = self._get_min_rotation_target(current_yaw, is_square_nut=False)

        # Aligned handle offset: computed from target yaw (not always -X)
        aligned_handle_offset = np.array([
            self.HANDLE_OFFSET_ROUND * np.cos(self.rd_align_target_yaw),
            self.HANDLE_OFFSET_ROUND * np.sin(self.rd_align_target_yaw),
            0
        ])

        # Waypoint 4: LIFT - use current nut pos + handle offset
        self.waypoints[4] = current_nut + hover + self.rd_handle_offset
        # Waypoint 5: ROTATE_ALIGN - same position as lift (rotate in place)
        self.waypoints[5] = current_nut + hover + self.rd_handle_offset
        # Waypoints 6, 7, 8: PEG phases - use aligned handle offset for straight line motion
        self.waypoints[6] = self.PEG2_POS + peg_hover + aligned_handle_offset
        self.waypoints[7] = self.PEG2_POS + peg_insert + aligned_handle_offset
        self.waypoints[8] = self.PEG2_POS + peg_insert + aligned_handle_offset

    def _update_waypoints_square(self, obs):
        """
        [Short Description]: Update square nut waypoints (14-18) with actual grasp position.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Update lift and peg waypoints once grasp is secured to use actual handle offset"

        Args:
            obs (dict): Current observation with SquareNut_pos.

        Returns:
            None (modifies self.waypoints in place).

        Notes:
            Called once in phase 13. Uses _get_min_rotation_target() to minimize rotation.
        """
        current_nut = np.array(obs['SquareNut_pos'])
        nut_quat = obs['SquareNut_quat']
        hover = np.array([0, 0, self.HOVER_HEIGHT])
        peg_hover = np.array([0, 0, self.PEG_HOVER_HEIGHT])
        peg_insert = np.array([0, 0, self.PEG_INSERT_HEIGHT])

        # Compute optimal target yaw that minimizes rotation (square nut has 4-fold symmetry)
        current_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
        self.sq_align_target_yaw = self._get_min_rotation_target(current_yaw, is_square_nut=True)

        # Aligned handle offset: computed from target yaw (not always -X)
        aligned_handle_offset = np.array([
            self.HANDLE_OFFSET_SQUARE * np.cos(self.sq_align_target_yaw),
            self.HANDLE_OFFSET_SQUARE * np.sin(self.sq_align_target_yaw),
            0
        ])

        # Waypoint 14: LIFT - use current nut pos + handle offset
        self.waypoints[14] = current_nut + hover + self.sq_handle_offset
        # Waypoint 15: ROTATE_ALIGN - same position as lift (rotate in place)
        self.waypoints[15] = current_nut + hover + self.sq_handle_offset
        # Waypoints 16, 17, 18: PEG phases - use aligned handle offset for straight line motion
        self.waypoints[16] = self.PEG1_POS + peg_hover + aligned_handle_offset
        self.waypoints[17] = self.PEG1_POS + peg_insert + aligned_handle_offset
        self.waypoints[18] = self.PEG1_POS + peg_insert + aligned_handle_offset

    def _advance_phase(self):
        """
        [Short Description]: Advance to next phase with selective PID reset.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Implement phase transition - don't reset PID for grasp/rotate phases"

        Args:
            None

        Returns:
            None

        Notes:
            PID NOT reset for phases 3, 5, 9, 13, 15 to maintain stable position.
        """
        if self.phase < len(self.waypoints) - 1:
            self.phase += 1
            # DON'T reset PID for grasp phases (3, 13) - stay at current position
            # The gripper is already at the handle from the previous tracking phase
            # Also don't reset for transition wait (9) or rotate_align (5, 15) - set dynamically
            if self.phase not in [3, 5, 9, 13, 15]:
                self.pid.reset(target=self.waypoints[self.phase])
            self.wait_counter = 0

    def _is_wait_phase(self):
        """
        [Short Description]: Check if current phase requires time-based waiting.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Identify phases that need wait counter for gripper actuation or rotation"

        Args:
            None

        Returns:
            bool: True if phase requires waiting, False otherwise.

        Notes:
            Wait phases: 3, 13 (grasp), 5, 15 (rotate), 8, 18 (release), 9 (transition).
        """
        return self.phase in [3, 5, 8, 9, 13, 15, 18]

    def _get_threshold(self):
        """
        [Short Description]: Get adaptive position error threshold based on current phase.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Design adaptive thresholds - tight for grasp/peg, loose for movement"

        Args:
            None

        Returns:
            float: Position error threshold in meters.

        Notes:
            TIGHT (2.5cm) for grasp, PEG (1cm) for insertion, LOOSE (6cm) for movement.
        """
        if self.phase in [3, 13]:  # Grasp wait phases - need precision
            return self.THRESHOLD_TIGHT
        elif self.phase in [2, 12]:  # Move-to-handle - slightly relaxed
            return 0.03  # 3cm
        elif self.phase in [6, 16]:  # Move-to-peg - need precise alignment
            return self.THRESHOLD_PEG  # 1.5cm (must be < clearance of 11mm for round)
        return self.THRESHOLD_LOOSE  # All other phases - movement tolerance OK

    def _get_gripper_state(self):
        """
        [Short Description]: Determine gripper command based on current phase.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Map 19 phases to gripper states for sequential round then square nut assembly"

        Args:
            None

        Returns:
            float: -1 (open) or +1 (close).

        Notes:
            OPEN: 0-2, 8-12, 18. CLOSE: 3-7, 13-17.
        """
        if self.phase < 3:
            return self.GRIPPER_OPEN
        elif self.phase < 8:
            return self.GRIPPER_CLOSE
        elif self.phase < 13:
            return self.GRIPPER_OPEN
        elif self.phase < 18:
            return self.GRIPPER_CLOSE
        else:
            return self.GRIPPER_OPEN


class DoorPolicy(object):
    """
    [Short Description]: 6-phase state machine policy for door opening with handle rotation.
    [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
    "Implement DoorPolicy with gripper orientation, handle rotation, and pull tracking"

    Phase Sequence:
        0: ORIENT       - Rotate gripper 90 degrees (roll) so fingers can wrap handle
        1: APPROACH     - Move to offset position near handle (gripper open)
        2: REACH        - Move to handle grip position (gripper open)
        3: GRIP         - Close gripper and wait for secure grip
        4: ROTATE       - Rotate handle to unlatch door
        5: PULL         - Pull door open while tracking handle

    Notes:
        Gripper rotation via action[3], handle rotation via action[4], pull in +Y direction.
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
        [Short Description]: Initialize policy with handle and end-effector positions.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Set up 6-phase state machine for door opening with initial positions"

        Args:
            obs (dict): Initial observation with handle_pos, robot0_eef_pos.

        Returns:
            None

        Notes:
            Pull direction computed after ROTATE phase. Offsets tuned experimentally.
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
        [Short Description]: Compute 7D action with position control, rotation, and gripper.
        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with the prompt:
        "Implement get_action with gripper rotation, handle rotation, and dynamic tracking"

        Args:
            obs (dict): Current observation with robot0_eef_pos, handle_pos, handle_qpos.

        Returns:
            np.ndarray: 7D action [dx, dy, dz, ax, ay, az, gripper].

        Notes:
            Handle tracked dynamically during phases 3-5. HANDLE_THRESHOLD triggers PULL.
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