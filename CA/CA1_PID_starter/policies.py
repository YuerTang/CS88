import numpy as np
from pid import PID
from scipy.spatial.transform import Rotation as R

class StackPolicy(object):
    """
    Policy for the Block Stacking task using a state machine approach.

    [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompts:
    "Implement StackPolicy state machine for pick and place cube stacking task"
    "Debug gripper release issue - robot stays at same position after releasing block"
    "Add adaptive thresholds and retract phase to achieve 86% success rate"

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

    Notes:
        Adaptive thresholds (THRESHOLD_TIGHT=0.015 for phases 0-4, THRESHOLD_LOOSE=0.03
        for phases 5-7) were added after debugging stuck phase 5 issue where min
        achievable error was ~0.024m due to block collision during stacking.
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
        Initialize the StackPolicy with waypoints computed from cube positions.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompt:
        "Set up waypoints for 8-phase state machine based on cube A and B positions"

        Args:
            obs (dict): Initial observation containing object positions.
                - obs['cubeA_pos']: [x, y, z] position of Cube A (to pick up)
                - obs['cubeB_pos']: [x, y, z] position of Cube B (to stack on)
                - obs['robot0_eef_pos']: [x, y, z] robot end-effector position

        Returns:
            None

        Notes:
            Waypoints are computed relative to cube positions at init time.
            RETRACT phase (waypoint 7) was added to physically separate gripper
            from block after release, fixing the "gripper still touching block" issue.
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
        Compute the 7D action vector for the current timestep.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompt:
        "Implement get_action with PID control, phase transitions, and gripper logic"

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

        Notes:
            Phase transition uses adaptive thresholds via _get_threshold().
            Wait phases (2, 6) require WAIT_STEPS timesteps before advancing.
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
        Advance to the next phase in the state machine.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompt:
        "Implement phase transition with PID reset for new waypoint target"

        Args:
            None

        Returns:
            None

        Notes:
            PID is reset (not just target updated) to clear integral and derivative
            error history, ensuring clean transitions between phases.
        """
        if self.phase < len(self.waypoints) - 1:
            self.phase += 1
            self.pid.reset(target=self.waypoints[self.phase])
            self.wait_counter = 0

    def _get_threshold(self):
        """
        Get adaptive position error threshold based on current phase.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompt:
        "Add adaptive thresholds - tight for grasping, loose for stacking phases"

        Args:
            None

        Returns:
            float: Position error threshold in meters.
                - THRESHOLD_TIGHT (0.015m) for phases 0-4 (free-space motion)
                - THRESHOLD_LOOSE (0.03m) for phases 5-7 (stacking with constraints)

        Notes:
            Adaptive thresholds were added after discovering that phase 5 was getting
            stuck because min achievable error during stacking is ~0.024m due to
            physical collision between blocks.
        """
        if self.phase <= 4:
            return self.THRESHOLD_TIGHT
        else:
            return self.THRESHOLD_LOOSE

    def _get_gripper_state(self):
        """
        Determine gripper command based on current phase.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompt:
        "Map phases to gripper states for pick and place sequence"

        Args:
            None

        Returns:
            float: Gripper command value.
                - GRIPPER_OPEN (-1) for phases 0-1 (approaching) and 6-7 (releasing)
                - GRIPPER_CLOSE (+1) for phases 2-5 (grasping and carrying)

        Notes:
            Phases 6-7 both return OPEN to ensure gripper releases before retract.
        """
        if self.phase < 2:

            return self.GRIPPER_OPEN
        elif self.phase < 6:
            return self.GRIPPER_CLOSE
        else:
            return self.GRIPPER_OPEN

class NutAssemblyPolicy(object):
    """
    Policy for the Nut Assembly task using a 19-phase state machine.

    [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompts:
    "Implement NutAssemblyPolicy to fit square nut on square peg and round nut on round peg"
    "Design outside-in approach strategy to avoid collision with nut body"
    "Implement quaternion to yaw extraction for handle offset calculation"
    "Add yaw rotation control to align gripper with handle during approach"
    "Debug rotation direction - use relative quaternion for gripper-handle alignment"
    "Implement flip direction logic to choose shortest rotation path"

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
        Handle offset calculation uses 2D trigonometry since nuts only rotate around
        Z-axis. The yaw rotation control uses relative quaternion during approach
        phases and absolute nut yaw during rotate-align phases.
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
    ALIGN_GAIN = 0.3          # Proportional gain for aligning nut with peg
    YAW_ALIGNED_THRESHOLD = 0.3  # Yaw must be within this (radians) to proceed from hover

    # Gripper states
    GRIPPER_OPEN = -1
    GRIPPER_CLOSE = 1

    # FIXED peg positions (discovered from MuJoCo simulation)
    PEG1_POS = np.array([0.23, 0.10, 0.85])   # Square peg
    PEG2_POS = np.array([0.23, -0.10, 0.85])  # Round peg

    def __init__(self, obs):
        """
        Initialize NutAssemblyPolicy with waypoints computed from nut positions.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompts:
        "Set up 19-phase state machine for dual nut assembly"
        "Compute handle offset from quaternion orientation using 2D trigonometry"
        "Build waypoints for outside-in approach with approach_extra offset"

        Args:
            obs (dict): Initial observation containing:
                - obs['SquareNut_pos']: [x, y, z] position of square nut
                - obs['SquareNut_quat']: [x, y, z, w] quaternion orientation
                - obs['RoundNut_pos']: [x, y, z] position of round nut
                - obs['RoundNut_quat']: [x, y, z, w] quaternion orientation
                - obs['robot0_eef_pos']: [x, y, z] robot end-effector position

        Returns:
            None

        Notes:
            Peg positions (PEG1_POS, PEG2_POS) are hardcoded as they are fixed
            objects not exposed in the observation space. Discovered by querying
            MuJoCo simulation directly.
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

        # Initialize PID controller
        self.pid = PID(
            kp=self.KP,
            ki=self.KI,
            kd=self.KD,
            target=self.waypoints[0]
        )

    def _get_handle_offset_and_yaw(self, nut_quat, handle_distance):
        """
        Compute world-frame offset to nut handle and yaw angle from quaternion.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompts:
        "Extract yaw angle from quaternion for Z-axis rotation"
        "Compute 2D rotation of handle offset from local to world frame"

        Uses 2D trigonometry since the nut only rotates around Z-axis (yaw).
        The handle is at local position (handle_distance, 0, 0) in nut's frame.

        Args:
            nut_quat (array): Quaternion orientation of the nut [x, y, z, w].
            handle_distance (float): Distance from nut center to handle (meters).

        Returns:
            tuple: (offset_array, theta)
                - offset_array (np.ndarray): 3D offset in world frame [x, y, 0]
                - theta (float): Yaw angle of handle (radians)

        Notes:
            MuJoCo/robosuite uses [x, y, z, w] quaternion format.
            For Z-axis rotation: w = cos(θ/2), z = sin(θ/2), so θ = 2*atan2(z, w).
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
        Build 9-waypoint sequence for one nut-to-peg assembly operation.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompts:
        "Design outside-in approach trajectory to avoid nut body collision"
        "Compute approach offset as handle_offset + extra in radial direction"

        Args:
            nut_pos (np.ndarray): 3D position of the nut [x, y, z].
            peg_pos (np.ndarray): 3D position of the target peg [x, y, z].
            handle_offset (np.ndarray): 3D offset to handle [x, y, 0] in world frame.
            handle_distance (float): Distance from nut center to handle (meters).

        Returns:
            list: List of 9 waypoints (np.ndarray) for phases 0-8 or 10-18.

        Notes:
            Approach direction computed as unit vector of handle_offset.
            APPROACH_EXTRA=0.04m added beyond handle to ensure gripper clears nut body.
            Aligned handle offset assumes handle pointing toward robot (-X direction).
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
        Compute 7D action vector with position control, yaw rotation, and gripper.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompts:
        "Implement dynamic target tracking for approach phases"
        "Add yaw rotation control using relative quaternion for gripper-handle alignment"
        "Implement flip_180 logic to choose shortest rotation path"
        "Add yaw control during lift and rotate-align phases to point handle toward robot"
        "Continue yaw correction during insertion phases to maintain alignment"

        Args:
            obs (dict): Current observation containing:
                - obs['robot0_eef_pos']: Current end-effector position
                - obs['RoundNut_pos'], obs['RoundNut_quat']: Round nut state
                - obs['SquareNut_pos'], obs['SquareNut_quat']: Square nut state
                - obs['RoundNut_to_robot0_eef_quat']: Relative quaternion (optional)
                - obs['SquareNut_to_robot0_eef_quat']: Relative quaternion (optional)

        Returns:
            np.ndarray: 7D action vector [dx, dy, dz, ax, ay, az, gripper]
                - [0:3]: Position delta from PID controller
                - [3:5]: Zero (no roll/pitch control)
                - [5]: Yaw rotation command for handle alignment
                - [6]: Gripper command (-1=open, +1=close)

        Notes:
            Approach phases (0-2, 10-12) use dynamic target tracking with PID reset.
            Grasp phases (3, 13) hold position without PID reset.
            Rotate-align phases (5, 15) use absolute nut yaw to rotate handle to -X.
            Flip direction is decided once per nut to ensure consistent rotation.
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
                    else:
                        nut_quat = obs['SquareNut_quat']
                    nut_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
                    current_yaw_error = abs(self.ALIGNED_HANDLE_YAW - nut_yaw)
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
            yaw_error = self.ALIGNED_HANDLE_YAW - nut_yaw
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            action[5] = self.YAW_GAIN * yaw_error
        elif self.phase == 5:  # Round nut ROTATE_ALIGN - continue rotating to target yaw
            nut_quat = obs['RoundNut_quat']
            nut_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
            # Target yaw = π (handle pointing toward robot along -X)
            yaw_error = self.ALIGNED_HANDLE_YAW - nut_yaw
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # Normalize
            action[5] = self.YAW_GAIN * yaw_error
        elif self.phase == 14:  # Square nut LIFT - start rotating toward aligned yaw early
            nut_quat = obs['SquareNut_quat']
            nut_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
            yaw_error = self.ALIGNED_HANDLE_YAW - nut_yaw
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            action[5] = self.YAW_GAIN * yaw_error
        elif self.phase == 15:  # Square nut ROTATE_ALIGN - continue rotating to target yaw
            nut_quat = obs['SquareNut_quat']
            nut_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
            # Target yaw = π (handle pointing toward robot along -X)
            yaw_error = self.ALIGNED_HANDLE_YAW - nut_yaw
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # Normalize
            action[5] = self.YAW_GAIN * yaw_error
        elif self.phase in [6, 7]:  # Round nut insertion - continue yaw correction
            nut_quat = obs['RoundNut_quat']
            nut_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
            yaw_error = self.ALIGNED_HANDLE_YAW - nut_yaw
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            action[5] = self.YAW_GAIN * yaw_error
        elif self.phase in [16, 17]:  # Square nut insertion - continue yaw correction
            nut_quat = obs['SquareNut_quat']
            nut_yaw = R.from_quat(nut_quat).as_euler('xyz')[2]
            yaw_error = self.ALIGNED_HANDLE_YAW - nut_yaw
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            action[5] = self.YAW_GAIN * yaw_error

        return action

    def _update_waypoints_round(self, obs):
        """
        Update round nut waypoints with actual grasp position at grasp time.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompt:
        "Update lift and peg waypoints once grasp is secured to use actual handle offset"

        Args:
            obs (dict): Current observation with RoundNut_pos.

        Returns:
            None (modifies self.waypoints in place)

        Notes:
            Called once in phase 3 when rd_grasp_target_set is False.
            Waypoints 4-8 are updated with current nut position and handle offset.
            Aligned_handle_offset assumes handle will point to -X after rotation.
        """
        current_nut = np.array(obs['RoundNut_pos'])
        hover = np.array([0, 0, self.HOVER_HEIGHT])
        peg_hover = np.array([0, 0, self.PEG_HOVER_HEIGHT])
        peg_insert = np.array([0, 0, self.PEG_INSERT_HEIGHT])

        # Aligned handle offset: after ROTATE_ALIGN, handle points toward robot (-X)
        aligned_handle_offset = np.array([-self.HANDLE_OFFSET_ROUND, 0, 0])

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
        Update square nut waypoints with actual grasp position at grasp time.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompt:
        "Update lift and peg waypoints once grasp is secured to use actual handle offset"

        Args:
            obs (dict): Current observation with SquareNut_pos.

        Returns:
            None (modifies self.waypoints in place)

        Notes:
            Called once in phase 13 when sq_grasp_target_set is False.
            Waypoints 14-18 are updated with current nut position and handle offset.
            Aligned_handle_offset assumes handle will point to -X after rotation.
        """
        current_nut = np.array(obs['SquareNut_pos'])
        hover = np.array([0, 0, self.HOVER_HEIGHT])
        peg_hover = np.array([0, 0, self.PEG_HOVER_HEIGHT])
        peg_insert = np.array([0, 0, self.PEG_INSERT_HEIGHT])

        # Aligned handle offset: after ROTATE_ALIGN, handle points toward robot (-X)
        aligned_handle_offset = np.array([-self.HANDLE_OFFSET_SQUARE, 0, 0])

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
        Advance to the next phase with selective PID reset.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompt:
        "Implement phase transition - don't reset PID for grasp/rotate phases"

        Args:
            None

        Returns:
            None

        Notes:
            PID is NOT reset for phases 3, 5, 9, 13, 15 to maintain stable position
            during gripper actuation or rotation. These phases set targets dynamically.
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
        Check if current phase requires time-based waiting before transition.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompt:
        "Identify phases that need wait counter for gripper actuation or rotation"

        Args:
            None

        Returns:
            bool: True if current phase requires waiting, False otherwise.

        Notes:
            Wait phases: 3, 13 (grasp), 5, 15 (rotate), 8, 18 (release), 9 (transition).
        """
        return self.phase in [3, 5, 8, 9, 13, 15, 18]

    def _get_threshold(self):
        """
        Get adaptive position error threshold based on current phase.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompt:
        "Design adaptive thresholds - tight for grasp/peg, loose for movement"

        Args:
            None

        Returns:
            float: Position error threshold in meters.
                - THRESHOLD_TIGHT (0.025m) for grasp phases (3, 13)
                - THRESHOLD_PEG (0.01m) for peg alignment phases (6, 16)
                - 0.03m for move-to-handle phases (2, 12)
                - THRESHOLD_LOOSE (0.06m) for all other movement phases

        Notes:
            Peg threshold must be smaller than clearance between nut hole and peg
            (11mm for round nut) to ensure successful insertion.
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
        Determine gripper command based on current phase for dual nut assembly.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompt:
        "Map 19 phases to gripper states for sequential round then square nut assembly"

        Args:
            None

        Returns:
            float: Gripper command value.
                - GRIPPER_OPEN (-1): Phases 0-2 (round approach), 8-12 (release+square approach), 18
                - GRIPPER_CLOSE (+1): Phases 3-7 (round grasp+place), 13-17 (square grasp+place)

        Notes:
            Gripper opens during transition (phase 9) and square approach (10-12)
            to prepare for second nut grasp.
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
    Policy for the Door Opening task using a 6-phase state machine.

    [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompts:
    "Implement DoorPolicy for door opening task with handle rotation and pull"
    "Fix gripper orientation - rotate 90 degrees so fingers can wrap handle"
    "Debug door not opening - need to rotate handle to unlatch before pulling"
    "Determine pull direction from hinge geometry - pull in +Y toward robot"
    "Track handle position dynamically during pull phase as door swings"

    Phase Sequence:
        0: ORIENT       - Rotate gripper 90 degrees (roll) so fingers can wrap handle
        1: APPROACH     - Move to offset position near handle (gripper open)
        2: REACH        - Move to handle grip position (gripper open)
        3: GRIP         - Close gripper and wait for secure grip
        4: ROTATE       - Rotate handle to unlatch door
        5: PULL         - Pull door open while tracking handle

    Notes:
        Gripper rotation uses action[3] (X-axis) to pitch fingers down.
        Handle rotation uses action[4] (Y-axis) to turn the latch.
        Pull direction is +Y (toward robot) since hinge is on the left side.
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
        Initialize DoorPolicy with initial handle and end-effector positions.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompts:
        "Set up 6-phase state machine for door opening"
        "Store initial positions for ORIENT phase positioning"

        Args:
            obs (dict): Initial observation containing:
                - obs['handle_pos']: [x, y, z] position of door handle
                - obs['robot0_eef_pos']: [x, y, z] robot end-effector position

        Returns:
            None

        Notes:
            Pull direction is set to None initially and computed after ROTATE phase
            based on hinge geometry. APPROACH_OFFSET and GRIP_OFFSET were tuned
            through experimentation with the robosuite Door environment.
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
        Compute 7D action vector for door opening with position and rotation control.

        [AI Declaration]: Generated using Claude (claude-opus-4-5-20251101) with prompts:
        "Implement get_action with phase-specific control logic"
        "Add gripper rotation during ORIENT phase using action[3]"
        "Add handle rotation during ROTATE phase using action[4]"
        "Track handle_pos dynamically during PULL phase as door swings"
        "Add constant pull force in +Y direction while maintaining handle grip"

        Args:
            obs (dict): Current observation containing:
                - obs['robot0_eef_pos']: Current end-effector position
                - obs['handle_pos']: Current handle position (moves as door opens)
                - obs['handle_qpos']: Handle joint angle (for latch state)

        Returns:
            np.ndarray: 7D action vector [dx, dy, dz, ax, ay, az, gripper]
                - [0:3]: Position delta from PID controller (+ pull force in phase 5)
                - [3]: X-axis rotation for gripper orientation (phase 0)
                - [4]: Y-axis rotation for handle turning (phases 4-5)
                - [5]: Zero (no Z-axis rotation)
                - [6]: Gripper command (open for 0-2, closed for 3-5)

        Notes:
            Handle position is tracked dynamically during GRIP, ROTATE, and PULL phases
            to maintain grip as the door swings. HANDLE_THRESHOLD (1.4 rad) triggers
            transition from ROTATE to PULL phase.
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