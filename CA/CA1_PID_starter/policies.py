import numpy as np
from pid import PID

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
    Goal: Fit square nut on square peg and round nut on round peg.
    """
    def __init__(self, obs):
        """
        Args:
            obs (dict): Includes 'SquareNut_pos', 'SquarePeg_pos', etc.
        """
        # Initialize PID controller and assembly sequences here
        pass

    def get_action(self, obs):
        """
        Returns:
            np.ndarray: 7D action (Delta-X, Delta-Y, Delta-Z, Axis-Angle [3], Gripper).
        """
        pass


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
    ROTATE_STEPS = 60

    GRIPPER_OPEN = -1
    GRIPPER_CLOSE = 1

    # Offsets discovered through experimentation
    APPROACH_OFFSET = np.array([0.12, 0.0, 0.07])
    GRIP_OFFSET = np.array([0.0, -0.02, -0.02])  # Y=-0.02 to center grip on handle
    PULL_OFFSET = 0.10  # Reduced from 0.25 for gentler pull

    # Rotation parameters
    # X-axis rotation (action[3]) - rotates gripper 90 degrees
    ORIENT_ROTATION = np.array([-0.3, 0.0, 0.0])  # Rotate around X-axis
    HANDLE_ROTATION = np.array([0.0, -0.1, 0.0])  # Y command to turn handle (pitch)

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

        # Pull direction will be calculated AFTER rotation phase
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

            self.step_counter += 1
            if self.step_counter >= self.ROTATE_STEPS:
                # Calculate pull direction NOW (after rotation)
                self.pull_direction = self.initial_eef_pos - handle_pos
                self.pull_direction[2] = 0  # Keep horizontal
                magnitude = np.linalg.norm(self.pull_direction)
                if magnitude > 0.01:
                    self.pull_direction = self.pull_direction / magnitude

                self.phase = 5
                self.step_counter = 0
            return action

        elif self.phase == 5:  # PULL
            # Dynamic target: follow handle with offset in pull direction
            target = handle_pos + self.GRIP_OFFSET + self.pull_direction * self.PULL_OFFSET
            self.pid.reset(target=target)
            control = self.pid.update(current_pos, self.DT)

            action[0:3] = control
            action[6] = self.GRIPPER_CLOSE
            return action

        return action