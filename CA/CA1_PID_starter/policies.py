import numpy as np
from pid import PID

class StackPolicy(object):
    """
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
    """

    KP = 3.0   # Proportional: higher = faster but may overshoot
    KI = 0.0   # Integral: helps eliminate steady-state error (start with 0)
    KD = 0.2   # Derivative: damping to reduce oscillation

    HOVER_HEIGHT = 0.08    # How high above objects to hover (safe clearance)
    CUBE_HEIGHT = 0.015     # Approximate height of cube (for grasp/stack offset)

    THRESHOLD = 0.01       # Position error threshold to advance phase (1cm)
    WAIT_STEPS = 30        # How many timesteps to wait for gripper actuation
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
            cube_a_pos + hover_offset,
            cube_a_pos + grasp_offset,
            cube_a_pos + grasp_offset,
            cube_a_pos + hover_offset,
            cube_b_pos + hover_offset + np.array([0, 0, self.CUBE_HEIGHT]),
            cube_b_pos + stack_offset,
            cube_b_pos + stack_offset,
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

        if error < self.THRESHOLD:
            if self.phase in [2, 6]:  # Grasp or Release phases
                # These phases require WAITING for gripper to actuate
                self.wait_counter += 1
                if self.wait_counter >= self.WAIT_STEPS:
                    self._advance_phase()
            else:
                self._advance_phase()
        action = np.zeros(7)

        action[0] = control[0]  # dx - move in X direction
        action[1] = control[1]  
        action[2] = control[2]  

        # Rotation control (not used - keep gripper orientation fixed)
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

    def _get_gripper_state(self):
        """
        Determine gripper state based on current phase.

        Gripper Logic:
        - Phases 0-1: OPEN (approaching cube A)
        - Phases 2-5: CLOSED (grasping and carrying cube A)
        - Phase 6:    OPEN (releasing cube A onto cube B)

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
    """
    def __init__(self, obs):
        """
        Args:
            obs (dict): Includes 'door_handle_pos'.
        """
        # Initialize PID controller and door opening trajectory waypoints
        pass

    def get_action(self, obs):
        """
        Returns:
            np.ndarray: 7D action (Delta-X, Delta-Y, Delta-Z, Axis-Angle [3], Gripper).
        """
        pass