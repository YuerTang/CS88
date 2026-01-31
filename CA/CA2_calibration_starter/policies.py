import numpy as np
from scipy.spatial.transform import Rotation as R
from robosuite.utils.transform_utils import get_orientation_error

"""
DO NOT MODIFY THIS FILE
"""

class PID:
    def __init__(self, kp, ki, kd, target):
        """
        Initialize a variable-dimension PID controller.

        Args:
            kp (float or list): Proportional gain(s) per axis (or scalar).
            ki (float or list): Integral gain(s) per axis (or scalar).
            kd (float or list): Derivative gain(s) per axis (or scalar).
            target (tuple or array): Target position of any dimension.
        """
        self.setpoint = np.array(target, dtype=np.float64)
        dim = self.setpoint.shape[0]

        self.kp = np.array(kp if hasattr(kp, '__len__') else [kp] * dim, dtype=np.float64)
        self.ki = np.array(ki if hasattr(ki, '__len__') else [ki] * dim, dtype=np.float64)
        self.kd = np.array(kd if hasattr(kd, '__len__') else [kd] * dim, dtype=np.float64)

        self._prev_error = np.zeros(dim)
        self._integral = np.zeros(dim)

    def reset(self, target=None):
        """
        Reset the internal state of the PID controller.

        Args:
            target (optional): New target to reset to.
        """
        if target is not None:
            self.setpoint = np.array(target, dtype=np.float64)
            dim = self.setpoint.shape[0]
            self._prev_error = np.zeros(dim)
            self._integral = np.zeros(dim)
        else:
            self._prev_error.fill(0)
            self._integral.fill(0)

    def get_error(self):
        """
        Returns:
            float: Magnitude of the last error vector.
        """
        return np.linalg.norm(self._prev_error)

    def update(self, current_pos, dt):
        """
        Compute the PID control signal.

        Args:
            current_pos (array-like): Current position (any dimension).
            dt (float): Time delta in seconds.

        Returns:
            np.ndarray: Control output vector.
        """
        current_pos = np.array(current_pos, dtype=np.float64)
        error = self.setpoint - current_pos

        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else np.zeros_like(error)

        output = (
            self.kp * error +
            self.ki * self._integral +
            self.kd * derivative
        )

        self._prev_error = error
        return output


class MoveToPolicy(object):
    """
    Policy to move end-effector to a target position (and optionally orientation).
    
    Supports both position-only and position+orientation control.
    """
     
    def __init__(self, target, orientation=None, kp_pos=1.0, ki_pos=2.0, kd_pos=0.5, 
                 kp_ori=1.5, ki_ori=0.0, kd_ori=0.3):
        """
        Initialize the MoveToPolicy.

        Args:
            target: Target position [x, y, z] or dict with 'pos' and 'ori' keys
            orientation: (Optional) Target orientation as quaternion [x, y, z, w] or rotation matrix (3x3)
            kp_pos, ki_pos, kd_pos: PID gains for position control
            kp_ori, ki_ori, kd_ori: PID gains for orientation control
        """
        self.dt = 0.05
        
        # Handle dict input (e.g., {'pos': [x,y,z], 'ori': [qx,qy,qz,qw]})
        if isinstance(target, dict):
            target_pos = np.array(target['pos'])
            orientation = target.get('ori', orientation)
        else:
            target_pos = np.array(target)
        
        # Position PID controller
        self.pid_pos = PID(kp_pos, ki_pos, kd_pos, target=target_pos)
        
        # Orientation control
        self.use_orientation = orientation is not None
        if self.use_orientation:
            # Convert orientation to quaternion if needed
            if isinstance(orientation, np.ndarray):
                if orientation.shape == (3, 3):
                    # Rotation matrix to quaternion
                    self.target_quat = R.from_matrix(orientation).as_quat()
                elif len(orientation) == 4:
                    # Already quaternion [x, y, z, w]
                    self.target_quat = np.array(orientation)
                else:
                    raise ValueError("Orientation must be quaternion [x,y,z,w] or 3x3 matrix")
            else:
                self.target_quat = np.array(orientation)
            
            # PID for orientation error (axis-angle representation)
            self.pid_ori = PID(kp_ori, ki_ori, kd_ori, target=np.zeros(3))
        

    def get_action(self, robot0_eef_pos, robot0_eef_quat=None):
        """
        Compute action to move end-effector to target.

        Args:
            robot0_eef_pos: Current end-effector position [x, y, z]
            robot0_eef_quat: (Optional) Current end-effector orientation as quaternion [x, y, z, w]

        Returns:
            action: 7D action array [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        action = np.zeros(7)

        # Position control
        action[:3] = self.pid_pos.update(robot0_eef_pos, self.dt)
        
        # Orientation control
        if self.use_orientation and robot0_eef_quat is not None:
            # Compute orientation error as axis-angle
            ori_error = get_orientation_error(robot0_eef_quat, self.target_quat)
            action[3:6] = self.pid_ori.update(ori_error, self.dt)
        
        # Keep gripper closed
        action[-1] = 1
        
        return action

 


class LiftPolicy(object):
    """
    A simple PID-based policy for a robotic arm to lift an object in phases:
    1. Move above the object.
    2. Lower to grasp the object.
    3. Secure grip (hold for stability).
    4. Lift the object.

    The policy uses a PID controller to drive the robot's end-effector to a sequence of target positions
    while managing the gripper state based on the current phase of motion.
    
    NO ORIENTATION CONTROL - gripper maintains default downward orientation.
    """

    def __init__(self, obs):
        """
        Initialize the LiftPolicy with the first observation from the environment.

        Args:
            obs: Can be either:
                - dict with 'cube_pos' key (for get_action_lowdim)
                - 3D numpy array [x, y, z] (for direct position)
        """
        if isinstance(obs, dict):
            target_pos = np.array(obs['cube_pos'])
        else:
            target_pos = np.array(obs)
            
        self.offset = np.array([0, 0.0, 0.1])  # Offset above the cube
        self.dt = 0.1  # Time step for PID update
        self.timeout = 100  # Default steps per phase
        self.phase_steps = 0

        # Gentler PID gains for stable gripping
        self.pid = PID(5, 0.5, 0.8, target=target_pos + self.offset)

        self.phase = 0  # 0: approach, 1: descend, 2: touch, 3: secure grip, 4: lift
        
        # More conservative descent for reliable grasping
        self.target_pos = [
            target_pos + self.offset,    # phase 0: approach from above
            target_pos,                # phase 1: reach
            target_pos-0.003,           # phase 2: secure grip
            target_pos + self.offset      # phase 3: lift up
        ]

    def get_action_proprio(self, robot0_eef_pos):
        """
        Compute the next action for the robot based on current proprioception eef_pos observation.

        Args:
            robot0_eef_pos: Current end-effector position [x, y, z]

        Returns:
            np.ndarray: 7D action array for robosuite:
                - action[:3]: XYZ end-effector velocity (from PID)
                - action[3:6]: Angular velocity (zeros - no orientation control)
                - action[-1]: Gripper command (1 to close, -1 to open)
        """
        action = np.zeros(7)
        self.phase_steps += 1

        # Gripper control - start closing early
        if self.phase >= 2:
            action[-1] = 1   # close gripper
        else:
            action[-1] = -1  # open gripper

        # Positional PID control only (no orientation)
        current_pos = robot0_eef_pos 
        action[:3] = self.pid.update(current_pos, self.dt)
        # action[3:6] remain zeros (no orientation control)

        # Check if close enough to target
        err = self.pid.get_error()
        
        
        # Normal phase transitions
        if (err < 0.004 or self.phase_steps > self.timeout) and self.phase < 3:
            print(f"Phase {self.phase} complete at step {self.phase_steps}")
            self.phase += 1
            self.phase_steps = 0
            if self.phase < len(self.target_pos):
                self.pid.reset(target=self.target_pos[self.phase])

        return action

    def get_action_lowdim(self, obs):
        """
        Compute the next action for the robot based on current low-dim observation.

        Args:
            obs (dict): Current observation. Must include:
                - 'robot0_eef_pos': Current end-effector position.

        Returns:
            np.ndarray: 7D action array for robosuite:
                - action[:3]: XYZ end-effector velocity (from PID)
                - action[3:6]: Angular velocity (zeros)
                - action[-1]: Gripper command (1 to close, -1 to open)
        """
        current_pos = obs['robot0_eef_pos']
        return self.get_action_proprio(current_pos)