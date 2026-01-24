import numpy as np
from pid import PID

class StackPolicy(object):
    """
    Policy for the Block Stacking task.
    Phase 1: Hover over Cube A.
    Phase 2: Lower and Grasp Cube A.
    Phase 3: Lift and Move above Cube B.
    Phase 4: Stack on Cube B and Release.
    """
    def __init__(self, obs):
        """
        Args:
            obs (dict): Includes 'cubeA_pos' and 'cubeB_pos'.
        """
        # Initialize PID controller and waypoints here
        pass
        
    def get_action(self, obs):
        """
        Returns:
            np.ndarray: 7D action (Delta-X, Delta-Y, Delta-Z, Axis-Angle [3], Gripper).
        """
        pass

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