import numpy as np


class PID:
    def __init__(self, kp, ki, kd, target):
        """
        Initialize a PID controller.

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            target (tuple or array): Target position.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = np.array(target)
        self.integral_error = np.array([0.0, 0.0, 0.0])
        self.previous_error = None

    def reset(self, target=None):
        """
        Reset internal error history for a new control task.

        Args:
            target (tuple or array, optional): New target position.
                   If None, keeps the current target.
        """
        self.integral_error = np.array([0.0, 0.0, 0.0])
        self.previous_error = None
        if target is not None:
            self.target = np.array(target)
    
    def get_error(self):
        """
        Get the magnitude of the last error.

        Returns:
            float: Euclidean distance from current position to target.
                   Returns 0.0 if update() hasn't been called yet.
        """
        if self.previous_error is None:
            return 0.0
        return np.linalg.norm(self.previous_error)


    def update(self, current_pos, dt):
        """
        Compute the control signal based on current position and time step.

        Args:
            current_pos (array-like): Current position (3D).
            dt (float): Time step since last update (seconds).

        Returns:
            np.ndarray: Control output vector (3D).
        """
        current_pos = np.array(current_pos)
        error = self.target - current_pos

        # Proportional term
        P = self.kp * error

        # Integral term (accumulate error over time)
        self.integral_error += error * dt
        I = self.ki * self.integral_error

        # Derivative term (rate of change of error)
        if self.previous_error is None:
            D = np.array([0.0, 0.0, 0.0])
        else:
            D = self.kd * (error - self.previous_error) / dt

        # Store current error for next iteration
        self.previous_error = error

        return P + I + D
