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


class RotationPID:
    """
    PID controller for angular control with proper angle wrapping.

    Handles the discontinuity at ±π by normalizing angular errors.
    Includes integral anti-windup by clamping the integral term.
    """

    def __init__(self, kp: float, ki: float, kd: float, target: float = 0.0):
        """
        Initialize rotation PID controller.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
            target: Target angle in radians.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral_error = 0.0
        self.previous_error = None
        self.integral_limit = np.pi  # Anti-windup limit

    def reset(self, target: float = None) -> None:
        """
        Reset internal state and optionally set new target.

        Args:
            target: New target angle in radians (optional).
        """
        self.integral_error = 0.0
        self.previous_error = None
        if target is not None:
            self.target = target

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-π, π] range.

        Args:
            angle: Angle in radians.

        Returns:
            Normalized angle in [-π, π].
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

    def get_error(self) -> float:
        """
        Get the magnitude of the last angular error.

        Returns:
            Absolute angular error in radians, or 0.0 if not yet updated.
        """
        if self.previous_error is None:
            return 0.0
        return abs(self.previous_error)

    def update(self, current_angle: float, dt: float) -> float:
        """
        Compute PID control signal for angular control.

        Args:
            current_angle: Current angle in radians.
            dt: Time step since last update (seconds).

        Returns:
            Control output (rotation rate command).
        """
        # Compute error with proper angle wrapping
        error = self._normalize_angle(self.target - current_angle)

        # Proportional term
        P = self.kp * error

        # Integral term with anti-windup
        self.integral_error += error * dt
        self.integral_error = np.clip(
            self.integral_error, -self.integral_limit, self.integral_limit
        )
        I = self.ki * self.integral_error

        # Derivative term
        if self.previous_error is None:
            D = 0.0
        else:
            # Use normalized difference for derivative
            error_diff = self._normalize_angle(error - self.previous_error)
            D = self.kd * error_diff / dt

        # Store current error for next iteration
        self.previous_error = error

        return P + I + D
