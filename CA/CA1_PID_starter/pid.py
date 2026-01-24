import numpy as np

class PID:
    def __init__(self, kp, ki, kd, target):
        # Initialize gains and setpoint
        pass
        
    def reset(self, target=None):
        # Reset internal error history
        pass
        
    def get_error(self):
        # Return magnitude of last error
        pass

    def update(self, current_pos, dt):
        # Compute and return control signal
        pass