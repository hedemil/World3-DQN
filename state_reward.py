import numpy as np


# Try pretraining to define the states before actually running, e.g do a standard run to get min-max. 
class StateNormalizer:
    def __init__(self):
        self.min_values = {}
        self.max_values = {}

    def update_stats(self, state):
        for key, value in state.items():
            if key in self.min_values:
                self.min_values[key] = min(value, self.min_values[key])
                self.max_values[key] = max(value, self.max_values[key])
            else:
                self.min_values[key] = value
                self.max_values[key] = value
    
    def normalize_state(self, state):
        normalized_state = {}
        for key, value in state.items():
            if key not in self.min_values:
                # Handle unseen key error or log a warning
                raise ValueError(f"Key {key} not found in running statistics.")
            min_val = self.min_values[key]
            max_val = self.max_values[key]
            range_val = max_val - min_val if max_val > min_val else 1
            normalized_val = (value - min_val) / range_val
            normalized_state[key] = normalized_val
        return normalized_state

def calculate_reward(current_world):
    reward = 0
    
    le_value = current_world.le[-1]
    le_derivative = calculate_derivative(current_world.le)

    target_value = 50
    error = le_value - target_value

    if error > 0:
        reward = -0.5 * abs(error)  # less penalty if above target
    else:
        reward = -2 * abs(error)    # more penalty if below target
    return reward

def calculate_derivative(values):
    # Simple numerical differentiation: finite difference
    return (values[-1]-values[-2])
    # return np.gradient(values)