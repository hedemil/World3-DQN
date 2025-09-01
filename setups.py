# States
def get_state(world3):
    state = {}
    state['le'] = world3.le[-1]
    state['le_derivative'] = calculate_derivative(world3.le)
    state['population'] = world3.pop[-1]
    state['iopc'] = world3.iopc[-1]
    state['sopc'] = world3.sopc[-1]
    state['ai'] = world3.ai[-1]  # consider normalizing by population if relevant
    state['ppgr'] = world3.ppgr[-1]

    state_normalizer.update_stats(state)
    # Normalize or scale state values as appropriate
    normalized_state = state_normalizer.normalize_state(state)  # Assuming state_normalizer is set up
    return np.array(list(normalized_state.values())).reshape(1, -1)

# Rewards
def calculate_reward(current_world):
    reward = 0
    
    le_value = current_world.le[-1]
    le_derivative = calculate_derivative(current_world.le)

    if le_value < 20:
        reward -= 100

    elif le_value < 30:
        reward -= 50

    # Encourage growth until LE reaches 60
    elif le_value < 60:
        reward += 10 * le_derivative  # Proportional reward based on the rate of increase

    # When LE is above 60, encourage maintaining it and penalize large changes
    elif le_value >= 60:
        reward += 50
        if abs(le_derivative) < 0.1:
            reward += 50  # Big reward for stability
        elif abs(le_derivative) < 0.2:
            reward += 10  # Smaller reward for less stability
        else:
            reward -= 100 * abs(le_derivative)  # Proportional penalty for instability

    # Penalize drastic drops in LE no matter the current value
    if le_derivative < -0.2:
        reward -= 100 * abs(le_derivative)  # Large penalty proportional to the rate of decrease
    
    return reward

# Update target network
# 1 - Update every 5 episodes
# 2 - Update every 2nd episode

# Action space
# 1 - [0.5, 1, 1.5]
# 2 - [0.5, 0.75, 1, 1.25, 1.5]

# Learning rate scheduele
# 1 - With
# 2 - Without

# Target network tau
# 1 - tau = tau*1.01 per update
# 2 - tau = 0.01

# Architecture
# 1 - 512 256 256 128 action
# 2 - 256 256 256 128 128 action
# 3 - 256 256 256 128 128 128 action
# 4 - 512 256 256 128 128 128 action

# Episodes
# 1 - 500 ep (Epsilon decay = 0.99)
# 2 - 2000 ep (Epsilon decay = 0.9974)

### Test alone
# Epsilon 
# 1 - Epsilon decay = 0.99 (500 ep)
# 2 - Epsilon decay = 0.9 (500 ep)
