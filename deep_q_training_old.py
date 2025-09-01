import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from dqn import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import itertools

from pyworld3 import World3, world3
from pyworld3.utils import plot_world_variables


params = {"lines.linewidth": "3"}
plt.rcParams.update(params)

actions = [0.9, 1.0, 1.1]  # Action space
control_signals = ['alai', 'lyf']

""" control_signals = ['alai', 'lyf', 'ifpc', 'lymap', 'llmy', 'fioaa', 
                   'icor', 'scor', 'alic', 'alsc', 'fioac', 'isopc', 
                   'fioas', 'ppgf', 'pptd', 'nruf', 'fcaor'] """
num_states = 81920
num_actions = len(actions)
num_control_signals = len(control_signals)


# Generate all combinations
action_combinations = list(itertools.product(actions, repeat=len(control_signals)))

def discretize_year(time):
    return (time - 2000)//10 + 1


# States for optimizing hsapc
def discretize_p1(p1):
    if p1 < 1e9: return 0
    elif p1 < 2e9: return 1
    elif p1 < 3e9: return 2
    else: return 3

def discretize_p2(p2):
    if p2 < 1e9: return 0
    elif p2 < 2e9: return 1
    elif p2 < 3e9: return 2
    else: return 3

def discretize_p3(p3):
    if p3 < 1e9: return 0
    elif p3 < 2e9: return 1
    elif p3 < 3e9: return 2
    else: return 3

def discretize_p4(p4):
    if p4 < 1e9: return 0
    elif p4 < 2e9: return 1
    elif p4 < 3e9: return 2
    else: return 3

def discretize_hsapc(hsapc):
    if hsapc < 25: return 0
    elif hsapc < 50: return 1
    elif hsapc < 75: return 2
    else: return 3

def discretize_ehspc(ehspc):
    if ehspc < 20: return 0
    elif ehspc < 40: return 1
    elif ehspc < 60: return 2
    else: return 3


def get_state_vector(p1, p2, p3, p4, hsapc, ehspc, time):
    p1_index = discretize_p1(p1)
    p2_index = discretize_p2(p2)
    p3_index = discretize_p3(p3)
    p4_index = discretize_p4(p4)

    hsapc_index = discretize_hsapc(hsapc)
    ehspc_index = discretize_ehspc(ehspc)

    time_index = discretize_year(time)
    # Return a numpy array with the state represented as a vector
    # return np.array([p1_index, hsapc_index, ehspc_index]).reshape(1, -1)

    return np.array([p1_index, p2_index, p3_index, p4_index, hsapc_index, ehspc_index, time_index]).reshape(1, -1)

# Reward calculation
def calculate_reward(current_world):
    reward = 0
    birth_death = current_world.cbr[-1] / current_world.cdr[-1]
    if  0.9 <= birth_death <= 1.1:
        reward += 100
    else:
        reward += 0
    reward += 0 if current_world.le[-1] < 55 else 100
    reward += 0 if current_world.hsapc[-1] < 50 else 100
    reward -= 10000 if current_world.pop[-1] < 6e9 or current_world.pop[-1] > 8e9 else 0
    return reward

    
def run_world3_simulation(year_min, year_max, dt=1, prev_run_data=None, ordinary_run=True, k_index=1):
    
    prev_run_prop = prev_run_data["world_props"] if prev_run_data else None

    world3 = World3(
            year_max=year_max,
            year_min=year_min, 
            dt=dt,
            prev_world_prop=prev_run_prop,
            ordinary_run=ordinary_run
        )
    
    if prev_run_data:
        world3.set_world3_control(prev_run_data['control_signals'])
        world3.init_world3_constants()
        world3.init_world3_variables(prev_run_data["init_vars"])
        world3.set_world3_table_functions()
        world3.set_world3_delay_functions(prev_delay=prev_run_data["delay_funcs"])
    else:
        world3.set_world3_control()
        world3.init_world3_constants()
        world3.init_world3_variables()
        world3.set_world3_table_functions()
        world3.set_world3_delay_functions()

    world3.run_world3(fast=False, k_index=k_index)
    state = world3.get_state()
    return state, world3

def update_control(control_signals_actions, prev_control):
    """
    Update control signals based on actions.
    :param control_signals_actions: List of tuples (control_signal, action_value)
    :param prev_control: Previous control signals dictionary
    :return: Updated control signals dictionary
    """
    for control_signal, action_value in control_signals_actions:
        prev_control[control_signal + '_control'] *= action_value
    return prev_control

def simulate_step(year, prev_data, action_combination_index, control_signals):
    """
    Simulate one step of the World3 model based on the given action and update control signals.

    :param year: Current year of simulation.
    :param prev_data: Previous run data of the World3 model.
    :param action_combination_index: Index of the selected action combination.
    :param control_signals: List of control signals to be adjusted.
    :return: Tuple of (next_state, reward, done)
    """
    
    # Retrieve the action combination using the selected index
    selected_action_combination = action_combinations[action_combination_index]
    
    # Update control signals based on the selected action
    control_variables_actions = list(zip(control_signals, selected_action_combination))
    prev_data['control_signals'] = update_control(control_variables_actions, prev_data['control_signals'])
    
    # Run the World3 model for the next step
    next_year = year + year_step
    prev_data, world3_current = run_world3_simulation(year_min=year, year_max=next_year, prev_run_data=prev_data, ordinary_run=False)
    
    # Extract necessary variables for state and reward calculation
    current_p1 = world3_current.p1[-1]
    current_p2 = world3_current.p2[-1]
    current_p3 = world3_current.p3[-1]
    current_p4 = world3_current.p4[-1]
    current_hsapc = world3_current.hsapc[-1]
    current_ehspc = world3_current.ehspc[-1]
    current_time = world3_current.time[-1]  
    
    # Calculate next state
    # next_state = get_state_vector(current_p1, current_hsapc, current_ehspc)
    next_state = get_state_vector(current_p1, current_p2, current_p3, current_p4, current_hsapc, current_ehspc, current_time)
    
    # Calculate reward (this function needs to be defined based on your criteria)
    reward = calculate_reward(world3_current)
    
    # Check if simulation is done (e.g., reached final year)
    done = next_year >= year_max

    
    return prev_data, next_state, reward, done

# Define the environment / simulation parameters
state_size = 7  # For example: population, life expectancy, food ratio
action_size = len(action_combinations)  
agent = DQNAgent(state_size, action_size)
episodes = 1
batch_size = 32
year_step = 5
year_max = 2200
year_start = 2000

#initialize start values for simulation
prev_data, world3_start = run_world3_simulation(year_min=1900, year_max=2000)

# Loop over episodes
for e in range(episodes):
    
    current_p1 = prev_data['init_vars']['population']['p1'][-1]
    current_p2 = prev_data['init_vars']['population']['p2'][-1]
    current_p3 = prev_data['init_vars']['population']['p3'][-1]
    current_p4 = prev_data['init_vars']['population']['p4'][-1]
    current_hsapc = prev_data['init_vars']['population']['hsapc'][-1]
    current_ehspc = prev_data['init_vars']['population']['ehspc'][-1]
    current_time = prev_data['world_props']['time'][-1]
    # state = get_state_vector(current_p1, current_hsapc, current_ehspc)
    state = get_state_vector(current_p1, current_p2, current_p3, current_p4, current_hsapc, current_ehspc, current_time)
    for year in range(year_start, year_max + 1, year_step): 
        action = agent.act(state)
        prev_data_ep, next_state, reward, done = simulate_step(year, prev_data if year == year_start else prev_data_ep, action, control_signals)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # Update the target network at the end of each episode
    agent.update_target_model()
    
    # Print out progress and save the model at intervals
    if (e + 1) % 100 == 0:  
        print(f"Episode: {e + 1}/{episodes}")
        agent.save(f"model_weights_episode_{e+1}.weights.h5")

agent.save("final_model.weights.h5")
        
