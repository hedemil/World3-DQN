import matplotlib.pyplot as plt
import numpy as np
import itertools

from pyworld3 import World3
from pyworld3.utils import plot_world_variables
from state_reward import StateNormalizer, calculate_derivative, calculate_reward

params = {"lines.linewidth": "3"}
plt.rcParams.update(params)


# Actions and control signals setup
actions = [0.5, 1, 1.5]  # Action space
control_signals = ['icor', 'fioac', 'fioaa']

# Generate all combinations
action_combinations = list(itertools.product(actions, repeat=len(control_signals)))
num_action_combos = len(action_combinations)

# Mapping each combination to an index
action_to_index = {combo: i for i, combo in enumerate(action_combinations)}

# Parameters
alpha = 0.3  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.99
epsilon_min = 0.01
num_bins = 4
state_vars = 7
num_states = 4**state_vars
num_actions = len(actions)
num_control_signals = len(control_signals)

# Initialize Q-table
Q = np.zeros((num_states, num_action_combos))

learning_episodes = 10 
exploraion_episode = 50
year_start = 2000
year_max = 2200
year_step = 5

# Create an instance of the StateNormalizer
state_normalizer = StateNormalizer()

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
    normalized_state = state_normalizer.normalize_state(state, num_bins)  # Assuming state_normalizer is set up
    return np.array(list(normalized_state.values())).reshape(1, -1)

# Q-learning update
def update_Q(state, action_index, reward, next_state):
    future = np.max(Q[next_state])
    Q[state, action_index] = Q[state, action_index] + alpha * (reward + gamma * future - Q[state, action_index])
   
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
        prev_control[control_signal + '_control'] = action_value*prev_control['initial_value'][control_signal + '_control']
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
    try:
        prev_data, world3_current = run_world3_simulation(year_min=year, year_max=next_year, prev_run_data=prev_data, ordinary_run=False)
    except Exception as ex:
        print(f"Failed to initialize the World3 simulation year: {year}, exception: {ex}")
    
    
    next_state = get_state(world3_current)


    # Calculate reward (this function needs to be defined based on your criteria)
    reward = calculate_reward(world3_current)
    
    # Check if simulation is done (e.g., reached final year)
    done = next_year >= year_max

    
    return prev_data, next_state, reward, done


episode_rewards = []

for episode in range(learning_episodes):
    reward_total = 0
    # Run the first simulation
    prev_data, world3_start = run_world3_simulation(year_min=1900, year_max=1905)
    state = get_state(world3_start)

     # Run the model with actions every 5th year
    for year in range(year_start, year_max + 1, year_step):
        
        if np.random.rand() < epsilon:  # Exploration
            action_index = np.random.choice(len(action_combinations))
        else:  # Exploitation
            action_index = np.argmax(Q[state])

        prev_data, next_state, reward, done = simulate_step(year, prev_data, action_index, control_signals)
        
        reward_total += reward

        update_Q(state, action_index, reward, next_state)

        state = next_state

        epsilon = max(epsilon_min, epsilon*epsilon_decay)
    
    episode_rewards.append(reward_total)

def plot_rewards(episode_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_rewards(episode_rewards)

# print(Q)
# optimal_policy = np.argmax(Q, axis=1)
# print("Optimal policy (state -> action index):", optimal_policy)

# prev_data_optimal, world3_optimal = run_world3_simulation(year_min=1900, year_max=2000)

# for year in range(year_start, year_max + 1, year_step):
#     state = get_state(world3_optimal)
    
#     # Use the optimal policy to find the optimal action combination index
#     optimal_action_combination_index = optimal_policy[state]
    
#     # Retrieve the optimal action combination
#     optimal_action_combination = action_combinations[optimal_action_combination_index]
    
#     # Construct the list of control signals and their corresponding actions
#     control_variables_actions = list(zip(control_signals, optimal_action_combination))
    
#     # Update the control signals for the next simulation
#     prev_data_optimal['control_signals'] = update_control(control_variables_actions, prev_data_optimal['control_signals'])
    
#     # Run the simulation for the next time step using the updated control signals
#     prev_data_optimal, world3_optimal = run_world3_simulation(year_min=year, year_max=year + year_step, prev_run_data=prev_data_optimal, ordinary_run=False, k_index=prev_data_optimal["world_props"]["k"])

# variables = [world3_optimal.le, world3_optimal.fr, world3_optimal.sc, world3_optimal.pop]
# labels = ["LE", "FR", "SC", "POP"]

# # Plot the combined results
# plot_world_variables(
#     world3_optimal.time,
#     variables,
#     labels,
#         [[0, 100], [0, 4], [0, 6e12],  [0, 10e9]],
#     figsize=(10, 7),
#     title="World3 Simulation from 1900 to 2100, optimal policy"
# )

# # Initialize a position for the first annotation
# x_pos = 0.05  
# y_pos = 0.95  
# vertical_offset = 0.05  


# ax = plt.gcf().gca()

# for var, label in zip(variables, labels):
#     max_value = np.max(var)
#     ax.text(x_pos, y_pos, f'{label} Max: {max_value:.2f}', transform=ax.transAxes,
#             verticalalignment='top', horizontalalignment='left')
#     y_pos -= vertical_offset  
# plt.show()