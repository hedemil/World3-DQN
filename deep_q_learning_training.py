import os
import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time

from dqn import DQNAgent
from state_reward import StateNormalizer, calculate_reward, calculate_derivative
from pyworld3 import World3

# Seed value
seed_value = 42

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 3. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Actions and control signals setup
actions = [0.5, 1, 2]  # Action space
control_signals = ['lmhs', 'nruf'] # ['icor', 'fioac', 'fioaa', 'nruf'] 


# Generate all action combinations
action_combinations = list(itertools.product(actions, repeat=len(control_signals)))

# Define the environment/simulation parameters
state_size = 6  # Number of components in the state vector
action_size = len(action_combinations)
agent = DQNAgent(state_size, action_size)
num_bins = 10
episodes = 500
batch_size = 32
year_step = 5
year_max = 2300
year_start = 1905

# Create an instance of the StateNormalizer
state_normalizer = StateNormalizer()

def get_state(world3):
    state = {}
    state['le'] = world3.le[-1]
    # state['le_derivative'] = calculate_derivative(world3.le)
    state['population'] = world3.pop[-1]
    state['iopc'] = world3.iopc[-1]
    # state['sopc'] = world3.sopc[-1]
    state['ai'] = world3.ai[-1]  # consider normalizing by population if relevant
    state['ppgr'] = world3.ppgr[-1]
    state['nrfr'] = world3.nrfr[-1]

    state_normalizer.update_stats(state)
    # Normalize or scale state values as appropriate
    normalized_state = state_normalizer.normalize_state(state)  # Assuming state_normalizer is set up
    return np.array(list(normalized_state.values())).reshape(1, -1)


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
        k_index = prev_run_prop['k']
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

# Paths for saving models
save_path = "/content/drive/My Drive/Colab Notebooks/"


try:
    episode_rewards = []  # List to store sum of rewards for each episode
    # Reward for doing nothing is -2102.7390384207674

    for e in range(episodes):
        # -52502.50734520315
        total_reward = 0
        # agent.reset()

        print('Epsilon: ', agent.epsilon)
        start_time = time.time()  # Start timing the episode
        # Initialize start values for simulation
        try:
            prev_data, world3_start = run_world3_simulation(year_min=1900, year_max=year_start)
        except Exception as ex:
            print(f"Failed to initialize the World3 simulation: {ex}")

        current_state = get_state(world3_start)

        for year in range(year_start, year_max + 1, year_step):
            
            try:
                action_index = agent.act(current_state)
                prev_data, next_state, reward, done = simulate_step(year, prev_data, action_index, control_signals)
                agent.remember(current_state, action_index, reward, next_state, done)
                total_reward += reward
                current_state = next_state
            except ValueError as ve:
                print(f"Model prediction or memory operation failed: {ve}")
            except Exception as ex:
                print(f"An error occurred during the simulation step {e}: {ex}") 
                continue  

            if done:
                agent.epsilon_dec()
                break

            if len(agent.memory) > batch_size:
                try:
                    agent.replay(batch_size)
                except RuntimeError as re:
                    print(f"Error during training: {re}")
                except Exception as ex:
                    print(f"Unexpected error during training: {ex}")
            
            


        # Update the target network at the end of each episode
        # agent.update_target_model()

        episode_rewards.append(total_reward)
        end_time = time.time()  # End timing the episode
        duration = end_time - start_time
        
        print(f"Episode {e+1} completed in {duration:.2f} seconds with Total Reward: {total_reward}")

        if (e + 1) % 500 == 0:
            print(f"Episode: {e + 1}/{episodes}")
            #agent.save(f"{save_path}model_weights_episode_{e+1}.h5")
            try:
                agent.save(f"{save_path}episode_{e+1}_model.keras", f"{save_path}episode_{e+1}_target_model.keras")
                # agent.save(f"episode_{e+1}_model.keras")
                print(f"Episode: {e + 1}/{episodes} saved sucesfully")
            except Exception as ex:
                print('Failed to save ' f'Episode: {e + 1}/{episodes}, exception: {ex}')

except Exception as ex:
    print(f"An unexpected error occurred: {ex}")

try:
    # agent.save('final_model.keras')
    # agent.save(f"{save_path}final_model.keras", f"{save_path}final_target_model.keras")
    # agent.save("final_model.weights.h5")
    print('Model saved succesfully')

except Exception as ex:
    print('Failed to save model')

def plot_rewards(episode_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode Over Time')
    plt.legend()
    plt.grid(True)
    # plot_path = f"{save_path}reward_plot.png"
    # plt.savefig(plot_path)
    plt.show()
    # print(f"Reward plot saved: {plot_path}")

def plot_loss(agent):
    plt.figure(figsize=(10, 5))
    plt.plot(agent.training_loss, label='Loss every 10 Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss every 10 Episodes Over Time')
    plt.legend()
    plt.grid(True)
    # plot_path = f"{save_path}loss_plot.png"
    # plt.savefig(plot_path)
    plt.show()
    # print(f"Reward plot saved: {plot_path}")

def moving_average(values, window_size):
    """ Compute a simple moving average. """
    cumsum = np.cumsum(values, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def plot_ma(value, metric):
    plt.plot(range(len(value)), value)
    plt.xlabel('Episodes')
    plt.ylabel(f'Moving Average of {metric}')
    plt.title(f'Moving Average of {metric}s Over Episodes')
    # plot_path = f"{save_path}ma{metric}_plot.png"
    # plt.savefig(plot_path)
    plt.show()

    

window_size = 10
ma_rewards = moving_average(episode_rewards, window_size)
ma_loss = moving_average(agent.training_loss, window_size)

plot_rewards(episode_rewards)
plot_ma(ma_rewards, 'Reward')

plot_loss(agent)
plot_ma(ma_loss, 'Loss')


