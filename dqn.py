import numpy as np
import random
from collections import deque
from keras.models import Sequential, load_model, clone_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.initializers import VarianceScaling
from keras.losses import Huber, MeanSquaredError
import os
import tensorflow as tf
# Next run, change gamma to 0.75, change the state normalizer to 10 states, e.g round 1 decimal
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, tau=0.01, 
                 epsilon=1.0, epsilon_decay=0.988, epsilon_min=0.01, memory_size=20000, 
                 verbose=0, model_path=None, step_count=0, seed_value=42):
        # Set random seeds for reproducibility
        np.random.seed(seed_value)
        random.seed(seed_value)
        tf.random.set_seed(seed_value)

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.step_count = step_count
        self.verbose = verbose
        self.episode = 0
        if model_path and os.path.isfile(model_path):
            self.model = load_model(model_path)
        else:
            self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.training_loss = []
        self.validation_loss = []
        

    def update_target_model(self):
        """Soft update model parameters."""
        q_model_theta = self.model.get_weights()
        target_model_theta = self.target_model.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_model_theta, target_model_theta):
            target_weight = target_weight * (1 - self.tau) + q_weight * self.tau
            target_model_theta[counter] = target_weight
            counter += 1
        self.target_model.set_weights(target_model_theta)



    def _build_model(self):
        """Neural Net for Deep-Q learning Model."""
        init = VarianceScaling(scale=2, mode='fan_in', distribution='uniform')
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu', kernel_initializer=init))
        # model.add(Dropout(0.2)) # Help reduce overfitting
        model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu')) # Add extra layer
        # model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # With StateNormalizer
    def remember(self, state, action, reward, next_state, done):
        """Store experiences in replay memory."""
        try:
            # Directly append the states assuming they are already the correct numpy arrays
            self.memory.append((state, action, reward, next_state, done))
        except Exception as ex:
            print(f"An exception occurred during agent remember: {ex}")

    def act(self, state):
        """Return action based on the current state."""
        try:
            state = np.array(state).reshape(1, -1)  # Ensure state is a 2D array
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])
        except Exception as ex:
            print(f"An exception occurred during action prediction: {ex}")
            # Handle the exception, for example by taking a random action
            return random.randrange(self.action_size)

    def replay(self, batch_size, validation=0.2):
        """Train the model using randomly sampled experiences from the memory."""
        if len(self.memory) < batch_size:
            return  # Ensure there are enough samples in the memory

        minibatch = random.sample(self.memory, batch_size)

        states_train, q_values_train = self.prepare_data(minibatch)
        
        # Train the model on the states and the updated Q-values
        self.model.fit(states_train, q_values_train, epochs=1, verbose=0, batch_size=batch_size)

        # Soft update the target model every 2nd step
        if self.step_count % 5 == 0:
            self.update_target_model()
            # self.tau = self.tau*1.01

        self.step_count += 1

        if (self.episode + 1) % 10 == 0:

            self.training_loss.append(self.model.evaluate(states_train, q_values_train, verbose=0))
        
    # Function to extract components and prepare data
    def prepare_data(self, samples):
        states = np.array([x[0] for x in samples])
        actions = np.array([x[1] for x in samples])
        rewards = np.array([x[2] for x in samples])
        next_states = np.array([x[3] for x in samples])
        dones = np.array([x[4] for x in samples])

        # Dynamic reshaping based on actual sample size
        actual_batch_size = states.shape[0]
        states = states.reshape((actual_batch_size, -1))
        next_states = next_states.reshape((actual_batch_size, -1))

        # Predict the next state Q-values from the target network for stability
        next_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1)

        # Compute the target Q-values for all actions; only update the action taken
        targets = rewards + (self.gamma * max_next_q_values * (~dones))

        # Get current Q-values predictions for all actions, only adjust those taken
        current_q_values = self.model.predict(states, verbose=0)
        current_q_values[np.arange(actual_batch_size), actions] = targets

        return states, current_q_values

    
    def epsilon_dec(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1
    
    

    def load(self, path, target_path=None):
        """Load saved model."""
        try:
            # Load the main model
            if path:
                self.model = load_model(path)
                print(f"Model loaded from {path}")

            # Load the target model, if a target path is provided
            if target_path:
                self.target_model = load_model(target_path)
                print(f"Target model loaded from {target_path}")

            # Optionally print the summary of models
            print("Main model summary:")
            self.model.summary()
            if target_path:
                print("Target model summary:")
                self.target_model.summary()

        except Exception as ex:
            print(f"An exception occurred loading the models: {ex}")

    def save(self, path_to_model, target_path):
        """Save the complete model."""
        try:
            self.model.save(path_to_model)
            self.target_model.save(target_path)
        except Exception as ex:
            print(f"An exception occurred saving the model: {ex}")


    def reset(self):
        """Reset the agent state between episodes."""
        self.memory.clear()
        # Reset epsilon to initial value
        # self.epsilon = self.initial_epsilon
        # Optionally reset model weights
        # self.model.set_weights(self.target_model.get_weights())
