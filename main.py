import gym
from gym import spaces
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Optional, Dict, Any, Tuple
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.io import arff

def discretize(value, bins):
    return np.digitize(value, bins, right=True)

def process_dataset():
    train_data_arff, meta_train = arff.loadarff("NSL-KDD/KDDTrain+.arff")
    test_data_arff, meta_test = arff.loadarff("NSL-KDD/KDDTest+.arff")

    # Convert ARFF data to pandas DataFrame
    train_data = pd.DataFrame(train_data_arff)
    test_data = pd.DataFrame(test_data_arff)

    # Convert byte-strings to strings for all columns
    for column in train_data.select_dtypes(include=[object]).columns:
        train_data[column] = train_data[column].str.decode('utf-8')
    for column in test_data.select_dtypes(include=[object]).columns:
        test_data[column] = test_data[column].str.decode('utf-8')

    # 1. Handling Missing Values
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)
    anomaly_count = sum(train_data.iloc[:, -1] == 'normal')
    normal_count = sum(train_data.iloc[:, -1] == 'anomaly')
    print(train_data.iloc[:, -1])
    print("total # of anomalies: ", anomaly_count)
    print("total # of normals: ", normal_count)

    # 2. Normalize Numerical Features
    numerical_cols = [0, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16,
                      17, 18, 19, 22, 23, 24, 25, 26, 27, 28,
                      29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

    # 3. Convert Categorical Features
    categorical_cols = [1, 2, 3, 6, 11, 20, 21, 41]

    scaler = StandardScaler()
    train_data.iloc[:, numerical_cols] = scaler.fit_transform(train_data.iloc[:, numerical_cols])
    test_data.iloc[:, numerical_cols] = scaler.fit_transform(test_data.iloc[:, numerical_cols])  # Use the same scaler on test data

    # Convert column indices to column names for categorical columns
    categorical_colnames = train_data.columns[categorical_cols].tolist()
    # One-hot encode categorical features
    train_data = pd.get_dummies(train_data, columns=categorical_colnames)
    test_data = train_data
    # Get missing columns in the test set
    missing_cols = set(train_data.columns) - set(test_data.columns)
    # Add a missing column in the test set with default value equal to 0
    for c in missing_cols:
        test_data[c] = 0

    test_data = test_data[train_data.columns]
    # for col in categorical_cols:
    #     le = LabelEncoder()
    #     train_data[col] = le.fit_transform(train_data[col])
    #     test_data[col] = le.transform(test_data[col])  # Use the same label encoder on test data
    return (train_data, test_data)

class IDSEnvironment(gym.Env):

    def __init__(self, dataset_path="KDDTrain+.txt"):
        super(IDSEnvironment, self).__init__()

        # Load and preprocess the dataset
        self.train_data, self.test_data = process_dataset()
        self.num_features = self.train_data.shape[1] - 1  # exclude label column
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_features,),
                                            dtype=np.float32)
        self.current_data_pointer = 0


        # Define action space (binary decision for now: alert vs. no alert)
        self.action_space = spaces.Discrete(2)
        self.state_space = 148
        self.state = self.train_data.iloc[self.current_data_pointer, :-1].values

    def discretize_state(self, state):
        # Define bins for each feature
        bins = [np.linspace(0, 1, 11) for _ in range(self.num_features)]  # 10 bins for example
        discretized_state = [discretize(state[i], bins[i]) for i in range(self.num_features)]
        return np.array(discretized_state)

    def step(self, curr_action):
        self.state = self.discretize_state(self.train_data.iloc[self.current_data_pointer, :-1].values)

        # Actual label
        intrusion = self.train_data.iloc[self.current_data_pointer, -1]
        reward_base = 1 if curr_action == intrusion else -1
        curr_reward = np.random.normal(loc=reward_base, scale=0.1)  # 0.1 is the standard deviation

        # Move the data pointer
        self.current_data_pointer += 1

        # complete = self.current_data_pointer >= len(self.train_data)

        complete = self.current_data_pointer >= 100

        return self.state, curr_reward, complete, {}

    def reset(self, *args, **kwargs):
        self.state = self.discretize_state(self.train_data.iloc[0, :-1].values)

        self.current_data_pointer = 0

        seed = kwargs.get('seed', None)
        options = kwargs.get('options', None)

        if seed is not None:
            np.random.seed(seed)

        # Handle the options as needed...

        return self.state

    def render(self, mode='human'):
        if mode == 'human':
            print("Current State:", self.state)
        elif mode == 'ansi':
            return "Current State: " + str(self.state)
        else:
            raise ValueError("Unsupported render mode: " + mode)

    def close(self):
        pass

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def visualize_training_results(rewards):
    """
    Visualizes the training results.

    Args:
    - rewards (list): A list of rewards received at each episode.
    """

    # Calculate moving average with window size of 100
    moving_avg = [np.mean(rewards[max(0, i - 100):i + 1]) for i in range(len(rewards))]

    plt.figure(figsize=(10, 5))

    plt.plot(rewards, label='Episode Reward', alpha=0.6)
    plt.plot(moving_avg, label='Moving Average (100 episodes)', color='red')

    plt.title("Training Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    process_dataset()
