import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from main import IDSEnvironment, ReplayBuffer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = []
        self.gamma = 0.95  # discount rate
        # self.epsilon = 1.0  # exploration rate

        self.epsilon = 0.1

        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def train(self, experiences):
        # Extract experiences
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert data to numpy arrays
        states = np.array(states).reshape(-1, self.state_size)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states).reshape(-1, self.state_size)
        dones = np.array(dones)

        # Compute the Q-values for the current states and next states
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        # Compute the target Q-values
        targets = rewards + self.gamma * np.max(next_q_values, axis=1) * (1 - dones)

        # Update the Q-values for the taken actions
        rows = np.arange(len(actions))
        q_values[rows, actions] = targets

        # Train the model
        self.model.train_on_batch(states, q_values)


def train_dqn_agent(env, num_episodes=100, batch_size=32, gamma=0.95):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    memory_buffer = ReplayBuffer(capacity=1000)

    rewards = []
    print("training")
    # Training
    for episode in range(num_episodes):
        curr_state = env.reset()
        curr_state = np.reshape(curr_state, [1, state_size])  # Reshape for the DNN
        total_reward = 0  # Reset total_reward for the episode
        complete = False

        print(f'Episode {episode}')

        while not complete:
            curr_action = agent.act(curr_state)  # Decide action
            nxt_state, curr_reward, complete, _ = env.step(curr_action)  # Execute action
            nxt_state = np.reshape(nxt_state, [1, state_size])  # Reshape for the DNN

            # Store experience
            memory_buffer.add(curr_state, curr_action, curr_reward, nxt_state, complete)

            # Train the agent with a batch from the replay buffer
            if len(memory_buffer) > batch_size:
                experiences = memory_buffer.sample(batch_size)
                agent.train(experiences)

            curr_state = nxt_state
            total_reward += curr_reward

        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Evaluation
    total_test_reward = 0
    curr_state = env.reset()
    curr_state = np.reshape(curr_state, [1, state_size])  # Reshape for the DNN
    complete = False

    while not complete:
        curr_action = agent.act(curr_state)  # Use act method for evaluation
        nxt_state, curr_reward, complete, _ = env.step(curr_action)
        nxt_state = np.reshape(nxt_state, [1, state_size])  # Reshape for the DNN
        total_test_reward += curr_reward
        curr_state = nxt_state
    return rewards, agent


def test(agent, env, num_episodes=10):
    """
    Test a DQNAgent on a given environment and compute classification metrics.

    :param agent: The DQNAgent to be tested.
    :param env: The environment to test the agent on.
    :param num_episodes: Number of test episodes.
    :return: A dictionary containing average reward and classification metrics.
    """
    total_rewards = []
    all_true_labels = []
    all_predicted_labels = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        print(f'Episode {episode}')
        while not done:
            action = agent.act(state)  # Use the trained policy to select an action
            next_state, reward, done, _ = env.step(action)
            true_label = env.train_data.iloc[
                env.current_data_pointer - 1, -1]
            episode_reward += reward
            state = next_state

            # Store the true labels and predicted actions
            all_true_labels.append(true_label)
            all_predicted_labels.append(action)

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
    avg_reward = np.mean(total_rewards)

    # Calculate classification metrics
    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    f1 = f1_score(all_true_labels, all_predicted_labels)
    precision = precision_score(all_true_labels, all_predicted_labels)
    recall = recall_score(all_true_labels, all_predicted_labels)
    confusion = confusion_matrix(all_true_labels, all_predicted_labels)

    results = {
        'Average Reward': avg_reward,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'Confusion Matrix': confusion
    }


    # Print results
    for key, value in results.items():
        print(f"{key}: {value}")

    return results, total_rewards


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
    env = IDSEnvironment()
    training_rewards, agent = train_dqn_agent(env)
    visualize_training_results(training_rewards)
    results, test_rewards = test(agent,env)
    # 1. Bar Plot for Metrics
    metrics = ['Average Reward', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
    values = [results[metric] for metric in metrics]

    plt.figure(figsize=(10, 5))
    plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.ylabel('Value')
    plt.title('Metrics Visualization')
    plt.ylim([0, 1])
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()

    # 2. Heatmap for Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['Confusion Matrix'], annot=True, cmap="YlGnBu", fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    visualize_training_results(training_rewards)
    print(results)
