import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from main import IDSEnvironment, ReplayBuffer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

class QRDQNAgent:
    def __init__(self, state_size, action_size, num_quantiles=51):
        self.state_size = state_size
        self.action_size = action_size
        self.num_quantiles = num_quantiles
        self.tau = np.linspace(0.0, 1.0, num_quantiles).astype(np.float32)

        self.memory = []
        self.gamma = 0.95  # discount rate
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
        model.add(tf.keras.layers.Dense(self.action_size * self.num_quantiles, activation='linear'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss=self.quantile_huber_loss, optimizer=optimizer)
        return model

    def quantile_huber_loss(self, y_true, y_pred):
        # Compute Quantile Huber Loss
        err = y_true - y_pred
        batch_size = tf.shape(err)[0]

        # Tile tau to the batch size first
        tau_tiled_batch = tf.tile(self.tau, [batch_size])  # Shape: [batch_size, num_quantiles]
        tau_reshaped = tf.reshape(tau_tiled_batch,
                                  [batch_size, 1, self.num_quantiles])  # Add action_size dimension
        tau_repeated = tf.tile(tau_reshaped, [1, self.action_size, 1])  # Tile across action_size

        err_reshaped = tf.reshape(err, [-1, self.action_size, self.num_quantiles])
        err_reshaped = tf.cast(err_reshaped, tf.float32)

        huber_loss = tf.where(tf.abs(err_reshaped) < 1.0, 0.5 * tf.square(err_reshaped),
                              tf.abs(err_reshaped) - 0.5)
        quantile_loss = tf.abs(
            tau_repeated - tf.cast(tf.less(err_reshaped, 0), dtype=tf.float32)) * huber_loss

        return tf.reduce_mean(tf.reduce_sum(quantile_loss, axis=-1))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        action_values = self.model.predict(state, verbose=0).reshape(self.action_size, self.num_quantiles)
        return np.argmax(np.sum(action_values, axis=1))  # returns action

    def train(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = np.array(states).reshape(-1, self.state_size)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states).reshape(-1, self.state_size)
        dones = np.array(dones)

        next_action_values = self.model.predict(next_states, verbose=0).reshape(-1, self.action_size, self.num_quantiles)
        next_actions = np.argmax(np.sum(next_action_values, axis=2), axis=1)

        target_quantiles = rewards[:, None] + (1 - dones[:, None]) * self.gamma * \
                           next_action_values[np.arange(len(next_actions)), next_actions]
        target_quantiles = target_quantiles[:, None, :].repeat(self.action_size, axis=1)

        action_indices = np.repeat(np.arange(self.action_size)[None, :], len(actions),
                                   axis=0) == actions[:, None]

        current_predictions = self.model.predict(states, verbose=0).reshape(-1, self.action_size,
                                                                            self.num_quantiles)
        targets = np.where(action_indices[:, :, None], target_quantiles, current_predictions)

        self.model.train_on_batch(states,
                                  targets.reshape(-1, self.action_size * self.num_quantiles))


def train_qrdqn_agent(env, num_episodes=100, batch_size=32, gamma=0.95):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = QRDQNAgent(state_size, action_size)
    memory_buffer = ReplayBuffer(capacity=1000)

    rewards = []

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

def train_qr_dqn_agent(env, num_episodes=10, batch_size=32, gamma=0.95):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = QRDQNAgent(state_size, action_size)
    memory_buffer = ReplayBuffer(capacity=1000)

    rewards = []

    for episode in range(num_episodes):
        curr_state = env.reset()
        curr_state = np.reshape(curr_state, [1, state_size])
        total_reward = 0
        complete = False

        while not complete:
            curr_action = agent.act(curr_state)
            nxt_state, curr_reward, complete, _ = env.step(curr_action)
            nxt_state = np.reshape(nxt_state, [1, state_size])

            memory_buffer.add(curr_state, curr_action, curr_reward, nxt_state, complete)

            if len(memory_buffer) > batch_size:
                experiences = memory_buffer.sample(batch_size)
                agent.train(experiences)

            curr_state = nxt_state
            total_reward += curr_reward

        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    return rewards, agent


def test(agent, env, num_episodes=100):
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
                env.current_data_pointer - 1, -1]  # -1 since we've already moved the pointer in the env.step() method
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
    training_rewards, agent = train_qr_dqn_agent(env)
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


