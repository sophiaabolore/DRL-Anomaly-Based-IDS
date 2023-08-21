from QRDQN import QRDQNAgent
import tensorflow as tf
import numpy as np
import random
from main import ReplayBuffer, IDSEnvironment
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

SEEDS = [42, 101, 123, 254, 999]  # Example seed values
class IQNAgent(QRDQNAgent):  # We can inherit from QRDQNAgent as many functionalities are shared
    def __init__(self, state_size, action_size, num_quantiles=51, embedding_dim=64):
        # super().__init__(state_size, action_size, num_quantiles)
        self.state_size = state_size
        self.action_size = action_size
        self.embedding_dim = embedding_dim
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
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu, True)
        inputs = tf.keras.layers.Input(shape=(self.state_size,))
        tau_samples = tf.keras.backend.random_uniform(shape=(self.num_quantiles,), minval=0, maxval=1)
        tau_embed = tf.keras.layers.Embedding(input_dim=self.num_quantiles, output_dim=self.embedding_dim)(tau_samples)
        tau_embed = tf.reduce_sum(tau_embed, axis=1)

        x = tf.keras.layers.Dense(24, activation='relu')(inputs)
        x = tf.keras.layers.Dense(24, activation='relu')(x)
        combined = tf.keras.layers.Multiply()([x, tau_embed])
        outputs = tf.keras.layers.Dense(self.action_size, activation='linear')(combined)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss=self.quantile_huber_loss, optimizer=optimizer)
        return model

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Here, we need to estimate the quantile function for each action and take the mean over quantiles
        action_values = [np.mean(self.model.predict(state, verbose=0)) for _ in range(self.num_quantiles)]
        return np.argmax(action_values)  # returns action

    def train(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = np.array(states).reshape(-1, self.state_size)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states).reshape(-1, self.state_size)
        dones = np.array(dones)

        next_action_values = self.model.predict(next_states, verbose=0).reshape(-1,
                                                                                self.action_size,
                                                                                self.num_quantiles)
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
        loss = self.model.train_on_batch(states,
                                         targets.reshape(-1, self.action_size * self.num_quantiles))
        return loss

def train_iqn_agent(env, num_episodes=10, batch_size=32, gamma=0.95):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = QRDQNAgent(state_size, action_size)
    memory_buffer = ReplayBuffer(capacity=1000)
    all_true_labels = []
    all_predicted_labels = []
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
            print(curr_action)
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
            print(action)
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

def visualize_epsilon_decay(agent, num_episodes):
    epsilons = [agent.epsilon * (agent.epsilon_decay ** i) for i in range(num_episodes)]
    plt.figure(figsize=(10, 5))
    plt.plot(epsilons)
    plt.title('Epsilon Decay over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    env = IDSEnvironment()

    # Store results from all seeds
    all_results = []
    all_test_rewards = []
    all_training_rewards = []

    for seed in SEEDS:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        training_rewards, agent = train_iqn_agent(env)
        results, test_rewards = test(agent, env)

        all_training_rewards.append(training_rewards)
        all_test_rewards.append(test_rewards)
        all_results.append(results)

    # Averaging and other processing can be done on all_results, all_test_rewards, and all_training_rewards if needed.

    visualize_training_results(training_rewards)
    visualize_epsilon_decay(agent, 100)

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

    # Detection Rate and False Positive Rate
    TP = results['Confusion Matrix'][1][1]
    TN = results['Confusion Matrix'][0][0]
    FP = results['Confusion Matrix'][0][1]
    FN = results['Confusion Matrix'][1][0]

    detection_rate = TP / (TP + FN)
    false_positive_rate = FP / (FP + TN)

    print(f'Detection Rate: {detection_rate}')
    print(f'False Positive Rate: {false_positive_rate}')
