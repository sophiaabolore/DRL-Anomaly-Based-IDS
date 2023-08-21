NUM_ATOMS = 51
V_MIN = -10
V_MAX = 10
from collections import deque
from DQN import DQNAgent, IDSEnvironment, ReplayBuffer
import numpy as np
import tensorflow as tf
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priority = 0.1

        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.pos_to_idx = deque(maxlen=capacity)  # keeps track of the global index
        self.next_idx = 0

    def add(self, *args):
        max_priority = max(self.priorities) if self.buffer else 1.0

        # Add to buffer, priorities, and map to global index
        self.buffer.append(args)
        self.priorities.append(max_priority)
        self.pos_to_idx.append(self.next_idx)
        self.next_idx += 1

    def sample(self, batch_size):
        sampling_probs = np.array(self.priorities) ** self.alpha
        sampling_probs /= sampling_probs.sum()

        pos_indices = np.random.choice(len(self.buffer), batch_size, p=sampling_probs)
        global_indices = [self.pos_to_idx[pos] for pos in pos_indices]

        samples = [self.buffer[idx] for idx in pos_indices]
        weights = (len(self.buffer) * sampling_probs[pos_indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = np.min([1., self.beta + self.beta_increment])
        return samples, global_indices, weights

    def set_priority(self, idx, error):
        try:
            pos = self.pos_to_idx.index(idx)
            self.priorities[pos] = error + self.priority
        except ValueError:
            # The given index doesn't exist in the current buffer
            pass

    def __len__(self):
        return len(self.buffer)

class C51Agent(DQNAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.state_size = state_size
        self.action_size = action_size
        self.num_atoms = NUM_ATOMS
        print(self.num_atoms)
        self.v_min = V_MIN
        self.v_max = V_MAX
        self.delta_z = (V_MAX - V_MIN) / (self.num_atoms - 1)
        self.z = np.linspace(V_MIN, V_MAX, self.num_atoms)
        self.target_model = self._build_model()  # Target network
        self.target_model.set_weights(self.model.get_weights())

        self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate

        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size * NUM_ATOMS, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        dists = self.model.predict(state, verbose=0).reshape((-1, self.action_size, NUM_ATOMS))
        q_values = np.sum(dists * self.z, axis=2)
        return np.argmax(q_values[0])  # returns action

    def train(self, experiences, indices=None, weights=None):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.array(states).reshape(-1, self.state_size)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states).reshape(-1, self.state_size)
        dones = np.array(dones)

        next_dists = self.model.predict(next_states, verbose=0).reshape(
            (-1, self.action_size, NUM_ATOMS))
        next_q_values = np.sum(next_dists * self.z, axis=2)
        next_actions = np.argmax(next_q_values, axis=1)
        m_probs = next_dists[range(len(next_actions)), next_actions]

        # Compute target distributions
        target_dists = np.zeros((len(actions), self.action_size, self.num_atoms))

        for i in range(len(actions)):
            action = actions[i]
            if dones[i]:
                tz = min(self.v_max, max(self.v_min, rewards[i]))
                bj = (tz - self.v_min) / self.delta_z
                l = max(0, min(self.num_atoms - 1, np.floor(bj).astype(int)))
                u = max(0, min(self.num_atoms - 1, np.ceil(bj).astype(int)))
                target_dists[i][action][l] = m_probs[i][l] + (u - bj)
                target_dists[i][action][u] = m_probs[i][u] + (bj - l)
            else:
                for j in range(self.num_atoms):
                    tz = min(self.v_max, max(self.v_min, rewards[i] + self.gamma * self.z[j]))
                    bj = (tz - self.v_min) / self.delta_z
                    l, u = np.floor(bj).astype(int), np.ceil(bj).astype(int)
                    target_dists[i][action][l] += m_probs[i][j] * (u - bj)
                    target_dists[i][action][u] += m_probs[i][j] * (bj - l)

        # Flatten target_dists outside the loop
        target_dists = target_dists.reshape((len(actions), self.action_size * NUM_ATOMS))

        # Train the model
        if weights is not None:
            sample_weights = np.array(weights).reshape(-1, 1)
            loss = self.model.train_on_batch(states, target_dists, sample_weight=sample_weights)
            td_errors = np.abs(
                target_dists.reshape((-1, self.action_size, NUM_ATOMS)) - self.model.predict(
                    states).reshape((-1, self.action_size, NUM_ATOMS)))
            for idx, error in zip(indices, td_errors):
                self.memory.set_priority(idx, error.max())  # max error for updating priority
        else:
            self.model.train_on_batch(states, target_dists)



def train_c51_agent(env, num_episodes=250, batch_size=32, gamma=0.95, use_prioritized_replay=True):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = C51Agent(state_size, action_size)
    if use_prioritized_replay:
        memory_buffer = PrioritizedReplayBuffer(capacity=1000)
    else:
        memory_buffer = ReplayBuffer(capacity=1000)

    rewards = []

    # Training
    for episode in range(num_episodes):
        curr_state = env.reset()
        curr_state = np.reshape(curr_state, [1, state_size])
        total_reward = 0
        complete = False

        print(f'Episode {episode}')

        while not complete:
            curr_action = agent.act(curr_state)
            nxt_state, curr_reward, complete, _ = env.step(curr_action)
            nxt_state = np.reshape(nxt_state, [1, state_size])

            # Store experience
            memory_buffer.add(curr_state, curr_action, curr_reward, nxt_state, complete)

            # Train the agent with a batch from the replay buffer
            if len(memory_buffer) > batch_size:
                if use_prioritized_replay:
                    experiences, indices, weights = memory_buffer.sample(batch_size)
                    agent.train(experiences, indices, weights)
                else:
                    experiences = memory_buffer.sample(batch_size)
                    agent.train(experiences)

            curr_state = nxt_state
            total_reward += curr_reward

        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # No change in the evaluation phase
    total_test_reward = 0
    curr_state = env.reset()
    curr_state = np.reshape(curr_state, [1, state_size])
    complete = False

    while not complete:
        curr_action = agent.act(curr_state)
        nxt_state, curr_reward, complete, _ = env.step(curr_action)
        nxt_state = np.reshape(nxt_state, [1, state_size])
        total_test_reward += curr_reward
        curr_state = nxt_state

    return rewards, agent


# The test function remains largely unchanged.
def test_c51_agent(agent, env, num_episodes=250):
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

    res = {
        'Average Reward': avg_reward,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'Confusion Matrix': confusion
    }
    return res, total_rewards

def visualize_training_results(rewards):
    """
    Visualizes the training results.

    Args:
    - rewards (list): A list of rewards received at each episode.
    """

    # Calculate moving average with window size of 100
    moving_avg = [np.mean(rewards[max(0, i - 100):i + 1]) for i in range(len(rewards))]

    plt.figure(figsize=(10, 5))

    plt.plot(rewards, label='C51 Episode Reward', alpha=0.6)
    plt.plot(moving_avg, label='C51 Moving Average (100 episodes)', color='red')

    plt.title("C51 Training Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

# Same as train_dqn_agent but use C51Agent instead of DQNAgent

if __name__ == '__main__':
    env = IDSEnvironment()
    training_rewards, agent = train_c51_agent(env)  # Train using C51
    visualize_training_results(training_rewards)
    results, test_rewards = test_c51_agent(agent, env)
    # 1. Bar Plot for Metrics
    metrics = ['Average Reward', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
    values = [results[metric] for metric in metrics]

    plt.figure(figsize=(10, 5))
    plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.ylabel('Value')
    plt.title('C51 Metrics Visualization')
    plt.ylim([0, 1])
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()

    # 2. Heatmap for Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['Confusion Matrix'], annot=True, cmap="YlGnBu", fmt='g')
    plt.title('C51 '
              'Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    visualize_training_results(training_rewards)
    print(results)

