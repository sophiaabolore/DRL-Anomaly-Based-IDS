# DRL-Anomaly-Based-IDS
Applying distributional variants to reinforcement learning approaches to anomaly-based IDS
What: 
Anomaly-Based Intrusion Detection Systems (IDS) 
monitor network traffic or system behavior for deviations from established norms or baselines. 
crucial because they can detect unknown threats and zero-day vulnerabilities by identifying unusual patterns, providing a broader view of security health, and reducing reliance on frequent signature updates. 
they can also produce a high number of false positives due to their sensitivity to deviations, require complex setup to establish accurate baselines, 
and tend to be resource-intensive because of constant comparisons against the baseline. 
Despite these challenges, anomaly-based IDS can be a vital component of a comprehensive security strategy.

Past Work 
Past work applying ML and reinforcement learning to anomaly-based intrusion detection
“In this paper we apply DRL using two of these datasets: NSL-KDD and AWID datasets. As a novel approach, we have made a conceptual modification of the classic DRL paradigm (based on interaction with a live environment), replacing the environment with a sampling function of recorded training intrusions. This new pseudo-environment, in addition to sampling the training dataset, generates rewards based on detection errors found during training.
We present the results of applying our technique to four of the most relevant DRL models: Deep Q-Network (DQN), Double Deep Q-Network (DDQN), Policy Gradient (PG) and Actor-Critic (AC). The best results are obtained for the DDQN algorithm.” - Manuel Lopez Martin

The NSL-KDD dataset, has seen varied results in numerous studies: Ingre & Yadav (2015) achieved 81.2% test data accuracy with an MLP model; Ibrahim et al. (2013) saw a 75.49% recall with SOM; while others, such as Dhanabal & Shantharajah (2015) and Kamel et al. (2016), reported high accuracies using SVM, Naïve Bayes, and AdaBoost but lacked clarity on their test sets. Works from Shone et al. (2018), Chen et al. (2018), and Woo, Song & Choi (2019) explored autoencoders and MLPs, attaining accuracies up to 98.5%.
Berman et al. (2019) extensively reviewed the application of diverse neural architectures on NSL-KDD, while Thomas & Pavithran (2018) synthesized 8 works, highlighting a hybrid model's peak accuracy of 99.81%. Other innovative approaches, such as Javaid et al.'s (2016) self-taught learning, delivered an F1 score of 90.4%, and Bhattacharjee, Fujail & Begum (2017) utilized clustering methods, with Fuzzy C-Means yielding the best detection rate at 45.95%.
The AWID dataset, introduced by Kolias et al. (2016), saw varied results from multiple studies: the J48 model performed best with 96% accuracy and an F1 score of 94.8%; while Wang et al. (2019) used Stacked Autoencoder and DNNs, achieving accuracies between 73.1% and 99.9%. Additionally, Abdulhammed et al. (2018) applied feature selection and classic ML models with Random Forest topping at 99.64%, and Rezvy et al. (2019) garnered a 99.9% accuracy using a deep autoencoder.
Qin et al. (2018) employed an SVM-based approach on AWID with reduced features, achieving accuracies between 87.34% and 99.98% for various attacks. Meanwhile, Moshkov (2017) and Thanthrige, Samarabandu & Wang (2016) emphasized the importance of feature selection and model choice, with gradient boosting and Random Tree models standing out in their respective studies, delivering top results on the AWID dataset for wireless network intrusion detection.
Servin (2007) explored reinforcement learning for intrusion detection in a simulated network environment, using a Q-learning algorithm based on a look-up table, differing from more recent DRL models which use neural networks for state generalization.
Multiple studies, including Xu (2010), Sukhanov, Kovalev & Stýskala (2015), and Xu & Xie (2005) applied look-up tables with temporal difference learning for live traffic flow intrusion detection, while Xu (2006) and Xu & Luo (2007) ventured into kernel versions of TD learning and host-based IDS respectively.
Servin (2009) and Malialis (2014) utilized a multi-agent hierarchy for intrusion detection via reinforcement learning, contrasting the single-agent approach presented in the main work.
DRL's application in cyber-physical-systems is highlighted by Feng & Xu (2017) and Akazaki et al. (2018), emphasizing the importance of detecting 'counterexamples' to ensure system security.
Nguyen & Reddi (2019) offered a comprehensive review of DRL in the context of cybersecurity, focusing on models operating in real or simulated live environments.
Wiering et al. (2011) introduced a DRL method using an actor-critic model for classification on UCI datasets, utilizing a unique approach that extends the state space with added variables as memory cells and a complex reward scheme, differing from the current work which uses simpler rewards without needing feature engineering. The following section details the datasets and DRL models used in the current experiments.

MDP
States (S):
Description: Each state in the environment represents a feature vector from the IDS dataset. 
Features are derived from the NSL-KDD dataset which is 
Representation: It's a vector of real numbers with length equal to the number of features (self.num_features). Each entry in the vector can have a value between 0 and 1, as represented by the observation_space attribute.
Total Number of States: 
the number of rows in the train_data
Actions (A):
Description: There are two possible actions, representing the decision of the IDS.
0: No Alert
1: Alert
Representation: 
It's defined by the action_space attribute as a discrete space with 2 actions.
Transitions (P):
Description: Transition dynamics are deterministic in terms of the state sequence (it processes through the dataset sequentially), but the reward has stochastic elements.
Representation: The environment uses self.current_data_pointer to keep track of the current position in the dataset. After each step, it moves to the next data point, and if it reaches the end of the dataset, the episode terminates (complete becomes True).
Rewards (R):
Description: The reward at each step depends on whether the agent's action matches the actual label (intrusion or not) from the dataset.
Representation: The reward is a random number drawn from a normal distribution. The mean (loc) of the distribution is 1 if the agent's action matches the actual label, and -1 otherwise. The standard deviation (scale) of the distribution is 0.1. This introduces variability in the rewards the agent receives, even if it takes the same action in the same state.

DQN Agent 

Policy (π)

- The agent uses an ε-greedy strategy to balance exploration and exploitation. If a random number is less than or equal to ε (self.epsilon), it will take a random action (exploration). Otherwise, it uses the neural network (self.model) to predict Q-values for each action given the current state and then selects the action with the highest Q-value (exploitation).
- The ε value starts high (1.0) and decays over time as per self.epsilon_decay until it reaches a minimum value (self.epsilon_min). This ensures that the agent starts by exploring widely and gradually shifts to exploiting what it has learned.
Q-value Approximation
- The Q-values, which represent the expected future rewards of state-action pairs, are approximated using a neural network.
- The network has an input layer corresponding to the state size, two hidden layers each with 24 neurons and ReLU activation, and an output layer with a neuron for each action. The output layer uses a linear activation function to predict the Q-values.
Learning (Bellman Equation)
- The replay method captures the essence of Q-learning. Here, the agent samples a minibatch of experiences from its memory and updates its Q-values based on the Bellman equation.
- For each experience in the minibatch, the agent computes the target Q-value as:
Q(next_state,a) If the episode has terminated (done is True), the target is simply the reward. This is because there are no future rewards to consider once the episode has ended.
The computed target Q-values are then used to update the neural network model.
Experience Replay
The agent stores experiences (state, action, reward, next_state, done) in its memory (self.memory).
The replay method samples a minibatch of experiences from this memory for learning. This breaks the temporal correlations between sequential experiences and helps stabilize the learning process.
The agent uses the train_on_batch method to update the model weights based on the sampled experiences.
Discount Factor (γ)
The discount factor γ (self.gamma) is set to 0.95. This means the agent gives 95% of the value of future rewards compared to immediate rewards. The closer this value is to 1, the more future-oriented the agent is.
