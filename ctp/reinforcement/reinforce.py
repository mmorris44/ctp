import numpy as np

import torch
from torch import nn, Tensor
from torch import optim


# Environment
class CTPEnv:
    def __init__(self, n_reformulators, embedding_size):
        self.action_space = list(range(n_reformulators))
        self.observation_size = 3 * embedding_size


# Gradient policy network
class PolicyEstimator:
    def __init__(self, env):
        self.n_inputs = env.observation_size
        self.n_outputs = len(env.action_space)

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))

    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


class ReinforceModule:
    def __init__(self, n_reformulators, embedding_size):
        self.env = CTPEnv(n_reformulators=n_reformulators, embedding_size=embedding_size)
        self.policy_estimator = PolicyEstimator(self.env)
        self.optimizer = optim.Adam(self.policy_estimator.network.parameters(), lr=0.01)

    # Call to get action
    def get_action(self, state: Tensor):
        action_probs = self.policy_estimator.predict(state).detach().numpy()
        action = np.random.choice(self.env.action_space, p=action_probs)
        return action

    # When reward is known, update the policy network
    # Arguments across batches
    def apply_reward(self, state: Tensor, action: Tensor, reward: Tensor):
        self.optimizer.zero_grad()

        # Calculate loss
        logprob = torch.log(self.policy_estimator.predict(state))
        selected_logprobs = reward * torch.gather(logprob, 1, action.unsqueeze(1)).squeeze()
        loss = -selected_logprobs.mean()

        # Calculate gradients
        loss.backward()
        # Apply gradients
        self.optimizer.step()