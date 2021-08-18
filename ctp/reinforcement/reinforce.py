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
# TODO: come up with good network architecture
class PolicyEstimator:
    def __init__(self, env):
        self.n_inputs = env.observation_size
        self.n_outputs = len(env.action_space)

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 30),  # Used to be 16
            nn.ReLU(),
            nn.Linear(30, self.n_outputs),
            nn.LogSoftmax(dim=-1))

    def predict(self, state: torch.FloatTensor):
        action_probs = self.network(state)
        return action_probs


class ReinforceModule:
    def __init__(self, n_reformulators: int, embedding_size: int, n_actions_selected: int, lr: float,
                 gradient_clip: float, use_rl: bool):
        self.env = CTPEnv(n_reformulators=n_reformulators, embedding_size=embedding_size)
        self.policy_estimator = PolicyEstimator(self.env)
        self.optimizer = optim.Adam(self.policy_estimator.network.parameters(), lr=lr)
        self.n_actions_selected = n_actions_selected
        self.mode = 'train'  # 'train' or 'test'
        self.gradient_clip = gradient_clip
        self.use_rl = use_rl

    # Call to get action
    # Argument is a batch of states
    def get_actions(self, state: Tensor):
        batch_size = state.shape[0]
        action_counts = [0] * len(self.env.action_space)  # Count the number of times each action is selected

        prediction = self.policy_estimator.predict(state)
        action_probs = torch.exp(prediction).detach().cpu().numpy()
        actions = np.zeros(shape=(action_probs.shape[0], self.n_actions_selected), dtype=int)
        for i in range(batch_size):
            try:
                actions[i] = np.random.choice(a=self.env.action_space, size=self.n_actions_selected,
                                              replace=False, p=action_probs[i])
                for action in actions[i]:
                    action_counts[action] += 1
            except ValueError as error:  # ValueError: Fewer non-zero entries in p than size
                print("Confident action distribution found:", action_probs[i], " | ", error)
                # Sample again but with replacement
                actions[i] = np.random.choice(a=self.env.action_space, size=self.n_actions_selected,
                                              replace=True, p=action_probs[i])

                for action in np.unique(actions[i]):
                    action_counts[action] += 1
        return actions, action_counts

    # When reward is known, update the policy network
    # Arguments across batches
    def apply_reward(self, state: Tensor, action: Tensor, reward: Tensor):
        self.optimizer.zero_grad()

        # Calculate loss
        logprob = self.policy_estimator.predict(state)
        selected_logprobs = reward * torch.gather(logprob, 1, action.unsqueeze(1)).squeeze()
        loss = -selected_logprobs.mean()

        # Calculate gradients
        loss.backward(retain_graph=True)  # retain_graph slows it down a lot, but seems necessary to not discard tensors

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_estimator.network.parameters(), self.gradient_clip)

        # Apply gradients
        self.optimizer.step()
