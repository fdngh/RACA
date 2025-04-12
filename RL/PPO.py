import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli
import numpy as np


class PPO(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        super(PPO, self).__init__()

        # Actor network: predicts action probabilities
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Sigmoid activation for binary actions (0-1 probabilities)
        ).to(device)

        # Critic network: estimates state values
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single value output for state value
        ).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.lmbda = lmbda  # GAE-Lambda parameter
        self.epochs = epochs  # Number of epochs per update
        self.eps = eps  # Clipping parameter
        self.device = device

        # Move model to device
        self.to(device)

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

    def take_action(self, state):
        state = state.float().to(self.device)
        probs = self.actor(state)
        action = torch.bernoulli(probs)  # Sample binary actions from probability distribution
        return action

    def min_max_normalize_rewards(self, transition_dict, B):
        rewards = torch.stack(transition_dict['rewards']).float().to(self.device)
        rewards = rewards.repeat(1, B)

        min_reward = torch.min(rewards)
        max_reward = torch.max(rewards)

        if max_reward == min_reward:
            normalized_rewards = torch.ones_like(rewards)
        else:
            normalized_rewards = (rewards - min_reward) / (max_reward - min_reward)
        return normalized_rewards

    def compute_advantage(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Initialize with bootstrapped value for last step
        running_returns = next_values[-1] * (1 - dones[-1])
        running_advs = 0

        # Compute advantages and returns in reverse order (from last to first timestep)
        for t in reversed(range(len(rewards))):
            # Compute discounted return (value target)
            running_returns = rewards[t] + self.gamma * running_returns * (1 - dones[t])
            returns[t] = running_returns

            # Compute TD error
            td_error = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]

            # Compute GAE
            running_advs = td_error + self.gamma * self.lmbda * running_advs * (1 - dones[t])
            advantages[t] = running_advs

        return advantages, returns

    def update(self, transition_dict):
        batch_size = 500
        total_steps = len(transition_dict['states'])

        # Process data in batches to prevent memory issues
        for start in range(0, total_steps, batch_size):
            end = min(start + batch_size, total_steps)

            # Prepare batch data
            states = torch.stack(transition_dict['states'][start:end]).squeeze(1).float().to(
                self.device).requires_grad_(True)
            _, B, ddd = states.shape
            states = states.view(-1, ddd)

            actions = torch.stack(transition_dict['actions'][start:end]).squeeze(1).float().to(self.device)
            _, aB, addd = actions.shape
            actions = actions.view(-1, addd)

            rewards = torch.stack(transition_dict['rewards']).float().to(self.device)
            rewards = rewards.repeat(1, B)
            rewards = rewards.view(-1, 1)

            next_states = torch.stack(transition_dict['next_states'][start:end]).squeeze(1).float().to(self.device)
            next_states = next_states.view(-1, ddd)

            dones = torch.stack(transition_dict['dones'][start:end]).float().to(self.device)
            dones = dones.unsqueeze(1)
            dones = dones.repeat(1, B)
            dones = dones.view(-1, 1)

            # Compute advantage estimates with GAE
            with torch.no_grad():
                values = self.critic(states)
                next_values = self.critic(next_states)
                advantages, returns = self.compute_advantage(rewards, values, next_values, dones.unsqueeze(1))

                # Normalize advantages for training stability
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update loop
            for _ in range(self.epochs):
                # Get current policy and value estimates
                action_probs, current_values = self.forward(states)
                dist = Bernoulli(action_probs)

                # Compute log probabilities for ratio calculation
                new_log_probs = dist.log_prob(actions).sum(1)
                with torch.no_grad():
                    old_log_probs = Bernoulli(self.actor(states)).log_prob(actions).sum(1)

                # Calculate policy ratio and PPO clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages.squeeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantages.squeeze(-1)

                # Calculate losses
                actor_loss = -torch.min(surr1, surr2).mean()  # PPO clipped objective
                critic_loss = F.mse_loss(current_values, returns)  # Value function loss

                # Update actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                # Update critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()