import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOAgent(nn.Module):
    def __init__(self, obs_dim, n_actions, lr, gamma, clip_range, gae_lambda,
                 entropy_coef, value_loss_coef, batch_size, update_epochs, hidden_size):
        super(PPOAgent, self).__init__()

        self.gamma = gamma
        self.clip_range = clip_range
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.batch_size = batch_size
        self.update_epochs = update_epochs

        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1)
        )

        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs.flatten())
        probs = self.policy_net(obs_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_net(obs_t)
        return action.item(), log_prob, value

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        next_value = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        returns = [a + v for a, v in zip(advantages, values)]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def update(self, trajectory):
        states, actions, log_probs, values, rewards, next_states, dones = zip(*trajectory)

        # Convert về tensor
        states = torch.FloatTensor(np.array([s.flatten() for s in states]))
        actions = torch.tensor(actions)
        old_log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Tính advantage & return
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize
        advantages = advantages.detach()
        returns = returns.detach()

        # PPO update loop
        for _ in range(self.update_epochs):
            idx = np.random.permutation(len(rewards))
            for start in range(0, len(rewards), self.batch_size):
                end = start + self.batch_size
                batch_idx = idx[start:end]

                obs_batch = states[batch_idx]
                action_batch = actions[batch_idx]
                old_log_batch = old_log_probs[batch_idx]
                adv_batch = advantages[batch_idx]
                ret_batch = returns[batch_idx]

                # Tính toán phân phối hành động mới
                probs = self.policy_net(obs_batch)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(action_batch)
                entropy = dist.entropy().mean()

                # Tính ratio giữa policy mới và cũ
                ratio = (new_log_probs - old_log_batch).exp()

                # Clipped surrogate objective
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values_pred = self.value_net(obs_batch).squeeze()
                value_loss = (ret_batch - values_pred).pow(2).mean()

                # Tổng loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate(self, obs):
        with torch.no_grad():
            values = self.value_net(obs)
        return values

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
