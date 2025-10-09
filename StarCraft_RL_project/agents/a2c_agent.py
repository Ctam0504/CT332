# agents/a2c_agent_fast.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# ----------------- Model -----------------
class A2CModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2)
        convh = conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2)
        linear_input_size = convw * convh * 32

        self.fc = nn.Linear(linear_input_size, 256)
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.actor(x), self.critic(x)

# ----------------- Agent -----------------
class A2CAgent:
    def __init__(self, input_shape, n_actions, lr=1e-4, gamma=0.99, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = A2CModel(input_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.n_actions = n_actions

    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 3:
            state = state.unsqueeze(0)

        logits, value = self.model(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    # ---- Update từng bước ----
    def update(self, log_prob, value, reward, next_value, done):
        target = reward + self.gamma * next_value * (1 - done)
        advantage = target - value
        actor_loss = -log_prob * advantage.detach()
        critic_loss = F.mse_loss(value, target.detach())
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ---- Batch update (⚡ nhanh hơn nhiều) ----
    def update_batch(self, log_probs, values, rewards, next_value, dones):
        values = torch.cat(values)
        log_probs = torch.stack(log_probs)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)

        next_value = next_value.detach()
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        returns = torch.cat(returns).detach()

        advantages = returns - values
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
