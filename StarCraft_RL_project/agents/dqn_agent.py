import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNNetwork, self).__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Linear(conv_out_size, 512)
        self.head = nn.Linear(512, n_actions)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(o.view(1, -1).size(1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.head(x)

class DQNAgent:
    def __init__(self, input_shape, n_actions, lr=0.0005, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=10000,
                 batch_size=32, memory_size=10000):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.policy_net = DQNNetwork(input_shape, n_actions).to(device)
        self.target_net = DQNNetwork(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.steps_done = 0

    def select_action(self, state):
        self.steps_done += 1
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(-1. * self.steps_done / self.eps_decay)
        if random.random() < eps_threshold:
            return random.randrange(self.n_actions)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).squeeze(2).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).squeeze(2).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def save(self, path):
        """Lưu trọng số model vào file .pth"""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """Tải trọng số model từ file .pth"""
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()


