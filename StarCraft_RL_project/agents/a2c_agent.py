# ===================== agents/a2c_agent.py =====================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Linear(conv_out_size, 512)
        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

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
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

class A2CAgent:
    def __init__(self, input_shape, n_actions, lr=1e-4, gamma=0.99, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.model = ActorCritic(input_shape, n_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits, _ = self.model(state)
        prob = F.softmax(logits, dim=-1)
        action = torch.multinomial(prob, num_samples=1)
        return action.item(), prob[:, action.item()].item()

    def update(self, trajectory):
        R = 0
        saved_log_probs = []
        values = []
        rewards = []

        for step in reversed(trajectory):
            state, action, reward, next_state, done = step
            R = reward + self.gamma * R * (1 - done)
            rewards.insert(0, R)

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits, value = self.model(state_tensor)
            prob = F.softmax(logits, dim=-1)
            log_prob = torch.log(prob[0, action])

            saved_log_probs.insert(0, log_prob)
            values.insert(0, value)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        values = torch.cat(values).squeeze()
        saved_log_probs = torch.stack(saved_log_probs)

        advantage = rewards - values
        actor_loss = -(saved_log_probs * advantage.detach()).mean()
        critic_loss = F.mse_loss(values, rewards)
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()