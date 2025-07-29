# ddpg_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import copy

from config import Config


# --- تعریف شبکه Actor ---
# ورودی: حالت (State) ؛ خروجی: عمل (Action)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_4 = nn.Linear(64, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        # استفاده از tanh برای محدود کردن خروجی بین -1 و 1 و سپس ضرب در max_action
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x


# --- تعریف شبکه Critic ---
# ورودی: حالت (State) و عمل (Action) ؛ خروجی: Q-value
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # لایه اول فقط حالت را پردازش می‌کند
        self.layer_1 = nn.Linear(state_dim, 128)
        # لایه دوم، حالت پردازش شده و عمل خام را با هم ترکیب می‌کند
        self.layer_2 = nn.Linear(128 + action_dim, 128)
        # لایه خروجی
        self.layer_3 = nn.Linear(128, 1)

    def forward(self, state, action):
        # 1. ابتدا state را از لایه اول عبور می‌دهیم
        state_out = torch.relu(self.layer_1(state))

        # 2. سپس خروجی مرحله قبل را با action خام ترکیب می‌کنیم
        concat = torch.cat([state_out, action], 1)

        # 3. داده ترکیب‌شده را از لایه‌های بعدی عبور می‌دهیم
        out = torch.relu(self.layer_2(concat))
        out = self.layer_3(out)

        return out


# --- تعریف کلاس عامل DDPG ---
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=0.001, gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # شبکه‌های اصلی
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # شبکه‌های هدف (Target Networks)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # پارامترها
        self.gamma = gamma
        self.tau = tau  # برای soft update
        self.max_action = max_action

        # بافر تجربه (Replay Buffer)
        self.memory = deque(maxlen=10000)

    def select_action(self, state, exploration_noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()

        # افزودن نویز برای اکتشاف
        noise = np.random.normal(0, self.max_action * exploration_noise, size=action.shape)
        action = (action + noise).clip(-self.max_action, self.max_action)

        return action

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        # نمونه‌برداری از بافر
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).reshape(-1, 1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- به‌روزرسانی Critic ---
        with torch.no_grad():
            # انتخاب عمل برای حالت بعدی توسط target actor
            next_actions = self.target_actor(next_states)
            # محاسبه Q-value هدف توسط target critic
            target_q = self.target_critic(next_states, next_actions)
            # محاسبه y (مقدار هدف نهایی)
            target_q = rewards + (self.gamma * (1 - dones) * target_q)

        # Q-value فعلی
        current_q = self.critic(states, actions)

        # محاسبه خطای Critic
        critic_loss = nn.MSELoss()(current_q, target_q)

        # به‌روزرسانی شبکه Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- به‌روزرسانی Actor ---
        # محاسبه خطای Actor (هدف، بیشینه کردن Q-value است)
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # به‌روزرسانی شبکه Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- به‌روزرسانی شبکه‌های هدف (Soft Update) ---
        self._soft_update_target_networks()

    def _soft_update_target_networks(self):
        # به‌روزرسانی تدریجی وزن‌های شبکه‌های هدف
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, filename="ddpg_model"):
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")

    def load_model(self, filename="ddpg_model"):
        self.actor.load_state_dict(torch.load(f"{filename}_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{filename}_critic.pth", map_location=self.device))
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)