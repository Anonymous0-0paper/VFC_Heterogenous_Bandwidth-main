import torch
import torch.nn as nn

class Actor(nn.Module):
    """
    شبکه Actor (سیاست) برای هر عامل.
    ورودی: مشاهده محلی عامل.
    خروجی: یک عمل مشخص (Deterministic Action).
    """
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        # از Softmax استفاده می‌کنیم تا خروجی برای انتخاب عمل مناسب باشد
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        # خروجی به صورت احتمالاتی برای هر عمل است
        return self.softmax(self.fc3(x))

class Critic(nn.Module):
    """
    شبکه Critic (ارزش).
    ورودی: مشاهدات و اقدامات تمام عامل‌ها (اطلاعات جهانی).
    خروجی: مقدار Q-value برای این حالت و اقدامات جهانی.
    """
    def __init__(self, global_state_dim, global_action_dim):
        super(Critic, self).__init__()
        # ورودی شامل وضعیت و عمل تمام عامل‌ها است
        self.fc1 = nn.Linear(global_state_dim + global_action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, global_state, global_action):
        # الحاق وضعیت‌ها و عمل‌ها برای ورودی به شبکه
        x = torch.cat([global_state, global_action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)