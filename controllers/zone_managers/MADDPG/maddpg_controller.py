import torch
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random

from torch import nn

from controllers.zone_managers.MADDPG.maddpg_actor_critic import Actor, Critic


class MADDPGController:
    """
    کنترلر مرکزی که تمام عامل‌ها را مدیریت و آموزش می‌دهد.
    """

    def __init__(self, num_agents, state_dims, action_dims, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=1e-3):
        self.num_agents = num_agents
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        # ابعاد کلی برای Critic
        global_state_dim = sum(state_dims)
        global_action_dim = sum(action_dims)

        # ایجاد Actor و Critic برای هر عامل
        self.agents_actors = [Actor(sd, ad).to(self.device) for sd, ad in zip(state_dims, action_dims)]
        self.agents_critics = [Critic(global_state_dim, global_action_dim).to(self.device) for _ in range(num_agents)]

        # ایجاد شبکه‌های هدف (Target Networks)
        self.target_actors = [Actor(sd, ad).to(self.device) for sd, ad in zip(state_dims, action_dims)]
        self.target_critics = [Critic(global_state_dim, global_action_dim).to(self.device) for _ in range(num_agents)]

        # کپی وزن‌ها به شبکه‌های هدف
        for i in range(num_agents):
            self.target_actors[i].load_state_dict(self.agents_actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.agents_critics[i].state_dict())

        # بهینه‌سازها
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.agents_actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=lr_critic) for critic in self.agents_critics]

        # حافظه تجربیات مشترک
        self.memory = deque(maxlen=100000)
        self.experience = namedtuple("Experience",
                                     field_names=["global_states", "global_actions", "rewards", "global_next_states",
                                                  "dones"])

    def select_actions(self, states, exploration_noise=0.1):
        """
        انتخاب عمل برای تمام عامل‌ها بر اساس مشاهدات محلی.
        states: لیستی از مشاهدات برای هر عامل.
        """
        actions = []
        for i, state in enumerate(states):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs = self.agents_actors[i](state_tensor)

            # در حین آموزش، نویز برای کاوش اضافه می‌شود
            if exploration_noise > 0:
                # انتخاب عمل بر اساس توزیع احتمال
                action = torch.distributions.Categorical(action_probs).sample()
            else:
                # در زمان تست، بهترین عمل انتخاب می‌شود
                action = torch.argmax(action_probs, dim=1)

            actions.append(action.item())
        return actions

    def store_experience(self, states, actions, rewards, next_states, dones):
        """ذخیره تجربه مشترک در حافظه."""
        global_states = np.concatenate(states)
        global_actions = np.concatenate([np.eye(self.action_dims[i])[actions[i]] for i in range(self.num_agents)])
        global_next_states = np.concatenate(next_states)

        exp = self.experience(global_states, global_actions, rewards, global_next_states, dones)
        self.memory.append(exp)

    def train(self, batch_size=256):
        if len(self.memory) < batch_size:
            return

        experiences = random.sample(self.memory, batch_size)
        batch = self.experience(*zip(*experiences))

        global_states = torch.FloatTensor(np.vstack(batch.global_states)).to(self.device)
        global_actions = torch.FloatTensor(np.vstack(batch.global_actions)).to(self.device)
        rewards = torch.FloatTensor(batch.rewards).to(self.device)
        global_next_states = torch.FloatTensor(np.vstack(batch.global_next_states)).to(self.device)
        dones = torch.FloatTensor(batch.dones).to(self.device)

        # تفکیک مشاهدات و اقدامات برای هر عامل
        # Note: This part needs careful slicing based on state_dims and action_dims
        # For simplicity, assuming all agents have same dims for now

        # --- آموزش برای هر عامل ---
        for i in range(self.num_agents):
            # --- به‌روزرسانی Critic ---
            # محاسبه اقدامات بعدی با شبکه‌های هدف Actor
            next_actions = []
            start_idx = 0
            for j in range(self.num_agents):
                end_idx = start_idx + self.state_dims[j]
                agent_next_state = global_next_states[:, start_idx:end_idx]
                next_actions.append(self.target_actors[j](agent_next_state))
                start_idx = end_idx
            global_next_actions = torch.cat(next_actions, dim=1)

            # محاسبه مقدار Q هدف
            with torch.no_grad():
                target_q = rewards[:, i].unsqueeze(1) + self.gamma * self.target_critics[i](global_next_states,
                                                                                            global_next_actions) * (
                                       1 - dones[:, i].unsqueeze(1))

            # مقدار Q فعلی
            current_q = self.agents_critics[i](global_states, global_actions)

            critic_loss = nn.MSELoss()(current_q, target_q)
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # --- به‌روزرسانی Actor ---
            # محاسبه اقدامات فعلی با شبکه‌های اصلی Actor
            actions_pred = []
            start_idx = 0
            for j in range(self.num_agents):
                end_idx = start_idx + self.state_dims[j]
                agent_state = global_states[:, start_idx:end_idx]
                actions_pred.append(self.agents_actors[j](agent_state))
                start_idx = end_idx
            global_actions_pred = torch.cat(actions_pred, dim=1)

            actor_loss = -self.agents_critics[i](global_states, global_actions_pred).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # --- به‌روزرسانی نرم شبکه‌های هدف ---
        self._soft_update_target_networks()

    def _soft_update_target_networks(self):
        for i in range(self.num_agents):
            for target_param, local_param in zip(self.target_actors[i].parameters(),
                                                 self.agents_actors[i].parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

            for target_param, local_param in zip(self.target_critics[i].parameters(),
                                                 self.agents_critics[i].parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)