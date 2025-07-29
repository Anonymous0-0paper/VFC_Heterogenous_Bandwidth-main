import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np

# برای جلوگیری از خطاهای احتمالی در محاسبات GAE
torch.autograd.set_detect_anomaly(True)


class ActorCritic(nn.Module):
    """
    شبکه عصبی Actor-Critic برای PPO.
    این شبکه دو خروجی دارد:
    1. خروجی Actor: توزیع احتمال روی عمل‌ها (Policy)
    2. خروجی Critic: ارزش وضعیت فعلی (Value)
    """

    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # لایه‌های مشترک
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # سر Actor (برای سیاست)
        self.actor_head = nn.Linear(128, action_dim)

        # سر Critic (برای ارزش)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state):
        x = self.shared_layers(state)

        # محاسبه توزیع احتمال عمل‌ها
        action_probs = torch.softmax(self.actor_head(x), dim=-1)

        # محاسبه ارزش وضعیت
        state_value = self.critic_head(x)

        return action_probs, state_value


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=4, clip_epsilon=0.2,
                 gae_lambda=0.95):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.K_epochs = K_epochs  # تعداد تکرار آموزش روی یک بچ از داده
        self.gae_lambda = gae_lambda

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # PPO از یک حافظه موقت برای ذخیره تجربیات یک دوره (rollout) استفاده می‌کند
        self.memory = []

        # ساخت مدل‌های Actor-Critic
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.shared_layers.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic}
        ])

        # یک شبکه قدیمی برای محاسبه نسبت سیاست‌ها
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.criterion = nn.MSELoss()  # برای Critic

    def select_action(self, state):
        """
        انتخاب عمل بر اساس سیاست فعلی
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action_probs, state_val = self.policy_old(state_tensor)

            # ایجاد یک توزیع دسته‌ای برای نمونه‌برداری از عمل
            dist = Categorical(action_probs)
            action = dist.sample()

            action_log_prob = dist.log_prob(action)

        # این اطلاعات برای ذخیره در حافظه و آموزش لازم است
        return action.item(), action_log_prob.cpu().numpy().flatten(), state_val.cpu().numpy().flatten()

    def store_experience(self, state, action, reward, done, log_prob, value):
        """
        ذخیره یک تجربه در حافظه موقت
        """
        self.memory.append((state, action, reward, done, log_prob, value))

    def update(self):
        """
        آموزش عامل با استفاده از داده‌های جمع‌آوری شده در حافظه
        """
        # محاسبه پاداش‌ها و مزیت‌ها (Advantages)
        rewards = []
        discounted_reward = 0

        # استخراج مقادیر از حافظه
        states, actions, old_log_probs, old_values, dones = [], [], [], [], []
        for state, action, reward, done, log_prob, value in self.memory:
            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob)
            old_values.append(value)
            dones.append(done)
            rewards.append(reward)

        # محاسبه پاداش‌های تجمعی (Returns) و مزیت‌ها (GAE)
        returns = []
        advantages = []
        last_gae_lam = 0
        for i in reversed(range(len(rewards))):
            # اگر اپیزود تمام شده، مقدار وضعیت بعدی صفر است
            next_non_terminal = 1 - dones[i]
            next_value = old_values[i + 1] if i + 1 < len(old_values) else 0

            delta = rewards[i] + self.gamma * next_value * next_non_terminal - old_values[i]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages.insert(0, last_gae_lam)
            returns.insert(0, last_gae_lam + old_values[i])

        # تبدیل به تنسور
        old_states = torch.FloatTensor(np.array(states)).to(self.device)
        old_actions = torch.LongTensor(np.array(actions)).to(self.device).view(-1, 1)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device).view(-1, 1)
        advantages = torch.FloatTensor(advantages).to(self.device).view(-1, 1)
        returns = torch.FloatTensor(returns).to(self.device).view(-1, 1)

        # نرمال‌سازی مزیت‌ها برای پایداری
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # آموزش برای K اپوک
        for _ in range(self.K_epochs):
            # ارزیابی مجدد با سیاست فعلی
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)

            new_log_probs = dist.log_prob(old_actions.squeeze())
            dist_entropy = dist.entropy()

            # محاسبه نسبت سیاست‌ها (r_t)
            ratios = torch.exp(new_log_probs.view(-1, 1) - old_log_probs.view(-1, 1))

            # محاسبه لاس تابع هدف PPO (Actor Loss)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2)

            # محاسبه لاس Critic (Value Loss)
            critic_loss = self.criterion(state_values, returns)

            # لاس نهایی
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.view(-1, 1)

            # به‌روزرسانی وزن‌ها
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # کپی کردن وزن‌های جدید در شبکه قدیمی
        self.policy_old.load_state_dict(self.policy.state_dict())

        # پاک کردن حافظه برای دوره بعدی
        self.memory.clear()

    def save_model(self, filename="ppo_model.pth"):
        torch.save(self.policy.state_dict(), filename)

    def load_model(self, filename="ppo_model.pth"):
        self.policy.load_state_dict(torch.load(filename, map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())