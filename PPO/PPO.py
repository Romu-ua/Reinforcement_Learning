import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np


# 環境設定
env = gym.make("BipedalWalker-v3", render_mode="human")
obs_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# Actorネットワーク (π(a|s))
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, x):
        mean = self.fc(x)
        std = torch.exp(self.log_std)
        return mean, std
    
class ValueNetwork(nn.Module):
    def __init__(self, obs_size):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.fc(x)
    
# 初期化関数
# オルソゴナル初期化がtorchの初期化よりも優れているらしい
def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2)) # √2がReLUに適したスケール因子
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

# 3.ネットワークの初期化
policy_net = PolicyNetwork(obs_size, action_size)
value_net = ValueNetwork(obs_size)
policy_net.apply(initialize_weights)
value_net.apply(initialize_weights)

# オプティマイザの初期化
actor_optimizer = optim.Adam(policy_net.parameters(), lr=0.0003)
critic_optimizer = optim.Adam(value_net.parameters(), lr=0.0003)

# 4. 学習率アニーリング
# PPOではAdamを使用するときに、学習率を徐々に下げていくとパフォーマンスが向上するらしい
lr_scheduler = torch.optim.lr_scheduler.LinearLR(actor_optimizer, start_factor=1.0, end_factor=0.0, total_iters=2000)

# 割引率とGAEの計算
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    values = torch.tensor(values + [0], dtype=torch.float32) # 終端状態
    gae = 0
    advantages = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages, dtype=torch.float32)

# ハイパーパラメータ
n_episodes = 1000
gamma = 0.99
epsilon = 0.2
lam = 0.95
epochs = 10
batch_size = 64

# メインループ
for episode in range(n_episodes):
    state, _ = env.reset()
    log_probs = []
    values = []
    rewards = []
    actions = []
    states = []
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        mean, std = policy_net(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, env.action_space.low[0], env.action_space.high[0]) # 確率分布からサンプリングしているからenvの行動の範囲を超えないように
        # 多次元の行動空間を持つときに、[0.6, -0.3]がサンプリングされたとする mean[0.5, -0.2] std[0.1, 0.3]
        # logガウス分布の式より、-0.918, -1.523となる。logπ(a|s) = Σ_次元数 logπ(a_i|s)
        # = logπ(a|s) = -0.918 -1.523 = -2.441
        # dim=-1で次元をつぶして、sumで計算している
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = value_net(state_tensor)
        next_state, reward, done, truncated, _ = env.step(action.numpy())
        done = done or truncated

        # データ収集
        log_probs.append(log_prob)
        values.append(value.item())
        rewards.append(reward)
        actions.append(action)
        states.append(state)
        state = next_state

    # advantege計算
    advantages = compute_gae(rewards, values, gamma, lam)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # 正規化
    values = torch.tensor(values, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    # 学習
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.stack(actions)

    # バッチ学習を開始する前にデータが十分か確認
    if len(states) < batch_size:
        print(f"Skipping update in episode {episode}: Not enough data (len(states) = {len(states)})")
        continue

    for _ in range(epochs):
        for i in range(0, len(states), batch_size):
            batch_states = states_tensor[i:i+batch_size]
            batch_actions = actions_tensor[i:i+batch_size]
            batch_advantages = advantages[i:i+batch_size]
            batch_values = values[i:i+batch_size]
            batch_log_probs = torch.tensor(log_probs[i:i+batch_size]) # log_probsだけリスト、torch tensorにする必要がある

            # 新しい方策 policy_netは更新されているので、ここから求められるlogπ(a|s)はnewの方
            mean, std = policy_net(batch_states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)

            # 方策の更新 PPOのメインの数式
            ratios = (new_log_probs - batch_log_probs.detach()).exp()
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 1. Value Clipping この実装上のテクニックはあってもなくてもいいっぽい
            value_pred = value_net(batch_states).squeeze()
            clipped_value_pred = torch.clamp(value_pred, batch_values - epsilon, batch_values + epsilon)
            value_loss_1 = (value_pred - batch_values).pow(2)
            value_loss_2 = (clipped_value_pred - batch_values).pow(2)
            value_loss = torch.max(value_loss_1, value_loss_2).mean()

            # 最適化ステップ
            actor_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5) # 勾配クリッピングをしている
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
            critic_optimizer.step()

    lr_scheduler.step()

    print(f"Episode {episode + 1}: Total Reward = {sum(rewards):.2f}")

    # if sum(rewards) >= 300:
    #     print(f"Solved in {episode + 1} epidoes !!!!!")
    #     break

env.close()