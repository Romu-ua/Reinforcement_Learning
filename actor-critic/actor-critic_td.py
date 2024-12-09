"""
式(9.6)の実装
TD法に拡張したので、連続タスクにする。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

env = gym.make("MountainCarContinuous-v0", render_mode="human")
obs_size = env.observation_space.shape[0] # 2  車の位置、車の速度
action_size = env.action_space.shape[0]   # 1  -1~1の連続値でスカラー


# Actorネットワーク(連続行動なので、平均と標準偏差を出力)
# ネットワークで平均を求めて、学習可能な固有のパラメータとしてstdを学習している
# 学習可能な固有のパラメータのイメージ
# input(state) -> hidden layers -> output (mean)
#                       \
#                         --------> Fixed Parameter(log_std)
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        self.log_std = nn.Parameter(torch.zeros(action_size)) # 対数を返り値として設定しているのは、標準偏差は正の値しか意味を持たないから

        
    def forward(self, x):
        mean = self.fc(x)
        std = torch.exp(self.log_std) # 対数なので元に戻す
        return mean, std


# Criticネットワーク
class ValueNetwork(nn.Module):
    def __init__(self, obs_size):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1) 
        )

    def forward(self, x):
        return self.fc(x)
    
# ネットワークの初期化
policy_net = PolicyNetwork(obs_size, action_size)
value_net = ValueNetwork(obs_size)

# オプティマイザ
actor_optimizer = optim.Adam(policy_net.parameters(), lr=0.0003)
critic_optimizer = optim.Adam(value_net.parameters(), lr=0.001)


def compute_td_target(reward, next_value, gamma, done):
    return reward + (1 - done) * gamma*next_value

n_steps = 500000
gamma = 0.99

state, _ = env.reset()
step = 0

while step < n_steps:
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # Actor:行動の平均と標準偏差を計算
    mean, std = policy_net(state_tensor)
    dist = torch.distributions.Normal(mean, std) # 正規分布として行動を定義
    action = dist.sample()
    log_prob = dist.log_prob(action).sum() # アクションの確率密度π(a1,a2|s)=π(a1|s)*π(a2|s) 対数確率logπ(a1,a2|s)=π(a1|s)+π(a2|s)

    #critic: 状態の価値を計算
    value = value_net(state_tensor)

    # 環境を1ステップ進める
    next_state, reward, done, truncated, _ = env.step(action.detach().numpy()) # テンソルは計算グラフに存在しているから計算グラフから切り離す。スカラーかnumpy配列を受け取れる
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

    env.render()

    # TDターゲットを計算
    # withで一時的に計算グラフから切り離して計算することができる
    with torch.no_grad(): 
        next_value = 0 if done else value_net(next_state_tensor).item()
    td_target = compute_td_target(reward, next_value, gamma, done)
    advantage = td_target - value.item()
    # advantages = (torch.tensor([advantage]) - torch.tensor([advantage]).mean()) / (torch.tensor([advantage]).std() + 1e-9) # 標準化

    # Actorの損失
    actor_loss = -log_prob * advantage

    # Criticの損失
    critic_loss = nn.functional.mse_loss(value, torch.tensor([td_target], dtype=torch.float32))

    # Actorの更新
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Criticの更新
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # 状態を次の状態にして更新
    state = next_state
    step += 1

    # 連続条件なので、エピソードが終わったら次のエピソード
    if done or truncated:
        state, _ = env.reset()

    if step % 200 == 0:
        print(f"Step {step}: Actor Loss = {actor_loss.item()}, Critic Loss = {critic_loss.item()}")
        