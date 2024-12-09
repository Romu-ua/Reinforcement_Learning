"""
複数のエピソードまたは時間ステップのデータを収集
Advantegeの計算
ActorとCriticの同時更新
"""
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

env = gym.make("MountainCarContinuous-v0", render_mode="human") # env.stepで自動的に描画
obs_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# Actorネットワーク
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Tanh() # envよりactionは-1~1の間に収まるので、Tanhにしている
        )
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, x):
        mean = self.fc(x)
        std = torch.exp(self.log_std)
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

# TDターゲット
# Gtを求めるのをr + V(s)にしているだけ。
# reversedで値を取っている。最初はnext_valueでリストに入っていない値が必要だが、それ以降はvaluesのリストを後ろから使っていけばよい。
def compute_td_target(rewards, values, next_value, gamma, dones):
    td_target = []
    for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
        target = r + gamma * next_value * (1 - done)
        td_target.insert(0, target)
        next_value = v
    return torch.tensor(td_target, dtype=torch.float32)

n_steps = 1000000
gamma = 0.99
batch_size = 10 # 10ステップごとに更新

state, _ = env.reset()
states = []
actions = []
log_probs = []
values = []
rewards = []
dones = []
step = 0

while step < n_steps:
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # Actor　行動の平均と標準偏差を計算
    mean, std = policy_net(state_tensor)
    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum()

    # Critic 状態の価値を計算
    value = value_net(state_tensor)

    # 環境を1ステップ進める
    next_state, reward, done, truncated, _ = env.step(action.detach().numpy())
    done = done or truncated
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

    # 経験を保存
    states.append(state_tensor)
    actions.append(action)
    log_probs.append(log_prob.unsqueeze(0)) # スカラーらしいので、1次元のテンソルにする
    values.append(value)
    rewards.append(reward)
    dones.append(done)

    # 次の状態に更新
    state = next_state
    step += 1

    # バッチが満たされたら更新
    if len(rewards) >= batch_size:
        with torch.no_grad():
            next_value = 0 if done else value_net(next_state_tensor).item()

        # TDターゲットとAdvantageを計算
        td_targets = compute_td_target(rewards, values, next_value, gamma, dones)
        values_tensor = torch.cat(values).squeeze()
        advantages = td_targets - values_tensor

        # Actorの損失
        log_probs_tensor = torch.cat(log_probs)
        actor_loss = -(log_probs_tensor * advantages.detach()).mean() # 期待値を計算するが実際には無理なので平均値で近似している

        # Criticの損失
        critic_loss = nn.functional.mse_loss(values_tensor, td_targets)

        # Actorの更新
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Criticの更新
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # バッファのリセット
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        print(f"Step {step}: Actor Loss = {actor_loss.item()}, Critic Loss = {critic_loss.item()}")

    # エピソードが終了したときにリセット
    if done:
        state, _ = env.reset()

env.close()
    
