"""
式(9.5)の実装
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# Actorネットワーク（方策）
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

# Criticネットワーク(価値関数)
class ValueNetwork(nn.Module):
    def __init__(self, obs_size):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 状態の価値はスカラー
        )
    
    def forward(self, x):
        return self.fc(x)
    
# 方策はπ(a|s)で価値関数もV(s)なので、forward(self, x)になるのが一瞬こんがらがる。
policy_net = PolicyNetwork(obs_size, n_actions)
value_net = ValueNetwork(obs_size)

actor_optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
critic_optimizer = optim.Adam(value_net.parameters(), lr=0.01)

def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        discounted_rewards.insert(0, G)
    return torch.tensor(discounted_rewards, dtype=torch.float32)

n_episodes = 1000
gamma = 0.99

for episode in range(n_episodes):
    state, _ = env.reset()
    log_probs = []
    values = []
    rewards = []
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = policy_net(state_tensor) # 行動の確率を計算
        value = value_net(state_tensor)

        action_dist = torch.distributions.Categorical(action_probs) # 確率分布オブジェクトにする
        action = action_dist.sample() # 確率分布からサンプリング

        log_probs.append(action_dist.log_prob(action)) # 選択したactionの対数確率を確率分布から計算
        values.append(value)

        next_state, reward, done , truncated, info = env.step(action.item())
        done = done or truncated
        rewards.append(reward)
        state = next_state
        
        env.render()

    discounted_rewards = compute_discounted_rewards(rewards, gamma)

    # Actorの損失
    values = torch.cat(values)
    advantages = discounted_rewards - value.squeeze() # A(s,a) = Q(s,a) - V(s)だが、Q(s,a)をGに置き換えたものをAとして考えている。数式のベースラインを計算している
    actor_loss = -torch.cat([log_prob.unsqueeze(0) for log_prob in log_probs]) * advantages.detach() # detach()でAを計算グラフから切り離すことで勾配計算にＡを入れない。
    actor_loss = actor_loss.sum()

    # Criticの損失
    critic_loss = nn.functional.mse_loss(value.squeeze(), discounted_rewards) # 価値関数の推定値Vを得られた割引報酬に近づけるように学習する。

    # Actorの更新   
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Criticの更新
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    print(f"Episode {episode + 1}: Total Reward = {sum(rewards)}")

    if sum(rewards) >= 195:
        print(f"Solved in {episode + 1} episodes!!!!")
        break


env.close()

    
