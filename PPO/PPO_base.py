import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

# 環境設定 
env = gym.make("MountainCarContinuous-v0", render_mode="human") # env.stepで自動的にrenderが呼ばれる
obs_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# Actorネットワーク(方策π(a|s))
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size), # action_sizeはスカラー。つまり、分布の平均値を出力している
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
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 状態の価値はスカラー
        )
    
    def forward(self, x):
        return self.fc(x)
    

# ネットワークの初期化
policy_net = PolicyNetwork(obs_size, action_size)
value_net = ValueNetwork(obs_size)

#オプティマイザの初期化
actor_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
critic_optimizer = optim.Adam(value_net.parameters(), lr = 0.001)

# 割引報酬の計算
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        discounted_rewards.insert(0, G)
    return torch.tensor(discounted_rewards, dtype=torch.float32)

# パイパーパラメータ
n_episodes = 1000
gamma= 0.99
epsilon = 0.2 # クリッピング範囲　π_oldとπ_newの比率が0.8 ~ 1.2の間に収まるようにする
epochs = 4 # それぞれのバッチを４回繰り返して学習する
batch_size = 64

# メインループ
for episode in range(n_episodes):
    # 数式との対応 logπ_θ(a|s), V_φ(s), R, a, s
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
        log_prob = dist.log_prob(action) # torchを使って確率分布にしている利点

        value = value_net(state_tensor)
        next_state, reward, done, truncated, _ = env.step(action.numpy())
        done = done or truncated

        # データ収集
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        actions.append(action)
        states.append(state)
        state = next_state

    # 割引報酬とAdvantageの計算
    discount_rewards = compute_discounted_rewards(rewards)
    values = torch.cat(values).squeeze() # shapeを整えている(5,1)->(5,)
    advantages = discount_rewards - values.detach()

    # 学習 (バッチをepochsの数だけ使う)
    states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
    actions_tensor = torch.cat(actions)
    for _ in range(epochs):
        for i in range(0, len(states), batch_size):
            batch_states = states_tensor[i:i+batch_size]
            batch_actions = actions_tensor[i:i+batch_size]
            batch_advantages = advantages[i:i+batch_size]
            batch_rewards = discount_rewards[i:i+batch_size]
            batch_log_probs = torch.stack(log_probs[i:i+batch_size])

            # 新しい方策
            mean, std = policy_net(batch_states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1) # 行動がスカラーの時はdim=-1は必要ないが多次元の時には(batch_size, action_dim)->(batch_size,)にする

            # 方策の更新
            ratios = (new_log_probs - batch_log_probs.detach()).exp() # 数式を見るとπ_oldが下 batch_log_probsを使って計算するときは計算グラフから外す 
            # batchの中でこれだけtorchの計算グラフに乗っているから、ここでの計算で変数のバージョンが更新されてしまったからテンソルがインプレースで変更された、というエラーが発生した。
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean() # 期待値が目的関数だから.meanをしている　torchは最適化ステップで最小化アルゴリズムを使っているから期待値の最大化ではなくて最小化の問題にしたいからマイナスを付けている

            # 価値関数の更新
            value_loss = nn.functional.mse_loss(value_net(batch_states).squeeze(), batch_rewards) # batch_rewardsはdiscounted_rewardsから取っているので大丈夫

            # 最適化ステップ
            actor_optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            policy_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            value_loss.backward()
            critic_optimizer.step()

    print(f"Episode {episode + 1}: Total Reward = {sum(rewards):.2f}")

    if sum(rewards) >= 90:
        print(f"Solved in {episode + 1} episodes!!!!")
        break

env.close()

        

