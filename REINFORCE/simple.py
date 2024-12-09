"""
REINFORCEアルゴリズム
逆に最も単純な勾配降下法の式を実装するのはめんどい
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy
import gymnasium as gym

# 環境の初期化
env = gym.make("CartPole-v1", render_mode="human")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

print(obs_size) # 4 位置、測度、角度、角速度
print(n_actions) # 2 0:左、1:右

# 方策ネットワーク
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(PolicyNetwork, self).__init__() # 親クラスの初期化を呼び出す
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1) # 行動確率を出力 dim=-1で（バッチサイズ、行動の数）の行動の数の次元に沿って確率にしている
        )

    def forward(self, x):
        return self.fc(x)

# ネットワークのインスタンス化
policy_net = PolicyNetwork(obs_size, n_actions)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01) # parametersは学習可能なパラメータを返すメソッド。オプティマイザに渡す目的で使用

# 報酬の割引和を計算する関数
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    # モンテカロル法の効率的な実装
    G = 0
    for r in reversed(rewards):
        G = r + gamma*G
        discounted_rewards.insert(0, G)
    return torch.tensor(discounted_rewards, dtype=torch.float32)
# 実装のイメージ
# [1, 2, 3]
# discounted_rewards = [3]
# discounted_rewards = [4.97, 3] -> 3+0.99*3
# discounted_rewards = [5.9203, 4.97, 3] -> 4.97+0.99*1

# メインループ
n_episodes = 1000
gamma = 0.99

for episode in range(n_episodes):
    state, _ = env.reset()
    log_probs = []
    rewards = []

    # エピソードごとのシュミレーション
    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = policy_net(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample() # 行動をサンプリング
        log_probs.append(action_dist.log_prob(action)) # log(π(a|s)) sでのaのサンプリングのlog確率を保存。

        # 環境を１ステップ進める
        next_state, reward, done, truncated, info = env.step(action.item()) # actionの.itemで実際の行動を取り出している
        done = done or truncated
        rewards.append(reward)
        state = next_state

        # カートポールの動きをレンダリング
        env.render()


    # 割引報酬の計算
    discounted_rewards = compute_discounted_rewards(rewards, gamma)

    # 損失の計算と方策の更新
    # 割引報酬を標準化
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
    loss = -torch.cat([log_prob.unsqueeze(0) for log_prob in log_probs]) * discounted_rewards # log_probsはリストでテンソルに変換する
    loss = loss.sum()

    # lossの計算のイメージ
    # log_probs = [tensor([-0.2]), tensor([-1.5]), tensor([-0.5])]
    # torch.cat => tensor([-0.2, -1.5, -0.5])
    # discounted_rewards = [5.9203, 4.97, 3]のとき
    # loss = -ΣR * logπ(a|s) (t = 1, 2, 3のとき)を計算している

    optimizer.zero_grad() # 前回の勾配をリセット
    loss.backward()        # 損失関数を利用して勾配の計算
    optimizer.step()      # パラメータの更新
    
    # ログの出力
    print(f"Episode {episode + 1}: Total Reward = {sum(rewards)}")

    # 条件を満たしたら終了
    if sum(rewards) >= 195: # CartPoleの基準
        print(f"Solved in {episode + 1} episodes!!!!")
        break

env.close()