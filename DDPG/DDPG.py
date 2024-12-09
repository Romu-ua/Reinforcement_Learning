import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from collections import deque

env = gym.make("MountainCarContinuous-v0", render_mode="human")
obs_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

gamma = 0.99
tau = 0.0005 # ターゲットネットワークを更新する比率
batch_size = 64
buffer_size = int(1e6)
learning_rate_actor = 0.001
learning_rate_critic = 0.001
n_steps = 100000

# リプレイバッファ
# バッファに経験を溜めてランダムサンプリングして返す
class ReplayBuffer:
	def __init__(self, size):
		self.buffer = deque(maxlen=size)

	def add(self, state, action, reward, next_state, done):
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)
		return (
			torch.tensor(np.array(states), dtype=torch.float32),
			torch.tensor(np.array(actions), dtype=torch.float32),
			torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
			torch.tensor(np.array(next_states), dtype=torch.float32),
			torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1),
		)

	def __len__(self):
		return len(self.buffer)

# random.sampleでサンプルされるデータ構造は以下。batch_size = 2の時。
# batch = [
# 	([1.0, 0.5], [0.1], 1.0, [1.0, 0.6], False),
# 	([2.0, 1.5], [0.2], 0.5, [2.1, 1.6], False),
# ]
# zip(*batch)で以下のデータ構造にして、これをそれぞれの変数に代入している
# (
#     ([1.0, 0.5], [2.0, 1.5]),  # states
#     ([0.1], [0.2]),            # actions
#     (1.0, 0.5),                # rewards
#     ([1.1, 0.6], [2.1, 1.6]),  # next_states
#     (False, False)             # dones
# )


# Actorネットワーク πθ(a|s)
class Actor(nn.Module):
	def __init__(self, obs_size, action_size):
		super(Actor, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(obs_size, 128),
			nn.ReLU(),
			nn.Linear(128, action_size),
			nn.Tanh() # アクションが-1~1なのでこの範囲にする。
		)

	def forward(self, x):
		return self.fc(x)


# Criticネットワーク
class Critic(nn.Module):
	def __init__(self, obs_size, action_size):
		super(Critic, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(obs_size + action_size, 128),
			nn.ReLU(),
			nn.Linear(128, 1)
		)

	def forward(self, state, action):
		x = torch.cat([state, action], dim=1)
		return self.fc(x)

# ネットワークの初期化
# DQNを使っているので、ターゲットネットワークを使用している。
# 数式 : Rt + γmax_aQ(St+1, a)がターゲットでDDPGなので、Rt + γQ(St+1, μ(s))であるため、
# ターゲットネットワークとしてactor(μ)とcritic(Q)の2つのネットワークをもう一個持っておく必要がある。
# ネットワークはそれぞれパラメータθとφになっている
actor = Actor(obs_size, action_size)
critic = Critic(obs_size, action_size)
target_actor = Actor(obs_size, action_size)
target_critic = Critic(obs_size, action_size)

# nn.Moduleを継承しているので、そちらのクラスにあるメソッドで、重みをコピーしている
# state_dict : 学習可能なパラメータと学習に関連する状態(例えばバッチ正規化やドロップアウトの統計情報)
# load_state_dict : 他のモデルや保存されたファイルから取得したstate_dictを現在のモデルにロードする
# スタートの重みを一致させておく。ランダムなので。
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

# オプティマイザの初期化
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate_actor)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate_critic)

# リプレイバッファのインスタンス化
replay_buffer = ReplayBuffer(buffer_size)

# 探索ノイズ (OUノイズを実装)
# Δt = 1として近似して実装する。
# tは通常、ステップ数として設定する。
# sizeはaction_sizeに対応していて、正規分布の次元数を意味している
# state は数式のxを意味し、現在のノイズ値を意味している。
class OUNoise:
	def __init__(self, size, mu = 0.0, theta=0.15, sigma=0.2):
		self.size = size
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.state = np.ones(self.size) * self.mu

	def reset(self):
		self.state = np.ones(self.size) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
		self.state += dx
		return self.state

noise = OUNoise(action_size)

# メインループ
state, _ = env.reset()
for step in range(n_steps):
	state_tensor = torch.tensor(state, dtype=torch.float32)
	action = actor(state_tensor).detach().numpy()
	action += noise.sample()
	action = np.clip(action, -1, 1) # 行動空間が-1~1なので、それをオーバーしたら最大値最小値にクリップする。

	# 実際に行動をして経験をためる
	next_state, reward, done, trucnated, _ = env.step(action)
	done = done or trucnated
	replay_buffer.add(state, action, reward, next_state, done)
	state = next_state

	# バッファが十分になったらTrainする
	if len(replay_buffer) > batch_size:
		states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

		# Critic(Qネットワーク)更新
		# target_critic/actorはネットワークを固定している。更新は下に実装している。値の計算だけはここでする。
		with torch.no_grad():
			# ここの処理でバッチの次元があるままtarget_criticやtarget_actorに値を入れているのですが、どうやらネットワークの
			# nn.Linearやnn.ReLUは先頭の次元を自動的にbatchサイズと認識するようなので、これで良い。
			target_q = rewards + gamma * target_critic(next_states, target_actor(next_states)) * (1 - dones)
		q_values = critic(states, actions)
		# インスタンス化と計算を一回で行なっている
		# ミニバッチ全体に対して計算された平均的なスカラー値をlossとしている
		critic_loss = nn.MSELoss()(q_values, target_q)

		critic_optimizer.zero_grad()
		critic_loss.backward()
		critic_optimizer.step()

		# Actor(πネットワーク)の更新
		# lossはバッチサイズを平均したもの。MSELossは関数内で平均の処理が走っているが、actorの方は平均にしていない
		# ので明示的に平均を取る操作をしている
		actor_loss = -critic(states, actor(states)).mean()

		actor_optimizer.zero_grad()
		actor_loss.backward()
		actor_optimizer.step()

		# ターゲットネットワークを更新する
		# ソフトターゲットで平滑平均で更新する
		for target_param, param in zip(target_critic.parameters(), critic.parameters()):
			target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

		for target_param, param in zip(target_actor.parameters(), actor.parameters()):
			target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	if done:
		state, _ = env.reset()
		noise.reset()

	# critic_loss,actor_lossが計算されている。つまりif len(replay_buffer) > batch_sizeが正の時。
	if step % 100 == 0 and len(replay_buffer) > batch_size:
		print(f"Step {step}: Critic Loss = {critic_loss.item()}, Actor Loss = {actor_loss.item()}")

env.close()
