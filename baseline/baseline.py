import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

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
    
policy_net = PolicyNetwork(obs_size, n_actions)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

def compute_discounted_rewards(rewards, gamma=0.99):
    discount_rewards = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        discount_rewards.insert(0, G)
    return torch.tensor(discount_rewards, dtype=torch.float32)

n_episodes = 1000
gamma = 0.99

for episode in range(n_episodes):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    done = False
    
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = policy_net(state) # forwardをオーバーライドしている継承元の昨日より__call__でforwardメソッドが自動的に実行される
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_probs.append(action_dist.log_prob(action))

        next_state, reward, done, truncated, info = env.step(action.item()) # actionはtorch.tensor, これを.item()でintにしている
        done = done or truncated # truncatedはtimelimitに達したかどうかcartpoli-v1は500step
        rewards.append(reward)
        state = next_state
        
        env.render()

    discounted_reward = compute_discounted_rewards(rewards, gamma) # 割引報酬はエピソードごとに計算して、勾配を更新させたらリセットされる。
    # ベースラインの導入---------------------------
    discounted_reward -= discounted_reward.mean()

    loss = -torch.cat([log_prob.unsqueeze(0) for log_prob in log_probs]) * discounted_reward
    loss = loss.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {episode + 1}: Total reward = {sum(rewards)}")

    if sum(rewards) >= 195:
        print(f"Solved in {episode + 1} episodes!!!!")
        break

env.close()
