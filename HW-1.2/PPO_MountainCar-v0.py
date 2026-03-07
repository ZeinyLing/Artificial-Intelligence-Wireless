import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from itertools import count
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# ===============================
# Parameters
# ===============================
env_name = "MountainCar-v0"
gamma = 0.99
seed = 1
render = False

# ===============================
# Environment
# ===============================
env = gym.make(env_name)

torch.manual_seed(seed)
state, _ = env.reset(seed=seed)

num_state = env.observation_space.shape[0]
num_action = env.action_space.n

Transition = namedtuple(
    "Transition", ["state", "action", "log_prob", "reward", "next_state"]
)

# ===============================
# Actor Network
# ===============================
class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_state, 128),
            nn.ReLU(),
            nn.LayerNorm(128),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, num_action)
        )

    def forward(self, x):
        logits = self.net(x)
        prob = F.softmax(logits, dim=1)
        return prob


# ===============================
# Critic Network
# ===============================
class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_state, 128),
            nn.ReLU(),
            nn.LayerNorm(128),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        value = self.net(x)
        return value


# ===============================
# PPO Agent
# ===============================
class PPO:

    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 25
    batch_size = 64

    def __init__(self):

        self.actor_net = Actor()
        self.critic_net = Critic()

        self.buffer = []
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=2e-4)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=1e-3)

    def select_action(self, state):

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_prob = self.actor_net(state)

        dist = Categorical(action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):

        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float32)
        action = torch.tensor([t.action for t in self.buffer]).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_log_prob = torch.tensor([t.log_prob for t in self.buffer]).view(-1,1)

        # ===== bootstrap value =====
        with torch.no_grad():
            last_state = torch.tensor(self.buffer[-1].next_state,
                                      dtype=torch.float32).unsqueeze(0)
            R = self.critic_net(last_state).item()

        Gt = []

        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)

        Gt = torch.tensor(Gt, dtype=torch.float32)

        with torch.no_grad():
            values = self.critic_net(state).squeeze()

        advantage = Gt - values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(self.ppo_update_time):

            for index in BatchSampler(
                SubsetRandomSampler(range(len(self.buffer))),
                self.batch_size,
                False,
            ):

                V = self.critic_net(state[index])

                dist = Categorical(self.actor_net(state[index]))
                action_log_prob = dist.log_prob(
                    action[index].squeeze()
                ).view(-1,1)

                ratio = torch.exp(action_log_prob - old_log_prob[index])

                surr1 = ratio * advantage[index].view(-1,1)
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_param,
                    1 + self.clip_param
                ) * advantage[index].view(-1,1)

                entropy = dist.entropy().mean()

                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(),
                    self.max_grad_norm
                )
                self.actor_optimizer.step()

                critic_loss = F.mse_loss(Gt[index].view(-1,1), V)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(),
                    self.max_grad_norm
                )
                self.critic_optimizer.step()

        self.buffer.clear()


# ===============================
# Training
# ===============================
def main():

    agent = PPO()

    best_step = float("inf")
    best_episode = -1

    for i_epoch in range(1000):

        state, _ = env.reset()

        for t in count():

            action, log_prob = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            progress = next_state[0] - state[0]
            velocity = abs(next_state[1])

            reward = reward + progress * 8 + velocity * 4

            if next_state[0] >= 0.5:
                reward += 200

            done = terminated or truncated

            trans = Transition(
                state,
                action,
                log_prob,
                reward,
                next_state
            )

            agent.store_transition(trans)

            state = next_state

            if done:

                agent.update()

                print(f"Episode {i_epoch} | Step {t}")

                if t < best_step:
                    best_step = t
                    best_episode = i_epoch

                break

    print("\nTraining Finished")
    print(f"Best Episode : {best_episode}")
    print(f"Minimum Step : {best_step}")

if __name__ == "__main__":
    main()
    print("end")