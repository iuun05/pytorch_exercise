import random

import numpy as np
import torch
import torch.nn as nn
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
actor_learing_rate = 0.8
critic_learing_rate = 0.5
GAMMA = 0.95
MEMORY_SIZE = 1000000
BATCH_SIZE = 128
# 主要是要经验取决于现在的状态，而不是之前的经验
# 之前的经验应该只占用一小部分权重
TAU = 6E-1

# critic network -> actor network -> replay buffer -> ddpg algorithm

class Actor(nn.Module):
    def __init__(self, state_dim,action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        # 三个全连接层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        # self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # why is tanh?->由于这里的动作空间是-2~2之间，tanh（）函数正好是在-1~1之间的，所以正好可以对应相关的映射关系
        x = torch.tanh(self.fc3(x)) * 2
        return x
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        # TODO
        # Q(s,a) , 两个输入，一个输出, 具体是Q值的大小，因此是一个一维
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    def forward(self, x, a):
        # why cat?
        # TODO
        x = torch.cat([x, a], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store_transition(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        # 升维度具体还要看游戏传出来的数据是什么维度，然后进行转换
        self.buffer.append((state, action, reward, next_state, done))

    # 取batch_size的经验出来
    def sample(self, batch_size):
        # 先取出来解包，然后在打包为元组赋值
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # 还原数据的格式，state，next_state就是一维三列
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    # 保证mini_batch_size大于等于buffer的长度，才能进行sample
    def __len__(self):
        return len(self.buffer)

# agent brain contain critic, actor and experience replay
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learing_rate)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learing_rate)

        self.replay_buffer = ReplayMemory(MEMORY_SIZE)

    def get_action(self, state):
        # 升维度
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        # reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        # done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(np.float32(done)).to(device)
        next_state = torch.FloatTensor(next_state).to(device)

        # update critic network
        next_action = self.actor_target(next_state)
        # target_Q = self.critic_target(next_action, next_state.detach())
        target_Q = self.critic_target(next_state, next_action.detach())
        target_Q = reward + (1 - done) * GAMMA * target_Q
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())

        # 梯度更新
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 更新参数
        self.critic_optimizer.step()

        # update actor network
        action_loss = -self.critic(state, self.actor(state)).mean()
        # 梯度更新
        self.actor_optimizer.zero_grad()
        action_loss.backward()
        # 更新参数
        self.actor_optimizer.step()

        # update critic and actor target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
             target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
             target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

