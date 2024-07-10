import datetime
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent_DDPG import DDPGAgent

# Hyperparameters
# step in for
MAX_EPISODES = 250
MAX_STEPS = 500
# 线性变化，初始以及
EPISODES_START = 3
# 前面50%的都是在探索，后面以0.02的探索率进行探索
EPISODES_END = 0.2
EPISODES_DECAY = 70000


# initialize the environment
env = gym.make(id='Pendulum-v1')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# initialize the agent
agent = DDPGAgent(STATE_DIM, ACTION_DIM)


# every episode, we store the reward
REWARD_BUFFER = np.empty(shape=[MAX_EPISODES], dtype=np.float32)


for episode_i in range(MAX_EPISODES):
    # reset the environment
    state, others = env.reset()
    episode_reward = 0

    for step_i in range(MAX_STEPS):
        episode = np.interp(x=episode_i*MAX_STEPS+step_i, xp=[0, EPISODES_DECAY], fp=[EPISODES_START, EPISODES_END])
        # take a random action
        random_sample = np.random.random()
        # 首先进行随机的探索
        if random_sample < episode:
             action = np.random.uniform(low=-2, high=2, size=ACTION_DIM)

        else:
            # take an action based on the current state
            # 体现了Deterministic 的思想，即基于原来的基础上进行探索
            action = agent.get_action(state)

        next_state, reward, done, truncation, info = env.step(action)

        # store the transition in the replay buffer
        agent.replay_buffer.store_transition(state, action, reward, next_state, done)

        # update the agent
        state = next_state
        episode_reward+=reward

        # minibatch, then TD learning, update the target network in RL brain
        agent.update()  # TODO

        if done:
             break

    REWARD_BUFFER[episode_i] = episode_reward
    print(f'Episode {episode_i+1}, Reward: {round(episode_reward, 2)}')

current_path = os.path.dirname(os.path.abspath(__file__))
model = current_path + '\\models\\'

time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
timestamp = time.replace(':', '-').replace(' ', '_')

# save model
torch.save(agent.actor.state_dict(), model + f'actor_{timestamp}.pth')



plt.plot(REWARD_BUFFER)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DDPG Reward')
plt.show()


env.close()



