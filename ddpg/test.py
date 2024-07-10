import torch
import gymnasium as gym
import os
from agent_DDPG import Actor
import pygame
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize env
env = gym.make('Pendulum-v1',render_mode='rgb_array')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

current_path = os.path.dirname(os.path.abspath(__file__))
model = current_path + '\\models\\'
actor_path = model + 'actor_2024-07-09_23-30-36.pth'

def process_frame(frame):
    frame = np.transpose(frame, axes= (1, 0, 2))
    frame = pygame.surfarray.make_surface(frame)

    return pygame.transform.scale(frame, (600, 400))


actor = Actor(STATE_DIM, ACTION_DIM).to(device)
actor.load_state_dict(torch.load(actor_path))

pygame.init()
screen = pygame.display.set_mode((600, 400))
clock = pygame.time.Clock()




MAX_EPISODES = 200
MAX_STEPS = 500
for episode in range(MAX_EPISODES):
     state, others = env.reset()
     episode_reward = 0

     for step in range(MAX_STEPS):
        action = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
        action = action.detach().cpu().numpy()[0]
        next_state, reward, done, truncation, info = env.step(action)
        state = next_state

        episode_reward += reward

        frame = env.render()

        frame = process_frame(frame)
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        clock.tick(30)
        # if done:
        #     break
     print(f'Episode: {episode}, Reward: {episode_reward}')

pygame.quit()
env.close()
print('Finished')