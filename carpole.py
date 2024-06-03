# 导入 Gym 的 Python 接口环境包
import gymnasium as gym
import gymnasium.envs as envs
import time
env = gym.make('CartPole-v1', render_mode="human")  # 构建实验环境
observion = env.reset()  # 重置一个回合
print(observion)
cnt = 0
for _ in range(1000):
    env.render()  # 显示图形界面
    action = env.action_space.sample() # 从动作空间中随机选取一个动作
    observation, reward, done, info, _ = env.step(env.action_space.sample())
    if done:
        print('game reset %d'%cnt)
        cnt = cnt + 1
        env.reset()
    print(observation)
    # time.sleep(3)
    # env.reset()
env.close() # 关闭环境


