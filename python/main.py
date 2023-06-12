import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import constant

env_name = 'CartPole-v1'
env = gym.make("CartPole-v1", render_mode="rgb_array")

episodes = 5
for episode in range(episodes):
    state  = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        truncated, reward, done, truncate, info = env.step(action)
        score += reward
    print(f'episodes: {episode+1}, score: {score}')

env.close()

env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1)

model.learn(total_timesteps=20000)

model.save(constant.PPO_MODEL_PATH)

env.close()