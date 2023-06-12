import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import constant
import routing_env

env = routing_env.Routing()


# env = gym.make('CartPole-v1', render_mode="human")
model = PPO.load(constant.CUSTOM_ENV_PPO_MODEL_PATH, env=env)
episodes = 1
for episode in range(episodes):
    i = 0
    obs  = env.reset()[0]
    done = False
    score = 0
    while not done:
        # env.render()
        # print("===", obs[0])
        # action, _state = model.predict(obs)
        i+= 1
        if i > 10:
            break
        action = env.action_space.sample()
        obs, reward, done, truncate, info = env.step(action)
        print("==============")
        print(action)
        print(obs)
        print("==============")
        score += reward
    print(f'episodes: {episode+1}, score: {score}')



