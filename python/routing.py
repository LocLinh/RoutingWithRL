import constant
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import routing_env

env = routing_env.Routing()
model = PPO('MlpPolicy', env, verbose=2)

model.learn(total_timesteps=200000)

model.save(constant.CUSTOM_ENV_PPO_MODEL_PATH)
