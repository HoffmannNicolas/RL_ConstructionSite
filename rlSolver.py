
import os.path
import gym
import torch as th
import torch.nn as nn

from constructionSite import ConstructionSite
from stable_baselines3 import DQN, PPO, A2C, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy


from gym.wrappers import TimeLimit

agentPath = "_data/myModel.zip"


env = ConstructionSite(gridWidth=5, gridHeight=5)
env = TimeLimit(env, max_episode_steps=2000)


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

if os.path.isfile(agentPath) :
    print(f"Load agent from {agentPath}")
    model = PPO.load(agentPath)
    model.set_env(env)
else :
    print(f"Instanciate new agent and save in {agentPath}")
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.save(agentPath)

# Test the trained agent
obs = env.reset()
for step in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode='console')
    env.render(mode='human')
    if done:
        print("Goal reached!", "reward=", reward)
        break


for _ in range(50) :
    model.learn(100000)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


    model.save(agentPath)
