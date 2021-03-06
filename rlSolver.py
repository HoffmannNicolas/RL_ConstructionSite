
import os.path
import gym
import torch as th
import torch.nn as nn

from environments.constructionSite import ConstructionSite
from environments.constructionSite_v2 import ConstructionSite_v2
from stable_baselines3 import DQN, PPO, A2C, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv

from gym.wrappers import TimeLimit

agentPath = "_data/myModel8.zip"


def makeEnv() :
    # env = ConstructionSite(gridWidth=5, gridHeight=5, seed=0, stochasticity=0, exploringStarts=True, metaLearning=False, continuousActions=False)
    env = ConstructionSite_v2(gridWidth=5, gridHeight=5, seed=0, stochasticity=0, exploringStarts=True, metaLearning=False, continuousActions=False)
    # env = ConstructionSite_v2(gridWidth=5, gridHeight=5, seed=0)
    env = TimeLimit(env, max_episode_steps=2000)
    return env

# env_vec = SubprocVecEnv([lambda: ConstructionSite_v2(gridWidth=5, gridHeight=5, seed=0) for _ in range(6)])
env = makeEnv()
env_vec = DummyVecEnv([lambda: makeEnv() for _ in range(8)])

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Flatten(),
            nn.Linear(400, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 8, bias=True),
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
    # model = PPO.load(agentPath)
    model = DQN.load(agentPath)
    model.set_env(env)
else :
    print(f"Instanciate new agent and save in {agentPath}")
    # model = PPO("CnnPolicy", env_vec, policy_kwargs=policy_kwargs, verbose=1)
    # model = DQN("CnnPolicy", env_vec, policy_kwargs=policy_kwargs, verbose=1)
    model = DQN("CnnPolicy", env, target_update_interval=1000, batch_size=512, exploration_final_eps=0.2, policy_kwargs=policy_kwargs, verbose=1)
    model.save(agentPath)

# Record gif of trained agent
imagesGrid =[]
obs = env.reset()
imagesGrid.append(env.render("human"))
for step in range(200):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    print("reward : ", reward)
    env.render(mode='console')
    imagesGrid.append(env.render("human"))
    if done:
        print("Goal reached!", "reward=", reward)
        break
imagesGrid[0].save(f'_data/visu.gif', save_all=True, append_images=imagesGrid[1:], optimize=True, duration=100, loop=0)


for _ in range(50) :
    model.learn(100000)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


    model.save(agentPath)
