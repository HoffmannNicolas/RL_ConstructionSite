
from stable_baselines3.common.env_checker import check_env
from constructionSite import ConstructionSite


env = ConstructionSite()
# If the environment don't follow the interface, an error will be thrown

obs = env.reset()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

for action in range(0, 6) :
    observation, reward, done, info = env.step(action)
    print(f"observation.shape : {observation.shape}\t\treward : {reward}\t\tdone : {done}\t\taction : {action}")

check_env(env, warn=True)
