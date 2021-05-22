
from environments.constructionSite_v2 import ConstructionSite_v2
from customAgents.naiveNearest import NaiveNearest
from gym.wrappers import TimeLimit

env = ConstructionSite_v2(gridWidth=20, gridHeight=20, seed=0, stochasticity=0, exploringStarts=True, metaLearning=False, continuousActions=False)

agent = NaiveNearest()

imagesGrid =[]
obs = env.reset()
imagesGrid.append(env.render("human"))
for step in range(10000):
    print(f"\rStep {step}", end='')
    action, _ = agent.act(obs)
    obs, reward, done, info = env.step(action)
    # print("reward : ", reward)
    # env.render(mode='console')
    imagesGrid.append(env.render("human"))
    if done:
        print("Goal reached!", "reward=", reward)
        break
print()
imagesGrid[0].save(f'_data/visuTradi.gif', save_all=True, append_images=imagesGrid[1:], optimize=True, duration=20, loop=0)
