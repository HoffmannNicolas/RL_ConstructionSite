
from environments.constructionSite_v2 import ConstructionSite_v2
from customAgents.naiveNearest import NaiveNearest
from gym.wrappers import TimeLimit

env = ConstructionSite_v2(gridWidth=6, gridHeight=6, seed=0, stochasticity=0, exploringStarts=True, metaLearning=False, continuousActions=False)

agent = NaiveNearest()

imagesGrid_alt =[]
imagesGrid_err =[]
obs = env.reset()
for _ in range(5) :
    imagesGrid_alt.append(env.render("human", toShow="altitude"))
    imagesGrid_err.append(env.render("human", toShow="error"))
for step in range(10000):
    print(f"\rStep {step}", end='')
    action, _ = agent.act(obs)
    obs, reward, done, info = env.step(action)
    # print("reward : ", reward)
    # env.render(mode='console')
    imagesGrid_alt.append(env.render("human", toShow="altitude"))
    imagesGrid_err.append(env.render("human", toShow="error"))
    if done:
        print("Goal reached!", "reward=", reward)
        break
print()
for _ in range(5) :
    imagesGrid_alt.append(env.render("human", toShow="altitude"))
    imagesGrid_err.append(env.render("human", toShow="error"))
imagesGrid_alt[0].save(f'_data/visuTradi_alt.gif', save_all=True, append_images=imagesGrid_alt[1:], optimize=True, duration=20, loop=0)
imagesGrid_err[0].save(f'_data/visuTradi_err.gif', save_all=True, append_images=imagesGrid_err[1:], optimize=True, duration=20, loop=0)
imagesGrid_err[0].save('_data/visuTradi_initialErrors.png')
imagesGrid_alt[0].save('_data/visuTradi_Start.png')
imagesGrid_alt[-1].save('_data/visuTradi_End.png')
