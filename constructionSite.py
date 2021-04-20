

import numpy as np
import gym


class ConstructionSite(gym.Env) :

    """ Custom "Construction Site" environment that follows gym interface """

    GoUp = 0
    GoLeft = 1
    GoDown = 2
    GoRight = 3
    Pick = 4
    Drop = 5

    def __init__(self, gridWidth=10, gridHeight=10) :
        super(ConstructionSite, self).__init__()

            # Properties of the construction site
        self.width = gridWidth
        self.height = gridHeight
        self.map = np.zeroes((self.width, self.height))

            # Properties of the operating machine
        self.w = randint(self.width)
        self.h = randint(self.height)
        self.isLoaded = False

            # Action space
        n_actions = 6 # GoUp, GoLeft, GoRight, GoDown, Pick, Drop
        self.action_space = gym.spaces.Discrete(n_actions)

            # Observation space
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.width, self.height, 2]),
            dtype=np.int32
        )


    def _computeObservation(self) :
        positionMap = np.zeroes((self.width, self.height))
        positionMap[self.w, self.h] = 1
        obs = np.stack(self.map, positionMap)
        return obs

    def reset(self) :
        self.w = randint(self.width)
        self.h = randint(self.height)
        self.isLoaded = False
        self.map = np.zeroes((self.width, self.height))

        self.map[randint(self.width), randint(self.height)] += 5
        self.map[randint(self.width), randint(self.height)] -= 5

        return self._computeObservation()


    def _isMapFlat(self) :
        sum = np.sum(np.abs(self.map))
        return sum == 0


    def step(self, action) :

        previousHeight = self.map[self.w, self.h]

        if (action == self.GoUp) :
            if (self.h < self.height-1) :
                self.h += 1
        elif (action == self.GoLeft) :
            if (self.w > 0) :
                self.w -= 1
        elif (action == self.GoDown) :
            if (self.h > 0) :
                self.h -= 1
        elif (action == self.GoRight) :
            if (self.w < self.width-1) :
                self.w += 1
        elif (action == self.Pick) :
            if not(self.isLoaded) :
                self.isLoaded = True
                self.map[self.w, self.h] -= 1
        elif (action == self.Drop) :
            if (self.isLoaded) :
                self.isLoaded = False
                self.map[self.w, self.h] += 1
        raise ValueError(f"Recieved invalid action '{action}'.")

        done = self._isMapFlat()

        reward = -0.05
        if (abs(self.map[self.w, self.h]) < abs(previousHeight)) :
            reward += 1

        info = {}

        return self._computeObservation(), reward, done, info
