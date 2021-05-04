

import numpy as np
import random
import gym

import math
from PIL import Image, ImageDraw

class ConstructionSite(gym.Env) :

    """ Custom "Construction Site" environment that follows gym interface """

    def __init__(self, gridWidth=10, gridHeight=10, highestAltitudeError=10, seed=random.random(), stochasticity=0, exploringStarts=False, metaLearning=False, continuousActions=False) :
        # <gridWidth> is the width of the grid
        # <gridHeight> is the height of the grid
        # <highestAltitudeError> is the maximum error, useful for visualization. # TODO : Remove
        # <seed> is the random seed to be used.
        # <stochasticity> is the likelihood that an action is not performed as intended.
        # <exploringStarts> : Is the first state random ?
        # <metaLearning> : Is the environment re-generated every episode ?
        # <continuousActions> : Are the actions continuous ?
        assert isinstance(gridWidth, int), f"<gridWidth> should be an int (got '{type(gridWidth)}' instead)."
        assert (gridWidth >= 1), f"<gridWidth> should be strickly positive (got '{gridWidth}')."
        assert isinstance(gridHeight, int), f"<gridHeight> should be an int (got '{type(gridHeight)}' instead)."
        assert (gridHeight >= 1), f"<gridHeight> should be strickly positive (got '{gridHeight}')."
        assert isinstance(highestAltitudeError, int), f"<highestAltitudeError> should be an int (got '{type(highestAltitudeError)}' instead)."
        assert (highestAltitudeError >= 1), f"<highestAltitudeError> should be strickly positive (got '{highestAltitudeError}')."
        assert isinstance(seed, int), f"<seed> should be an int (got '{type(seed)}' instead)."
        assert (stochasticity >= 0 and stochasticity <= 1), f"<stochasticity> should be in [0, 1] (got '{stochasticity}')."
        assert isinstance(exploringStarts, bool), f"<exploringStarts> should be a boolean (got '{type(exploringStarts)}' instead)."
        assert isinstance(metaLearning, bool), f"<metaLearning> should be a boolean (got '{type(metaLearning)}' instead)."
        assert isinstance(continuousActions, bool), f"<continuousActions> should be a boolean (got '{type(continuousActions)}' instead)."

        super(ConstructionSite, self).__init__()

        self.seed = seed

            # Properties of the construction site
        self.width = gridWidth
        self.height = gridHeight
        self.highestAltitudeError = highestAltitudeError
        self.map = np.zeros((self.width, self.height))
        self.initPosition = None
        self.exploringStarts = exploringStarts
        self.initMap = None
        self.metaLearning = metaLearning

            # Properties of the operating machine
        self.w = None
        self.h = None
        self.isLoaded = None

            # Action space
        self.GoUp = 0
        self.GoLeft = 1
        self.GoDown = 2
        self.GoRight = 3
        self.Pick = 4
        self.Drop = 5
        self.actions = [self.GoUp, self.GoLeft, self.GoDown, self.GoRight, self.Pick, self.Drop]
        self.n_actions = len(self.actions)
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.stochasticity = stochasticity

            # Observation space
        self.observation_space = gym.spaces.Box(
            low=-highestAltitudeError,
            high=highestAltitudeError,
            shape=(2, self.width, self.height)
        )


    def _defineInitPosition(self) :
        """ Define a random initial state of the agent """
        w = random.randint(0, self.width-1)
        h = random.randint(0, self.height-1)
        isLoaded = False
        return [w, h, isLoaded]


    def _defineInitMap(self) :
        """ Define a random initial state of the environment """
        initMap = np.copy(self.map)
        initMap[random.randint(0, self.width-1), random.randint(0, self.height-1)] += 5
        initMap[random.randint(0, self.width-1), random.randint(0, self.height-1)] -= 5
        return initMap


    def _computeObservation(self) :
        """ Computes the observation to be fed to the agent """
        positionMap = np.zeros((1, self.width, self.height))
        if (self.isLoaded) :
            positionMap[0, self.w, self.h] = 1
        else :
            positionMap[0, self.w, self.h] = -1
        _map = np.expand_dims(self.map, axis=0)
        obs = np.concatenate((_map, positionMap), axis=0)
        return obs


    def reset(self) :
        """ Reset the environment, either before it is used the very first time, or when an episode is finished """
        if ((self.initPosition is None) or self.exploringStarts) :
            random.seed(self.seed)
            np.random.seed(seed=self.seed)
            self.initPosition = self._defineInitPosition()
        self.w = self.initPosition[0]
        self.h = self.initPosition[1]
        self.isLoaded = self.initPosition[2]
        if ((self.initMap is None) or self.metaLearning) :
            random.seed(self.seed)
            np.random.seed(seed=self.seed)
            self.initMap = self._defineInitMap()
        self.map = np.copy(self.initMap)
        return self._computeObservation()


    def _measureFlatness(self) :
        """ Measure how far the environment is to be flat """
        sum = -np.sum(np.abs(self.map))
        return float(sum)


    def _isMapFlat(self) :
        """ Episode terminaison condition """
        sum = self._measureFlatness()
        return bool(sum == 0) # Cast <np._bool> to <bool>


    def step(self, action) :
        """ Compute the next state and reward from the current state and the agent's action """
        assert (action in self.actions), f"<action> should be one of the self.actions={str(self.action)}, but got '{action}' instead."

        previousFlatness = self._measureFlatness()
        previousW = self.w
        previousH = self.h
        if (self.stochasticity > 0) : # Handle stochasticity
            if (random.random() < self.stochasticity) :
                action = random.choice(self.actions)
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
        else :
            raise ValueError(f"Recieved invalid action '{action}'.")
        obs = self._computeObservation()
        currentFlatness = self._measureFlatness()
        reward = -0.01 + currentFlatness - previousFlatness
        done = self._isMapFlat()
        info = {}
        return obs, reward, done, info


    def render(self, mode='console'):
        """ Visualize the environment, either on the terminal or on images """
        if mode == 'console':
            print("\n=== Env ===")
            for h in range(self.height) :
                print("\t", end='')
                for w in range(self.width) :
                    if (self.w == w) and (self.h == h) :
                        print(f"X\t", end='')
                    else :
                        print(f"{int(self.map[w, h])}\t", end='')
                print()
            print()

        elif mode == "human" :
            cellWidth = 20 # Pixels
            cellHeight = 20 # Pixels
            leveledCellColor = (125, 90, 50)
            cellBorderColor = (0, 0, 0)
            agentColor = (255, 255, 0)
            agentMargin = 5
            image = Image.new('RGB', (self.width * cellWidth, self.height * cellHeight), leveledCellColor)
            imageDraw = ImageDraw.Draw(image)

                # Draw cells
            for cellCoordX in range(self.width):
                for cellCoordY in range(self.height):
                    cellLeftCoord = cellCoordX * cellWidth
                    cellRightCoord = cellLeftCoord + cellWidth - 1
                    cellTopCoord = cellCoordY * cellHeight
                    cellBottomCoord = cellTopCoord + cellWidth - 1
                    cellAltitude = self.map[cellCoordX, cellCoordY]
                    cellAltitudeError = abs(cellAltitude)
                    if (cellAltitude < 0) : cellColor = tuple([int(val) for val in (1 - cellAltitudeError / self.highestAltitudeError) * np.array(leveledCellColor)])
                    elif (cellAltitude > 0) : cellColor = tuple([int(val) for val in (1 + cellAltitudeError / self.highestAltitudeError) * np.array(leveledCellColor)])
                    else : cellColor = leveledCellColor
                    imageDraw.rectangle([(cellLeftCoord, cellTopCoord), (cellRightCoord, cellBottomCoord)], cellColor)

                # Draw cells borders
            for cellCoordX in range(self.width):
                for cellCoordY in range(self.height):
                    imageDraw.line([(cellCoordX * cellWidth, cellCoordY * cellHeight), ((cellCoordX+1) * cellWidth - 1, cellCoordY * cellHeight)], fill=cellBorderColor, width=0)
                    imageDraw.line([((cellCoordX+1) * cellWidth - 1, cellCoordY * cellHeight), ((cellCoordX+1) * cellWidth - 1, (cellCoordY+1) * cellHeight - 1)], fill=cellBorderColor, width=0)
                    imageDraw.line([((cellCoordX+1) * cellWidth - 1, (cellCoordY+1) * cellHeight - 1), (cellCoordX * cellWidth, (cellCoordY+1) * cellHeight - 1)], fill=cellBorderColor, width=0)
                    imageDraw.line([(cellCoordX * cellWidth, (cellCoordY+1) * cellHeight - 1), (cellCoordX * cellWidth, cellCoordY * cellHeight)], fill=cellBorderColor, width=0)

                # Draw agent
            leftCoord = self.w * cellWidth
            rightCoord = leftCoord + cellWidth - 1
            topCoord = self.h * cellHeight
            bottomCoord = topCoord + cellHeight - 1
            imageDraw.ellipse((leftCoord+agentMargin, topCoord+agentMargin, rightCoord-agentMargin, bottomCoord-agentMargin), fill=agentColor)

            image.save("_data/image.png")
            return image

        else :
            raise NotImplementedError()
