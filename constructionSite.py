

import numpy as np
import random
import gym

import math
from PIL import Image, ImageDraw

class ConstructionSite(gym.Env) :

    """ Custom "Construction Site" environment that follows gym interface """

    def __init__(self, gridWidth=10, gridHeight=10, highestAltitudeError=10, seed=random.random()) :
        super(ConstructionSite, self).__init__()
        random.seed(seed)

            # Properties of the construction site
        self.width = gridWidth
        self.height = gridHeight
        self.highestAltitudeError = highestAltitudeError
        self.map = np.zeros((self.width, self.height))
        self.initMap = None

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
        self.n_actions = 6 # GoUp, GoLeft, GoRight, GoDown, Pick, Drop
        self.action_space = gym.spaces.Discrete(self.n_actions)

            # Observation space
        self.observation_space = gym.spaces.Box(
            low=-highestAltitudeError,
            high=highestAltitudeError,
            shape=(2, self.width, self.height)
        )


    def _defineInitMap(self) :
        initMap = np.copy(self.map)
        initMap[random.randint(0, self.width-1), random.randint(0, self.height-1)] += 5
        initMap[random.randint(0, self.width-1), random.randint(0, self.height-1)] -= 5
        return initMap


    def _computeObservation(self) :
        positionMap = np.zeros((1, self.width, self.height))
        if (self.isLoaded) :
            positionMap[0, self.w, self.h] = 1
        else :
            positionMap[0, self.w, self.h] = -1
        _map = np.expand_dims(self.map, axis=0)
        obs = np.concatenate((_map, positionMap), axis=0)
        return obs


    def reset(self) :
        self.w = random.randint(0, self.width-1)
        self.h = random.randint(0, self.height-1)
        self.isLoaded = False

        if (self.initMap is None) :
            self.initMap = self._defineInitMap()
        self.map = np.copy(self.initMap)
        return self._computeObservation()


    def _measureFlatness(self) :
        sum = -np.sum(np.abs(self.map))
        return float(sum)


    def _isMapFlat(self) :
        sum = self._measureFlatness()
        return bool(sum == 0) # Cast <np._bool> to <bool>


    def step(self, action) :

        previousFlatness = self._measureFlatness()
        previousW = self.w
        previousH = self.h

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
