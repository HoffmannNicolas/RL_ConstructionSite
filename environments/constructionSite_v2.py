

import numpy as np
from .constructionSite import ConstructionSite
import random
import gym

import math
from PIL import Image, ImageDraw

class ConstructionSite_v2(ConstructionSite) :

    """ Variation of a ConstructionSite environment that creates more complex environments """

    def __init__(self, *args, **kwargs) :
        super(ConstructionSite_v2, self).__init__(*args, **kwargs)


    def _defineInitMap(self, numPoints=100, numIterations=100) :
        initMap = np.zeros((self.width, self.height))

        pickZoneCenter = np.random.rand(2) # [x, y] coordinates of the center
        dropZoneDefined = False
        while not(dropZoneDefined) :
            dropZoneCenter = np.random.rand(2) # [x, y] coordinates of the center
            dropZoneDefined = np.linalg.norm(pickZoneCenter - dropZoneCenter) > 0.7

        def _getCells(center) :
            cellPosX = np.random.normal(center[0], 0.25, size=(numPoints, 1))
            cellPosY = np.random.normal(center[1], 0.25, size=(numPoints, 1))
            cellPos = np.concatenate((cellPosX * self.width, cellPosY*self.height), axis=-1)
            cellPos = cellPos.astype('int64')
            cellPos = cellPos[np.where(cellPos[:, 0] >= 0)]
            cellPos = cellPos[np.where(cellPos[:, 0] < self.width)]
            cellPos = cellPos[np.where(cellPos[:, 1] >= 0)]
            cellPos = cellPos[np.where(cellPos[:, 1] < self.height)]
            return cellPos
        pickCells = _getCells(pickZoneCenter)
        dropCells = _getCells(dropZoneCenter)

        for i in range(numIterations) :
            pickCellChoice = random.choice(pickCells)
            dropCellChoice = random.choice(dropCells)
            if (abs(initMap[pickCellChoice[0], pickCellChoice[1]]) >= self.highestAltitudeError) :
                continue
            if (abs(initMap[dropCellChoice[0], dropCellChoice[1]]) >= self.highestAltitudeError) :
                continue
            initMap[pickCellChoice[0], pickCellChoice[1]] -= 1
            initMap[dropCellChoice[0], dropCellChoice[1]] += 1
        return initMap
