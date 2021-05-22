
import numpy as np
import random

class NaiveNearest() :

    def __init__(self) :
        pass

    def _isLoaded(self, obs) :
        return np.sum(obs[1]) == 1

    def _position(self, obs) :
        return np.nonzero(obs[1])

    def act(self, obs) :

        altiMap = obs[0]
        loaded = self._isLoaded(obs)

        if (loaded) :
            targets = np.where(altiMap < 0, altiMap, 0)
        else :
            targets = np.where(altiMap > 0, altiMap, 0)

        # print("targets : ", targets)

        targets = np.array(np.nonzero(targets))

        # print("targets : ", targets)

        myPos = np.array(self._position(obs))

        # print("myPos : ", myPos)
        # print("myPos.shape : ", myPos.shape)
        # print("targets.shape : ", targets.shape)

        deltas = targets - myPos

        # print("deltas : ", deltas)
        # print("deltas.shape : ", deltas.shape)

        deltaNorms = np.linalg.norm(deltas, axis=0, ord=1)
        # print("deltaNorms : ", deltaNorms)
        # print("deltaNorms.shape : ", deltaNorms.shape)

        minDelta = np.min(deltaNorms)
        # print("minDelta : ", minDelta)
        if (minDelta == 0) :
            if (loaded) :
                return 5, None # Drop
            else :
                return 4, None # Pick

        possibleActions = []
        delta = deltas[:, np.argmin(deltaNorms)]

        if (delta[0] > 0) : possibleActions.append(3) # I need to go right
        if (delta[0] < 0) : possibleActions.append(1) # I need to go left
        if (delta[1] > 0) : possibleActions.append(0) # I need to go up
        if (delta[1] < 0) : possibleActions.append(2) # I need to go down

        return random.choice(possibleActions), None
