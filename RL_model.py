import util, math, random
from collections import defaultdict
from util import ValueIteration
from typing import List, Callable, Tuple, Any
import numpy as np
from Laser_alignment import *
import random

class LaserBeamMDP(util.MDP):
    def __init__(self, phi1, theta1, phi2, theta2):
        self.phi1 = phi1
        self.theta1 = theta1
        self.phi2 = phi2
        self.theta2 = theta2

    def startState(self) -> Tuple:
        # The design of our state representation
        # keep track of the degree of freedoms of the mirror, and final dist
        return (self.phi1, self.theta1, self.phi2, self.theta2)

    def actions(self, state: Tuple) -> List[Any]:
        return ['Move']

    def forModel(self, state, tLoc):
        iLoc = np.array([0,0,0])
        tLoc = tLoc
        phi1, theta1, phi2, theta2 = state
        laser_beam = LaserBeam(iLoc, (phi1, theta1), (phi2, theta2), tLoc)
        laser_beam.initial()
        laser_beam.raytracing()
        return laser_beam.iLoc

    def angleRefine(self, state):
        phi1, theta1, phi2, theta2 = state
        if phi1 >= 360:
            phi1 -= 360
        if phi2 >= 360:
            phi2 -= 360
        if theta1 >= 360:
            theta1 -= 360
        if theta2 >= 360:
            theta2 -= 360
        return (phi1, theta1, phi2, theta2)

    def rewardfun(self, dist):
        return 1/dist

    def succAndProbReward(self, state: Tuple, action: Any) -> List[Tuple]:
        # Given a |state| and |action|, return a list of (newState, prob, reward)
        # tuples, corresponding to the states reachable from |state| when taking
        # |action|
        result = []
        tLoc = np.array([10, 10, 12 * 2.54])
        phi1, theta1, phi2, theta2 = state
        alist = [+0.1, -0.1]

        Beam_tloc = self.forModel(state, tLoc)
        dist = 0
        for i in range(len(Beam_tloc)):
            dist += (Beam_tloc[i] - tLoc[i]) ** 2
        dist = np.sqrt(dist)
        #print('old distance', dist, state)

        if (dist <= 1):
            return []

        elif action == 'Move':
            list = [0, 1, 2, 3]
            imove = random.sample(list, 1)[0]

            #print(imove)
            if imove == 0:

                newState = (phi1 + alist[0], theta1, phi2, theta2)
                Beam_tloc = self.forModel(newState, tLoc)
                dist = 0
                for i in range(len(Beam_tloc)):
                    dist += (Beam_tloc[i] - tLoc[i])**2
                dist = np.sqrt(dist)
                newReward = dist * 0.01
                newReward = self.rewardfun(newReward)
                newState = (phi1 + alist[0], theta1, phi2, theta2)
                newState = self.angleRefine(newState)
                result.append((newState, 0.25, newReward))
                #print(imove, newState, newReward, 'distance', dist)

            elif imove == 1:
                newState = (phi1, theta1 + alist[0], phi2, theta2)
                Beam_tloc = self.forModel(newState, tLoc)
                dist = 0
                for i in range(len(Beam_tloc)):
                    dist += (Beam_tloc[i] - tLoc[i]) ** 2
                dist = np.sqrt(dist)
                newReward = dist * 0.01
                newReward = self.rewardfun(newReward)
                newState = (phi1, theta1 + alist[0], phi2, theta2)
                newState = self.angleRefine(newState)
                result.append((newState, 0.25, newReward))
                #print(imove, newState, newReward, 'distance', dist)

            elif imove == 2:
                newState = (phi1, theta1, phi2 + alist[0], theta2)
                Beam_tloc = self.forModel(newState, tLoc)
                dist = 0
                for i in range(len(Beam_tloc)):
                    dist += (Beam_tloc[i] - tLoc[i]) ** 2
                dist = np.sqrt(dist)
                newReward = dist * 0.01
                newReward = self.rewardfun(newReward)
                newState = (phi1, theta1, phi2 + alist[0], theta2)
                newState = self.angleRefine(newState)
                result.append((newState, 0.25, newReward))
                #print(imove, newState, newReward, 'distance', dist)

            elif imove == 3:
                newState = (phi1, theta1, phi2, theta2 + alist[0])
                Beam_tloc = self.forModel(newState, tLoc)
                dist = 0
                for i in range(len(Beam_tloc)):
                    dist += (Beam_tloc[i] - tLoc[i]) ** 2
                dist = np.sqrt(dist)
                newReward = dist * 0.01
                newReward = self.rewardfun(newReward)
                newState = (phi1, theta1, phi2, theta2 + alist[0])
                newState = self.angleRefine(newState)
                result.append((newState, 0.25, newReward))
                #print(imove, newState, newReward, 'distance', dist)
        #print(result)
        return result


    def discount(self):
        return 1







































