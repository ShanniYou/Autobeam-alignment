
import random, util, collections
from RL_model import *

def main():

    Lmdp = LaserBeamMDP(phi1 = 45, theta1=10, phi2=10, theta2=10)
    startState = Lmdp.startState()
    alg = util.ValueIteration()
    alg.solve(Lmdp, 0.01)
    print(alg.V[startState])


if __name__ == '__main__':
    main()







