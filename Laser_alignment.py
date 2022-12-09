# Created by Shanni You: 11/17/2022
# First, design a ray tracing model of the light.
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import scipy.optimize as optimize
sy.init_printing()

class LaserBeam:
    def __init__(self, iLoc, angle1, angle2, tLoc):
        self.iLoc = iLoc
        self.phi1 = angle1[0]     # towards x
        self.theta1 = angle1[1]   # towards y
        self.phi2 = angle2[0]
        self.theta2 = angle2[1]
        self.tLoc = tLoc
        self.l1 = None
        self.l2 = None
        self.loclist = np.zeros((4, 3))

    def initial(self):
        # Free space matrix design, 1 inch = 2.54 cm:
        self.l1 = 8 * 2.54   # cm
        self.l2 = 4 * 2.54   # cm
        self.l3 = 12* 2.54   # cm

    def conRot(self, theta, phi):
        R = np.zeros((3,))
        R[0] = 2 * (np.cos(theta)**2) * np.cos(phi) * np.sin(phi)
        R[1] = -2 * np.cos(theta) * np.sin(theta) * np.cos(phi)
        R[2] = -1 + 2 * np.cos(theta)**2 * np.cos(phi)**2
        return R

    def fun1(self, A):
        return self.iLoc[2] + A * (-1 + 2*np.cos(self.theta1)**2 * np.cos(self.phi1)**2) - self.l2

    def fun2(self, A):
        return self.iLoc[2] + A * (-1 + 2*np.cos(self.theta2)**2 * np.cos(self.phi2)**2) - self.tLoc[-1]

    def raytracing(self):
        # Configuration design, first mirror at z = self.l1:

        self.loclist[0,:] = self.iLoc    # original point
        self.iLoc[2] = self.l1
        R1 = self.conRot(self.theta1, self.phi1)
        self.loclist[1, :] = self.iLoc    # First mirror

        # new location
        A1 = optimize.newton(self.fun1, 10)
        self.iLoc = self.iLoc + A1 * R1
        self.loclist[2, :] = self.iLoc  # Second mirror

        R2 = self.conRot(self.theta2, self.phi2)
        A2 = optimize.newton(self.fun2, 50)

        self.iLoc = self.iLoc + A2 * R2
        self.loclist[3, :] = self.iLoc   # Third mirror
        self.loclist = np.array(self.loclist)

def main():
    iLoc = np.array([0, 0, 0])
    tLoc = np.array([0, 0, 12*2.54])
    #laser_test = LaserBeam(iLoc, (48, 14), (14, 11), tLoc)
    laser_test = LaserBeam(iLoc, (48, 14), (14, 11), tLoc)
    laser_test.initial()
    laser_test.raytracing()

    plt.figure()
    plt.plot(laser_test.loclist[:, -1])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(laser_test.loclist[:,0], laser_test.loclist[:,1], laser_test.loclist[:,2])
    ax.scatter3D(laser_test.loclist[:, 0], laser_test.loclist[:, 1], laser_test.loclist[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for i in range(4):  # plot each point + it's index as text above
        x = laser_test.loclist[i, 0]
        y = laser_test.loclist[i, 1]
        z = laser_test.loclist[i, 2]
        label = i
        ax.scatter(x, y, z, color='b')
        ax.text(x, y, z, '%s' % (label), size=20, zorder=1, color='k')

    ax.scatter(tLoc[0], tLoc[1], tLoc[2])

    ax.text(tLoc[0], tLoc[1], tLoc[2], 'targeted', size=20, zorder=1, color='k')

    print('target location:', tLoc)
    print('test location:', laser_test.loclist[-1,:])
    plt.show()

if __name__ == '__main__':
    main()

