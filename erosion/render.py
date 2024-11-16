import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid

#state - 3D array each entry- 0 air 1 water 2 rock
#create mesh out of that 

def render(state):
    #do marching cubes do generate rendering?
    vertsW, facesW, normalsW, valuesW = measure.marching_cubes(state, 0)
    vertsR, facesR, normalsR, valuesR = measure.marching_cubes(state, 1)

    #plot the thingy
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    #create mesh for rock
    meshR = Poly3DCollection(vertsR[facesR])
    meshR.set_color("sienna")
    meshR.set_edgecolor("k")
    ax.add_collection3d(meshR)

    #create mesh for water
    meshW = Poly3DCollection(vertsW[facesW])
    meshW.set_color("blue")
    meshW.set_edgecolor("k")
    ax.add_collection3d(meshW)

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_zlim(0)

    plt.tight_layout()
    plt.show()

arr = np.zeros((5, 5, 5))

for x in range(5):
    for y in range(5):
        for z in range(5):
            if z < 2:
                arr[x, y, z] = 2
            elif z < 3:
                arr[x, y, z] = 1
            else:
                arr[x, y, z] = 0

render(arr)