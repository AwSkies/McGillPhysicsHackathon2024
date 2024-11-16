import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#state - 3D array each entry- 0 air 1 water 2 rock
#create mesh out of that create open USD file
#filename- USD file name
#

def render(state, filename):
    #do marching cubes do generate rendering?
    vertsW, facesW, normalsW, valuesW = measure.marching_cubes(state, 1)
    vertsR, facesR, normalsR, valuesR = measure.marching_cubes(state, 2)

    #plot the thingy
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    #create mesh for rock
    meshR = Poly3DCollection(vertsR[facesR])
    meshR.set_color("brown")