import numpy as np


Nparticles = 1000

particle_positions = np.zeros((Nparticles,3))

particle_velocities = np.zeros((Nparticles,3))

g = np.array([0, 0, -9.8])

def gravity():
    global particle_velocities
    particle_velocities += g

def particles_to_grid():
    pass







