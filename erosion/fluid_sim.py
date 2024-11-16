import numpy as np



class Erosion():
    def __init__(self):
        self.Nparticles = 10

        self.dimx, self.dimy, self.dimz = (1000, 100, 100)
        self.dims = np.array([dimx, dimy, dimz])

        self.Dt = 0.01

        self.g = np.array([0, 0, 9.8])

        self.gDt = self.g*self.Dt
        self.particle_positions = np.random.uniform(size=(self.Nparticles,3)) * self.dims

        self.particle_velocities = np.zeros((self.Nparticles,3))

        self.grid_velocities = np.zeros((self.dimx, self.dimy, self.dimz, 3))

        self.dims3 = (self.dimx, self.dimy, self.dimz, 3)
        self.h = 1
        self.offsets = [np.array([self.h/2, 0, 0]), np.array([0, self.h/2, 0]), np.array([0, 0, self.h/2])]
        self.oneoffsets = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

    #grid dimension



def gravity(self):
    global particle_velocities
    particle_velocities += self.gDt

def particles_to_grid_velocities(self):
    for i in range(3):
        grid_locations = np.floor((self.particle_positions - self.offsets[i])/self.h)
        grid_offsets = np.remainder((self.particle_positions-self.offsets[i]),self.h)
        self.grid_velocities *= 0
        grid_velocity_weights = np.zeros(self.dims3)
        grid_velocity_weights[grid_locations, i] = grid_offsets[:,:,:,0]*grid_offsets[:,:,:,1]*grid_offsets[:,:,:,2]

        for i in range(8):

        grid_velocity_weights[grid_locations]
        grid_velocities[grid_locations] +=


particles_to_grid_velocities()





