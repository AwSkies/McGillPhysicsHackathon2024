import numpy as np



class Erosion():
    def __init__(self):
        self.Nparticles = 1

        self.dimx, self.dimy, self.dimz = (10, 10, 10)
        self.dims = np.array([self.dimx, self.dimy, self.dimz])

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
        self.onesvecs = []
        self.onesvecszero = []
        for a1 in range(2):
            for a2 in range(2):
                for a3 in range(2):
                    self.onesvecs.append(np.array([a1, a2, a3]))
                    self.onesvecszero.append(np.array([0, a1, a2, a3]))

    #grid dimension



    def gravity(self):
        self.particle_velocities += self.gDt

    def particles_to_grid_velocities(self):
        for i in range(3):
            grid_locations = np.floor((self.particle_positions - self.offsets[i])/self.h)
            grid_offsets = np.remainder((self.particle_positions-self.offsets[i]),self.h)/self.h
            self.grid_velocities *= 0
            grid_velocity_weights = np.zeros(self.dims3)
            griv_velocity_weights_product = np.zeros(self.dims3)
            for j in range(8):
                grid_velocity_weights[grid_locations +
                                      self.onesvecs[j], i] += np.abs(np.prod(self.onesvecs[j] - grid_offsets, axis=1))
                print(grid_velocity_weights)







erosion = Erosion()

erosion.particles_to_grid_velocities()





