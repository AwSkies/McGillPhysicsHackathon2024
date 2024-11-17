import numpy as np



class Erosion():
    def __init__(self):
        self.Nparticles = 20

        self.dimx, self.dimy, self.dimz = (5, 5, 5)
        self.dims = np.array([self.dimx, self.dimy, self.dimz])

        self.Dt = 0.01

        self.g = np.array([0, 0, 9.8])

        self.gDt = self.g*self.Dt
        np.random.seed(123)
        self.particle_positions = np.random.uniform(size=(self.Nparticles,3)) * (self.dims - 2) + 1
        # self.particle_positions = np.array([[2.2, 2.1, 2.1]])

        self.isWater = np.zeros(self.dims, dtype=bool)
        self.particle_velocities = np.zeros((self.Nparticles,3))
        self.particle_velocities = np.random.uniform(size=(self.Nparticles,3))

        self.grid_velocities = np.zeros((self.dimx, self.dimy, self.dimz, 3))
        self.rocks = np.zeros(self.dims, dtype=bool)
        self.dims3 = (self.dimx, self.dimy, self.dimz, 3)
        self.s = np.ones(self.dims3)
        self.s[self.rocks] = 0
        self.s[np.roll(self.rocks, 1, axis=0), 0] = 0
        self.s[np.roll(self.rocks, 1, axis=1), 1] = 0
        self.s[np.roll(self.rocks, 1, axis=2), 2] = 0
        self.o = 1.9

        self.h = 1
        self.oneoffsets = [np.array([0, 1, 1]), np.array([1, 0, 1]), np.array([1, 1, 0])]
        self.offsets = [self.h/2* x for x in self.oneoffsets]

        self.particle_density = np.zeros(self.dims)
        self.grid_offsets = 0
        self.grid_locations = 0
        self.onesvecs = []
        self.onesvecszero = []


        for a1 in range(2):
            for a2 in range(2):
                for a3 in range(2):
                    self.onesvecs.append(np.array([a1, a2, a3]))
                    self.onesvecszero.append(np.array([0, a1, a2, a3]))
        self.onesvecsopposite = [1 - x for x in self.onesvecs]


        self.simstep = 10000

    #grid dimension



    def gravity(self):
        self.particle_velocities += self.gDt

    def particles_to_grid_velocities(self):
        hasVel = np.zeros(self.dims3, dtype=bool)
        for i in range(1):
            self.grid_locations = np.floor((self.particle_positions - self.offsets[i])/self.h)
            self.grid_offsets = np.remainder((self.particle_positions-self.offsets[i]),self.h)/self.h
            # print(grid_offsets)
            # print(grid_locations)
            # self.grid_velocities *= 0
            grid_velocity_weights = np.zeros(self.dims3)
            grid_velocity_weights_product = np.zeros(self.dims3)
            for j in range(8):
                for p in range(self.grid_offsets.shape[0]):
                    hasVel[tuple((self.grid_locations[p]).astype(int)), i] = True
                    w = np.abs(np.prod(self.onesvecsopposite[j] - self.grid_offsets[p]))
                    index = tuple((self.grid_locations[p] + self.onesvecs[j]).astype(int))
                    grid_velocity_weights[index, i] += w
                    grid_velocity_weights_product[index, i] += w*self.particle_velocities[p,i]

            # print(hasVel)
            self.grid_velocities[hasVel] = grid_velocity_weights_product[hasVel] / grid_velocity_weights[hasVel]
            # print(self.grid_velocities)

    def non_compress(self):
        grid_locations = np.floor((self.particle_positions) / self.h)
        self.isWater[:] = False
        # print(self.particle_positions)
        self.particle_density[:] = 0
        for p in range(grid_locations.shape[0]):
            self.particle_density[tuple(grid_locations[p].astype(int))] += 1

        self.isWater = self.particle_density != 0

        for i in range(10):
            d = -self.grid_velocities[:,:,:,0] + np.roll(self.grid_velocities[:,:,:,0], -1, axis=0)
            d += -self.grid_velocities[:, :, :, 1] + np.roll(self.grid_velocities[:, :, :, 0], -1, axis=1)
            d += -self.grid_velocities[:, :, :, 2] + np.roll(self.grid_velocities[:, :, :, 0], -1, axis=2)

            d*=self.o
            # print(d)

            s = -self.s[:, :, :, 0] + np.roll(self.s[:, :, :, 0], -1, axis=0)
            s += -self.s[:, :, :, 1] + np.roll(self.s[:, :, :, 0], -1, axis=1)
            s += -self.s[:, :, :, 2] + np.roll(self.s[:, :, :, 0], -1, axis=2)

            self.grid_velocities[:, :, :, 0] += d*np.roll(self.s[:, :, :, 0], 1, axis=0)/s
            self.grid_velocities[:, :, :, 0] -= np.roll(d, 1, axis=0)*self.s[:,:,:,0]/s
            self.grid_velocities[:, :, :, 0] += d * np.roll(self.s[:, :, :, 1], 1, axis=1) / s
            self.grid_velocities[:, :, :, 0] -= np.roll(d, 1, axis=1) * self.s[:, :, :, 1] / s
            self.grid_velocities[:, :, :, 0] += d * np.roll(self.s[:, :, :, 2], 1, axis=2) / s
            self.grid_velocities[:, :, :, 0] -= np.roll(d, 1, axis=2) * self.s[:, :, :, 2] / s

    def grid_velocities_to_part(self):
        self.particle_velocities[:,:,:,0] = (self.grid_velocities(tuple(self.grid_locations)) + np.roll(self.grid_locations, -1, axis = 0))

        # print(self.isWater)







                    # grid_velocity_weights[tuple((grid_locations + self.onesvecs[j]).astype(int)), i] += np.abs(np.prod(self.onesvecs[j] - grid_offsets, axis=1))
            # print(grid_velocity_weights[2,2,1,0])


    def simulate(self):
        while 1:
            for i in range(self.simsteps):
                self.gravity()
                self.particles_to_grid_velocities()
                yield self.rocks, self.particle_positions, self.particle_velocities, self.grid_velocities




erosion = Erosion()

erosion.particles_to_grid_velocities()
erosion.non_compress()




