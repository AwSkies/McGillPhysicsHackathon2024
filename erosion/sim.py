import numpy as np
import math
import random


class Erosion():
    def __init__(self):
        self.Nparticles = 20

        self.dimx, self.dimy, self.dimz = (30, 10, 10)
        self.dims = np.array([self.dimx, self.dimy, self.dimz])

        self.h = 1
        self.V = 1
        self.A = 300
        self.H = 7
        self.areas = np.zeros(self.dimx)
        self.rocks = np.ones(self.dims, dtype=bool)
        self.rocks[:, 3:7, 4:] = False
        self.rocks[:, :, 9:] = False

        print(self.rocks)


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
        # self.rocks = np.zeros(self.dims, dtype=bool)
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


        self.simsteps = 10

    #grid dimension


    '''
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

'''
    # def process_rocks(self):
    #     rocks_killlist = np.zeros(self.dims, dtype=bool)
    #     for i in range(self.dims[0]):
    #         for j in range(self.dims[1]):
    #             for k in range(self.dims[2]):
    #                 '''
    #                 if(not(i == 0 or i == self.dims[0] - 1 or j==0 or j == self.dims[1] - 1 or k == 0 or k == self.dims[2] - 1) and
    #                 ((not self.rocks[i - 1][j][k] and not self.rocks[i + 1][j][k] and
    #                     not self.rocks[i][j - 1][k] and not self.rocks[i][j + 1][k] and
    #                     not self.rocks[i][j][k - 1] and not self.rocks[i][j][k + 1]))):
    #                     self.kill_rock(i, j, k)
    #                 print(i, j, k, self.rocks[2][2][2])
    #                 '''
    #                 rocks_killlist[i,j,k] = self.erode_rock(i, j, k)
    #                 self.kill_rock(rocks_killlist)

    # def erode_rock(self, x, y, z):
    #     rock_HP = 0.0
    #     damage = 6
    #     Median_fluid_velocity = 1000
    #     normal_vector, flow_vector, rock_HP = self.rock_character(x, y, z)
    #     normal_magnitude = math.sqrt(normal_vector[0]**2 + normal_vector[1]**2 + normal_vector[2]**2)
    #     print("Position", x, y, z, 'RockHP', rock_HP, "Vector", normal_vector[0], normal_vector[1], normal_vector[2])
    #     if(normal_magnitude == 0):
    #         return True
    #     cosine = np.dot(normal_vector, flow_vector)
    #     flow_magnitude = flow_vector[0]**2 + flow_vector[1]**2 + flow_vector[2]**2
    #     mean = cosine * flow_magnitude / Median_fluid_velocity
    #     damage = self.gaussian_random(mean, 1, 0, 5.99)
    #     if(damage > rock_HP):
    #         return True

    # def kill_rock(self, rocks_killlist):
    #     for i in range(self.dims[0]):
    #         for j in range(self.dims[1]):
    #             for k in range(self.dims[2]):
    #                 if(rocks_killlist[i][j][k]):
    #                     self.rocks[i, j, k] = False

    # def gaussian_random(self, mean, std_dev, min_val, max_val):
    #     while True:
    #         value = np.random.normal(mean, std_dev)
        
    #         if min_val <= value < max_val:
    #             return value

    # def rock_character(self, x, y, z):
    #     normal_smoothing = 1
    #     rock_HP = 0.0
    #     normal_vector = np.array([0.0, 0.0, 0.0])
    #     flow_vector = np.array([0.0, 0.0, 0.0])
    #     flow_magnitude = 0
    #     if(x == 0 or x == self.dims[0] - 1 or y == 0 or y == self.dims[1] - 1 or z == 0 or z == self.dims[2] - 1):
    #         return normal_vector, flow_vector, rock_HP
    #     else:
    #         for i in range(x-1, x+1+1):
    #             for j in range(y-1, y+1+1):
    #                 for k in range(z-1, z+1+1):
    #                     normal_magnitude = math.sqrt((x-i)**2 + (y-j)**2 + (z-k)**2)
    #                     #flow_magnitude = math.sqrt((x-i)**2 + (y-j)**2 + (z-k)**2)
    #                     if(normal_magnitude != 0 and self.rocks[i, j, k]):
    #                         normal_vector[0] = normal_vector[0] + ((x-i) / normal_magnitude)
    #                         normal_vector[1] = normal_vector[1] + ((y-j) / normal_magnitude)
    #                         normal_vector[2] = normal_vector[2] + ((z-k) / normal_magnitude)
    #                     if(normal_magnitude != 0 and self.rocks[i, j, k]):
    #                         flow_vector[0] = flow_vector[0] + self.grid_velocities[i][j][k][0] * ((x-i) / normal_magnitude)
    #                         flow_vector[1] = flow_vector[1] + self.grid_velocities[i][j][k][1] * ((y-j) / normal_magnitude)
    #                         flow_vector[2] = flow_vector[2] + self.grid_velocities[i][j][k][2] * ((z-k) / normal_magnitude)
    #                         if(((i==x and j==y) or
    #                             (j==y and k==z) or
    #                             (i==x and k==z))):
    #                                 rock_HP = rock_HP + 1
    #         normal_magnitude = math.sqrt(normal_vector[0]**2 + normal_vector[1]**2 + normal_vector[2]**2)
    #         if(normal_magnitude != 0):
    #             normal_vector = normal_vector / normal_magnitude
    #         flow_magnitude = math.sqrt(flow_vector[0]**2 + flow_vector[1]**2 + flow_vector[2]**2)
    #         if(flow_magnitude != 0):
    #             flow_vector = flow_vector / flow_magnitude
    #         return normal_vector, flow_vector, rock_HP

#if __name__ == "__main__":


    def simulate(self):
            # do simulation n times
            self.H -= 1
            # for i in range(self.simsteps):
            #     pass
            state = np.zeros(self.dims)
            state[:,:,:self.H+1] = 1
            state[self.rocks] = 2
            # print(state)
            return state


# def test():
#     for i in range(10):
#         yield i

if __name__ == "__main__":
    erosion = Erosion()
    for state in erosion.simulate():
        print("a")


# erosion.rocks[2][2][2] = True

# erosion.rocks[2][2][1] = True
# erosion.rocks[2][2][3] = True
# erosion.rocks[2][1][2] = True
# erosion.rocks[2][3][2] = True
# erosion.rocks[1][2][2] = True
# erosion.rocks[3][2][2] = True
# #erosion.rocks[1][2][4] = True
# erosion.process_rocks()
        




#erosion = Erosion()

#erosion.particles_to_grid_velocities()





