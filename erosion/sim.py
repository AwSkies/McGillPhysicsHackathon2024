import numpy as np
import math
import random


class Erosion():
    def __init__(self):
        self.Nparticles = 20

        self.dimx, self.dimy, self.dimz = (32*4, 16*4, 32*4)
        self.dims = np.array([self.dimx, self.dimy, self.dimz])
        self.xs = np.arange(0, self.dimx, 1)

        self.h = 1
        self.V = 1
        self.A = 20*16
        self.AV = self.A*self.V
        self.H = 7
        self.areas = np.zeros(self.dimx)
        self.velocities = np.zeros(self.dimx)
        self.rocks = np.ones(self.dims, dtype=bool)
        self.rocks[:, 7*4:9*4, (20)*4:] = False
        self.rocks[:10*4, 5*4:11*4, (20)*4:] = False
        self.rocks[22*4:, 5*4:11*4, (20)*4:] = False
        self.rocks[:, :, (29)*4:] = False
        self.rock_hardness = np.ones(self.dims)
        self.rock_hardness[:,:, self.dimz//3:2*self.dimz//3] = 2
        # self.rocks = np.roll(self.rocks, -2, axis=1)
        self.edgerocks = np.zeros(shape=self.dims, dtype=int)
        self.bottomrocks = np.zeros(shape=self.dims, dtype=bool)
        self.step = 0
        self.edgerock1_threshold = 0.03/10*3
        self.bottomrock_threshold = 0.1/10*3
        self.edgerock2_threshold = 0.6/10

        self.smoothing_factor = 8




        # print(self.rocks)


        self.Dt = 0.01
        
        self.g = np.array([0, 0, 9.8])

        self.gDt = self.g*self.Dt
        np.random.seed(123)
        self.particle_positions = np.random.uniform(size=(self.Nparticles,3)) * (self.dims - 2) + 1
        # self.particle_positions = np.array([[2.2, 2.1, 2.1]])

        self.simsteps = 40




    def getAstrip(self, x, z):
        return np.count_nonzero(np.logical_not(self.rocks[x,:,z]))

    def getArea(self, x):
        return np.count_nonzero(np.logical_not(self.rocks[x,:,:self.H+1]))

    def calculateAreas(self):
        max_H = 0
        for x in range(5):
            area = 0
            for z in range(self.dimz):
                area += self.getAstrip(0, z)
                if area > self.A:
                    H = z
                    break
            area2 = area-self.getAstrip(0, self.H)
            if abs(area2 - self.A) < (self.A - area):
                H -= 1
            if H > max_H:
                max_H = H

        self.H = max_H

        self.areas = np.count_nonzero(np.logical_not(self.rocks[:,:,:self.H+1]),axis=(1,2))
        # print(self.areas)
        #
        # for x in self.xs:
        #     self.areas[x] = self.getArea(x)
        # print(self.areas)


    def calculateVelocities(self):
        self.velocities = self.AV/self.areas
        # self.velocities = np.random.uniform(shape=(self.dimx))
        # self.velocities = np.array([[self.velocities]])
        # self.velocities = np.repeat(self.velocities,self.dimy,axis=1)
        self.velocities = np.tile(self.velocities,(self.dimz,self.dimy, 1))
        self.velocities=np.transpose(self.velocities,axes=(2,1,0))
        # print(self.velocities.shape)
        # print(self.velocities[4,8,1])

    # def calculateEdgeRocks(self):
    #     self.edgerocks[:, :, 0:self.H + 1] += np.logical_and(self.rocks[:, :, 0:self.H + 1],
    #                                             np.logical_not(
    #                                                              np.roll(self.rocks[:, :, 0:self.H + 1], 1, axis=1)))
    #     # print(self.rocks[:,:,0:self.H+1])
    #     self.edgerocks[:, :, 0:self.H + 1] += np.logical_and(self.rocks[:, :, 0:self.H + 1],
    #                                                          np.logical_not(
    #                                                              np.roll(self.rocks[:, :, 0:self.H + 1], -1, axis=1)))


    def calculateEdgeRocks(self):
        # print(self.rocks[4, :, 5])
        # print(np.roll(self.rocks[:, :, 5:6],1,axis=1))
        self.edgerocks[:,:,0:self.H+1] += np.logical_and(self.rocks[:,:,0:self.H+1],
                                        np.logical_not(np.roll(self.rocks[:,:,0:self.H+1],1,axis=1)))
        # print(self.rocks[:,:,0:self.H+1])
        self.edgerocks[:,:,0:self.H+1] += np.logical_and(self.rocks[:,:,0:self.H+1],
                                        np.logical_not(np.roll(self.rocks[:,:,0:self.H+1],-1,axis=1)))
        self.bottomrocks[:,:,0:self.H] += np.logical_and(self.rocks[:,:,0:self.
                                                         H],np.logical_not(np.roll(self.rocks[:,:,0:self.H],-1,axis=2)))
        # print(np.count_nonzero(self.edgerocks[:,:,8]))




        # self.edgerocks_old = np.copy(self.edgerocks)
        # self.edgerocks[:] = 0
        # for x in range(self.dimx):
        #     for y in range(self.dimy-1):
        #         for z in range(self.H+1):
        #             if self.rocks[x,y,z] and not self.rocks[x,y+1,z]:
        #                 self.edgerocks[x,y,z] += 1
                            # print("EEE")
        #             except:
        #                 pass
        # try:
        #     if self.rocks[x, y, z] and not self.rocks[x, y - 1, z]:
        #         self.edgerocks[x, y, z] += 1
        # except:
        #     pass
            # print(self.rocks[x,y+1,z], self.rocks[x,y-1,z])
        # print(self.edgerocks == self.edgerocks.astype(int))
        # print(self.edgerocks)
        # print(edgerocks)

    def remove_rocks(self):
        random_field = np.random.uniform(size=(self.dims))/self.velocities #/self.rock_hardness
        edge_rocks_to_remove = np.logical_and(self.edgerocks, random_field < self.edgerock1_threshold)
        bottom_rocks_to_remove = np.logical_and(self.bottomrocks, random_field < self.bottomrock_threshold)
        rocks_to_remove = np.logical_or(edge_rocks_to_remove, bottom_rocks_to_remove)
        self.rocks[rocks_to_remove] = False

        # np.logical_and(self.rocks & np.logical_not(self.rocks), np.logical_not(self.rocks))

    def smooth_state(self, fact):
        compressedarray = np.zeros((self.dimx//fact, self.dimy//fact, self.dimz//fact))
        for i in range(fact):
            for j in range(fact):
                for k in range(fact):
                    compressedarray += self.rocks[i::fact, j::fact, k::fact]
        smoothed_rock_array = compressedarray > 0.5*fact**3
        return smoothed_rock_array



    def simulate(self):
        for i in range(self.simsteps):
            self.calculateAreas()
            self.calculateVelocities()
            self.calculateEdgeRocks()
            self.remove_rocks()
            self.step += 1
        # do simulation n times
        # self.H -= 1
        # for i in range(self.simsteps):
        #     pass
        smoothed_rocks = self.smooth_state(self.smoothing_factor)
        # print(smoothed_rocks)
        state = np.zeros(self.dims//self.smoothing_factor)
        state[:,:,:self.H//self.smoothing_factor+1] = 1
        state[smoothed_rocks] = 2
        state = np.rot90(state,axes=(0,1))
        # print(state)
        return state






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





# def test():
#     for i in range(10):
#         yield i

if __name__ == "__main__":
    erosion = Erosion()
    erosion.simulate()
    # for state in erosion.simulate():

        # print("a")


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





