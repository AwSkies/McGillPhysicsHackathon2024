import numpy as np
import math


class Erosion():
    def __init__(self):
        self.Nparticles = 4

        self.dimx, self.dimy, self.dimz = (5, 5, 5)
        self.dims = np.array([self.dimx, self.dimy, self.dimz])

        self.Dt = 0.01

        self.g = np.array([0, 0, 9.8])

        self.gDt = self.g*self.Dt
        self.particle_positions = np.random.uniform(size=(self.Nparticles,3)) * self.dims

        self.particle_velocities = np.zeros((self.Nparticles,3))

        self.grid_velocities = np.zeros((self.dimx, self.dimy, self.dimz, 3))
        self.rocks = np.zeros(self.dims, dtype=bool)

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


    '''
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

    '''
    def process_rocks(self):
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                for k in range(self.dims[2]):
                    '''
                    if(not(i == 0 or i == self.dims[0] - 1 or j==0 or j == self.dims[1] - 1 or k == 0 or k == self.dims[2] - 1) and
                    ((not self.rocks[i - 1][j][k] and not self.rocks[i + 1][j][k] and
                        not self.rocks[i][j - 1][k] and not self.rocks[i][j + 1][k] and
                        not self.rocks[i][j][k - 1] and not self.rocks[i][j][k + 1]))):
                        self.kill_rock(i, j, k)
                    print(i, j, k, self.rocks[2][2][2])
                    '''
                    self.erode_rock(i, j, k)

    def erode_rock(self, x, y, z):
        normal_vector, flow_vector = self.flow_vectors(x, y, z)
        cosine = np.dot(normal_vector, flow_vector)


        

    
    def kill_rock(self, x, y, z):
        self.rocks[x, y, z] = False
        #print("rock killed")

    def flow_vectors(self, x, y, z):
        normal_smoothing = 1
        normal_vector = np.array([0.0, 0.0, 0.0])
        flow_vector = np.array([0.0, 0.0, 0.0])
        flow_magnitude = 0
        if(x == 0 or x == self.dims[0] - 1 or y == 0 or y == self.dims[1] - 1 or z == 0 or z == self.dims[2] - 1):
            return normal_vector 
        else:
            for i in range(x-1, x+1+1):
                for j in range(y-1, y+1+1):
                    for k in range(z-1, z+1+1):
                        normal_magnitude = math.sqrt((x-i)**2 + (y-j)**2 + (z-k)**2)
                        #flow_magnitude = math.sqrt((x-i)**2 + (y-j)**2 + (z-k)**2)
                        if(normal_magnitude != 0 and self.rocks[i, j, k]):
                            normal_vector[0] = normal_vector[0] + ((x-i) / normal_magnitude)
                            normal_vector[1] = normal_vector[1] + ((y-j) / normal_magnitude)
                            normal_vector[2] = normal_vector[2] + ((z-k) / normal_magnitude)
                        if(normal_magnitude != 0 and self.rocks[i, j, k]):
                            flow_vector[0] = flow_vector[0] + self.grid_velocities[0] * ((x-i) / normal_magnitude)
                            flow_vector[1] = flow_vector[1] + self.grid_velocities[1] * ((y-j) / normal_magnitude)
                            flow_vector[2] = flow_vector[2] + self.grid_velocities[2] * ((z-k) / normal_magnitude)
            
            normal_magnitude = math.sqrt(normal_vector[0]**2 + normal_vector[1]**2 + normal_vector[2]**2)
            if(normal_magnitude != 0):
                normal_vector = normal_vector / normal_magnitude
            flow_magnitude = math.sqrt(flow_vector[0]**2 + flow_vector[1]**2 + flow_vector[2]**2)
            if(flow_magnitude != 0):
                flow_vector = flow_vector / flow_magnitude
            return normal_vector, flow_vector

print("Hello")
erosion = Erosion()
erosion.rocks[2][2][2] = True
#erosion.rocks[1][2][1] = True
#erosion.rocks[1][2][2] = True
#erosion.rocks[1][2][3] = True
#erosion.rocks[1][2][4] = True
print(erosion.rocks[2][2][2])
erosion.process_rocks()
print(erosion.rocks[2][2][2])
        




#erosion = Erosion()

#erosion.particles_to_grid_velocities()





