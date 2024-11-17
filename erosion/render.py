import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

class Renderer:
    def __init__(self):
        # Make plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def render_surface(self, state, color, alpha):
        try:
            verts, faces, normals, values = measure.marching_cubes(state, 0)
            mesh = Poly3DCollection(verts[faces])
            mesh.set_color(color)
            mesh.set_alpha(alpha)
            return self.ax.add_collection3d(mesh)
        except ValueError:
            pass
        
    # Split water and rock data into separate arrays
    def split_state(self, state, val):
        shape = np.shape(state)
        arr = np.zeros(shape)
        for i in np.ndindex(shape):
            arr[i] = 1 if state[i] == val else -1
        return arr

    def process_frame(self, state):
        self.ax.clear()
        shape = np.shape(state)

        self.ax.set_xlim(0, shape[0])
        self.ax.set_ylim(0, shape[1])
        self.ax.set_zlim(0, shape[2])
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        return (self.render_surface(self.split_state(state, 2), "sienna", 0.75), self.render_surface(self.split_state(state, 1), "blue", 0.5))

    def format(self):
        plt.tight_layout()

    def render(self, frames, file, number):
        print("Rendering...")
        anim = animation.FuncAnimation(self.fig, self.process_frame, frames, interval = 300, save_count=number)
        self.format()
        plt.show()
        print("Saving...")
        anim.save(file, "pillow")
