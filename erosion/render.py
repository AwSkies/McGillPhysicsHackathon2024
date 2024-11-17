import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

class Renderer:
    def __init__(self, edges):
        # Make plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.edges = edges

    def process_frame(self, state):
        self.ax.clear()
        shape = np.shape(state)

        # Split water and rock data into separate arrays
        def split_state(val):
            arr = np.zeros(shape)
            for i in np.ndindex(shape):
                arr[i] = 1 if state[i] == val else -1
            return arr
        
        def render_surface(state, color, alpha):
            try:
                verts, faces, normals, values = measure.marching_cubes(state, 0)
                mesh = Poly3DCollection(verts[faces])
                mesh.set_color(color)
                if self.edges:
                    mesh.set_edgecolor("k")
                mesh.set_alpha(alpha)
                return self.ax.add_collection3d(mesh)
            except ValueError:
                pass

        self.ax.set_xlim(0, shape[0])
        self.ax.set_ylim(0, shape[1])
        self.ax.set_zlim(0, shape[2])
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        return (render_surface(split_state(2), "sienna", 0.75), render_surface(split_state(1), "blue", 0.5))

    def format(self):
        plt.tight_layout()

    def render(self, frames, file):
        anim = animation.FuncAnimation(self.fig, self.process_frame, frames, interval = 300, save_count=75)
        self.format()
        anim.save(file, "pillow")
        plt.show()
