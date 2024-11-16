# Remove this import once it's working since it should be able to get input directly from the simulator
import numpy as np

from gooey import Gooey
from argparse import ArgumentParser, FileType
from erosion import Erosion, Renderer

@Gooey
def main():
    parser = ArgumentParser(prog="erosion-simulator")

    args = parser.parse_args()

    sim = Erosion()
    ren = Renderer()

    def anim_states():
        for j in range(0, 10):
            shp = (25, 25, 25)
            arr = np.zeros(shp)

            for i in np.ndindex(shp):
                if ((i[0] < shp[0] / 4 or i[0] > shp[0] / (4 / 3)) and i[2] < shp[2] / 2) or ((i[0] > shp[0] / 4 or i[0] < shp[0] / (4 / 3)) and i[2] < 10 - j):
                    arr[i] = 2
                elif i[2] < shp[2] / 2:
                    arr[i] = 1
                else:
                    arr[i] = 0
            
            yield arr

    ren.render(anim_states, "anim.gif")

if __name__ == '__main__':
    main()