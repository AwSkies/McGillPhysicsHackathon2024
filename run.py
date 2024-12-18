# Remove this import once it's working since it should be able to get input directly from the simulator
import numpy as np

from gooey import GooeyParser, Gooey
from erosion import Erosion, Renderer

@Gooey
def main():
    parser = GooeyParser(prog="erosion-simulator")
    render_group  = parser.add_argument_group("Render Options")
    render_group.add_argument('output', help = "Output File Name of Animation", widget = 'FileSaver')
    render_group.add_argument('frames', type = int, help = "Number of frames to render", default = 75)
    render_group.add_argument('fps', type = int, help = 'Animation frames per second', default = 5)

    simulation_group = parser.add_argument_group("Simulation Options")
    simulation_group.add_argument('steps_per_frame', type = int, help = "Simulation steps per frame", default = 40)

    args = parser.parse_args()

    sim = Erosion(args.steps_per_frame)
    ren = Renderer()

    def advance_frame():
        for i in range(args.frames):
            arr1 = sim.simulate()
            yield arr1

    def test_animation_advance_frame():
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

    ren.render(advance_frame(), args.output, args.frames, args.fps)

if __name__ == '__main__':
    main()