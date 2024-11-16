from gooey import Gooey
from argparse import ArgumentParser, FileType
from .erosion import Erosion, render

@Gooey
def main():
    parser = ArgumentParser(prog="erosion-simulator")

    args = parser.parse_args()

if __name__ == '__main__':
    main()