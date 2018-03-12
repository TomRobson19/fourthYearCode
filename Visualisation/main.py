import vtk
import sys
import os

import argparse

parser = argparse.ArgumentParser(description="Generate an isosurface")
parser.add_argument('xPixelsPerImage',type=int)
parser.add_argument('yPixelsPerImage',type=int)
parser.add_argument('noOfImages',type=int)
parser.add_argument('imageFiles',type=str,nargs='+')

args = parser.parse_args()


