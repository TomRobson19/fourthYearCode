"""
python main.py 256 256 113 head/*

python main.py 256 256 109 brain/*

python main.py 512 512 360 bunny/*
"""
import sys
import os
import vtk
import argparse
import re

parser = argparse.ArgumentParser(description="Generate an isosurface...")
parser.add_argument('xPixelsPerImage',type=int)
parser.add_argument('yPixelsPerImage',type=int)
parser.add_argument('noOfImages',type=int)
parser.add_argument('imageFiles',type=str,nargs='+')
args = parser.parse_args()
xPixels = args.xPixelsPerImage
yPixels = args.yPixelsPerImage
noImages = args.noOfImages
images = args.imageFiles



def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


images = natural_sort(images)


vtkFiles = vtk.vtkStringArray()
for f in images:
	vtkFiles.InsertNextValue(f)

readerVolume = vtk.vtkImageReader()
readerVolume.SetDataScalarType( vtk.VTK_UNSIGNED_SHORT )
readerVolume.SetFileDimensionality( 2 )
readerVolume.SetDataExtent ( 0, xPixels-1, 0, yPixels-1, 0, noImages-1)

readerVolume.SetDataSpacing( 1,1,2 )
#readerVolume.SetDataSpacing( 1,1,1 ) # for bunny

readerVolume.SetNumberOfScalarComponents( 1 )
readerVolume.SetDataByteOrderToBigEndian()
readerVolume.SetFileNames( vtkFiles )

readerVolume.Update()
maximumValue = readerVolume.GetOutput().GetScalarRange()[1]
print(maximumValue)

# Create renderer
ren = vtk.vtkRenderer()
ren.SetBackground( 0, 0, 0 ) 
#ren.SetBackground( 1,0,1 ) 

# Create a window for the renderer of size 250x250
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(1000, 1000)

# Set an user interface interactor for the render window
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

thresholds = [0.35, 0.25]
colours = [(1,1,1), (0,1,0)]

# thresholds = [0.3]
# colours = [(1,1,1)]

# thresholds = [0.48, 0.45]
# colours = [(1,1,1), (0,1,0)]

# thresholds = [0.03]
# colours = [(1,1,1)]


for i in range(len(thresholds)):
	# Generate an isosurface
	contours = vtk.vtkMarchingCubes()
	contours.SetInputConnection( readerVolume.GetOutputPort() )
	contours.ComputeNormalsOn()
	contours.ComputeGradientsOn()
	contours.SetValue( 0, int(thresholds[i]*maximumValue ))  # isovalue

	confilter = vtk.vtkPolyDataConnectivityFilter()
	confilter.SetInputConnection(contours.GetOutputPort())
	confilter.SetExtractionModeToLargestRegion()

	# Take the isosurface data and create geometry
	geoBoneMapper = vtk.vtkPolyDataMapper()
	geoBoneMapper.SetInputConnection( confilter.GetOutputPort() )
	geoBoneMapper.ScalarVisibilityOff()

	actorBone = vtk.vtkLODActor()
	actorBone.SetNumberOfCloudPoints( 1000000 )
	actorBone.SetMapper( geoBoneMapper )
	actorBone.GetProperty().SetColor(colours[i])
	if i == 0:
		actorBone.GetProperty().SetOpacity( 1.0 )
	else:
		actorBone.GetProperty().SetOpacity( 0.5 )

	ren.AddActor(actorBone)

# Start the initialization and rendering
iren.Initialize()
renWin.Render()
iren.Start()