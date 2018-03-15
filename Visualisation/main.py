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

###################################################################################

# Generate an isosurface
contours = vtk.vtkMarchingCubes()
contours.SetInputConnection( readerVolume.GetOutputPort() )
contours.ComputeNormalsOn()
contours.ComputeGradientsOn()
contours.SetValue(0,1500)
#contours.SetValue( 0, int(0.03*maximumValue ))  # Bone isovalue
#for head, 800 gives skin, 1500 gives skull
#for brain, use 1500
#for bunny, use 2000

confilter = vtk.vtkPolyDataConnectivityFilter()
confilter.SetInputConnection(contours.GetOutputPort())
confilter.SetExtractionModeToLargestRegion()

# Take the isosurface data and create geometry
geoBoneMapper = vtk.vtkPolyDataMapper()
geoBoneMapper.SetInputConnection( confilter.GetOutputPort() )
geoBoneMapper.ScalarVisibilityOff()

# Take the isosurface data and create geometry
actorBone = vtk.vtkLODActor()
actorBone.SetNumberOfCloudPoints( 1000000 )
actorBone.SetMapper( geoBoneMapper )
actorBone.GetProperty().SetColor( 1, 1, 1 )
actorBone.GetProperty().SetOpacity( 1.0 )

###################################################################################

# Generate an isosurface
contoursSkin = vtk.vtkMarchingCubes()
contoursSkin.SetInputConnection( readerVolume.GetOutputPort() )
contoursSkin.ComputeNormalsOn()
contoursSkin.ComputeGradientsOn()
contoursSkin.SetValue(0,800)
#contours.SetValue( 0, int(0.03*maximumValue ))  # Bone isovalue
#for head, 800 gives skin, 1500 gives skull
#for brain, use 1500
#for bunny, use 2000

confilterSkin = vtk.vtkPolyDataConnectivityFilter()
confilterSkin.SetInputConnection(contoursSkin.GetOutputPort())
confilterSkin.SetExtractionModeToLargestRegion()

# Take the isosurface data and create geometry
geoBoneMapperSkin = vtk.vtkPolyDataMapper()
geoBoneMapperSkin.SetInputConnection( confilterSkin.GetOutputPort() )
geoBoneMapperSkin.ScalarVisibilityOff()

# Take the isosurface data and create geometry
actorSkin = vtk.vtkLODActor()
actorSkin.SetNumberOfCloudPoints( 1000000 )
actorSkin.SetMapper( geoBoneMapperSkin )
actorSkin.GetProperty().SetColor( 0, 1, 0 )
actorSkin.GetProperty().SetOpacity( 0.5 )

####################################################################################

# Create renderer
ren = vtk.vtkRenderer()
ren.SetBackground( 0.329412, 0.34902, 0.427451 ) 
ren.AddActor(actorBone)
ren.AddActor(actorSkin)

# Create a window for the renderer of size 250x250
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(1000, 1000)

# Set an user interface interactor for the render window
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Start the initialization and rendering
iren.Initialize()
renWin.Render()
iren.Start()