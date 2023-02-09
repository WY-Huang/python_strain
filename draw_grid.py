import pygmsh
import vtk

# with pygmsh.geo.Geometry() as geom:
#     geom.add_polygon(
#         [
#             [0.0, 0.0],
#             [1.0, 0.0],
#             [1.0, 1.0],
#             [0.0, 1.0],
#         ],
#         mesh_size = 0.1,
#     )
#     mesh = geom.generate_mesh()

#     mesh.write("plate.vtk")


aRenderer=vtk.vtkRenderer()
renWin=vtk.vtkRenderWindow()
renWin.AddRenderer(aRenderer)
iren=vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
vtkReader=vtk.vtkPolyDataReader()
vtkReader.SetFileName("plate.vtk")
vtkReader.Update()

skinMapper=vtk.vtkPolyDataMapper()
skinMapper.SetInputConnection(vtkReader.GetOutputPort())
skinMapper.ScalarVisibilityOff()
skin=vtk.vtkActor()
skin.SetMapper(skinMapper)
aCamera=vtk.vtkCamera()
aCamera.SetViewUp(0,0,-1)
aCamera.SetPosition(0,1,0)
aCamera.SetFocalPoint(0,0,0)
aCamera.ComputeViewPlaneNormal()
aCamera.Azimuth(30.0)
aCamera.Dolly(1.5)
aRenderer.AddActor(skin)
aRenderer.SetActiveCamera(aCamera)
aRenderer.SetBackground(.2,.3,.4)
aRenderer.ResetCameraClippingRange()
renWin.Render()
iren.Initialize()
iren.Start()
