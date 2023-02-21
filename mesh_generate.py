import vtk
import pygmsh

with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        mesh_size = 0.1,
    )
    mesh = geom.generate_mesh()

    mesh.write("plate.vtk")

# 加载VTK文件
reader = vtk.vtkPolyDataReader()
reader.SetFileName("plate.vtk")
reader.Update()

# 创建一个mapper，用于将数据映射到图形中
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(reader.GetOutputPort())

# 创建一个actor，用于在3D空间中渲染图形
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# 创建一个reneder，用于渲染actor
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.2, 0.4)

# 创建一个render window，用于显示图形
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(400, 400)

# 创建一个render window interactor，用于与用户交互
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# 开始渲染并显示
interactor.Initialize()
render_window.Render()
interactor.Start()
