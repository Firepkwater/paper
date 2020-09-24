import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pde_2d import MyPde
from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate, show_error_table

p = 1
n = 10
pde = MyPde()
domain = pde.domain()
mf = MeshFactory()
mesh = mf.boxmesh2d(domain, nx=n, ny=n, meshtype='tri')
NDof = np.zeros(4)
Ndof = np.zeros(4, dtype = mesh.itype)
errorMatrix = np.zeros((2, 4), dtype=mesh.ftype)
errorType = ['$||u - u_h||_0$','$||\\nabla u - \\nabla u_h||_0$']
for i in range(4):
    print('Step:', i)
    space = LagrangeFiniteElementSpace(mesh, p=p, q=4)
    NDof[i] = space.number_of_global_dofs()
    uh = space.function()#返回一个有限元函数，初始自由度值是 0
    A = space.stiff_matrix()
    M = space.mass_matrix()
    F = space.source_vector(pde.source)

    bc = DirichletBC(space, pde.dirichlet, threshold=pde)
    A, F = bc.apply(A, F, uh)
    uh[:] = spsolve(A, F)

    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh.value)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)

    if i < 3:
        mesh.uniform_refine()

fig = plt.figure()
axes = fig.gca(projection = '3d')
uh.add_plot(axes, cmap = 'rainbow')
showmultirate(plt, 0, NDof, errorMatrix, errorType, propsize=20)
plt.show()



