import numpy as np
from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import sys
import sympy
from mpl_toolkits.mplot3d import Axes3D
from fealpy.decorator import cartesian
from scipy.sparse.linalg import spsolve
from fealpy.boundarycondition import DirichletBC
from fealpy.pde.poisson_2d import CosCosData as PDE
from fealpy.tools.show import showmultirate, show_error_table

n = 10
p = 4
mf = MeshFactory()
mesh = mf.boxmesh2d([0, 1, 0, 1], nx=n, ny=n, meshtype='tri')
pde = PDE()
domain = pde.domain()
maxit = 4
errorType = ['$|| u - u_h||_{\Omega,0}$', # L2 误差
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$' ]# H1 误差

errorMatrix = np.zeros((2, maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float)
for i in range(maxit):
    print('Step:', i)
    space = LagrangeFiniteElementSpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()
    uh = space.function()
    a = np.array([(10.0, -1.0), (-1.0, 2.0)], dtype=np.float64)
    A = space.stiff_matrix(c=a)

    @cartesian
    def r(p):
        x = p[..., 0]
        y = p[..., 1]
        return 1 + x ** 2 + y ** 2
    M = space.mass_matrix(c=r)
    b = np.array([1.0, 1.0], dtype=np.float64)
    B = space.convection_matrix(c=b)
    @cartesian
    def f(p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        return 2*pi**2*np.sin(pi*x)*np.sin(pi*y)+12*pi**2*np.cos(pi*x)*np.cos(pi*y)+(-pi)*np.sin(pi*x)*np.cos(pi*y)-pi*np.cos(pi*x)*np.sin(pi*y)+(1+x**2+y**2)*np.cos(pi*x)*np.cos(pi*y)
    F = space.source_vector(f)
    A += B + M

    # 画出误差阶

    bc = DirichletBC(space, pde.dirichlet, threshold=pde)
    A, F = bc.apply(A, F, uh)
    uh[:] = spsolve(A, F).reshape(-1)
    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh.value)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value)
    # eta = residual_estimate(uh, pde.source)
    # eta = recovery_estimate(uh)
    # errorMatrix[2, i] = np.sqrt(np.sum(eta ** 2))

    if i < maxit - 1:
        mesh.uniform_refine()

# 画函数图像
fig = plt.figure()
axes = fig.gca(projection='3d')
uh.add_plot(axes, cmap='rainbow')

# 画收敛阶图像
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)

# 输出 latex 误差表格
show_error_table(NDof, errorType, errorMatrix)
plt.show()
