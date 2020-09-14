import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian, barycentric
class MyPde:
    """
    -\Delta u + 3u = f
    u = x^2+y^2
    """
    def __init__(self):
        pass
    def domain(self):
        return np.array([0,1,0,1])#模型的求解区域

    @cartesian
    def solution(self, p):#模型的真解
        """
        The exact solution parameters-----
        p :(....,2)==>(2,), (10,2), (3,10,2)

        Examples
        ------
        p = np.array([0,1],dtype=np.float64)
        p = np.array([[0,1],[0.5, 0.5]],dtype=np.float64)

        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = x**2 + y**2
        return val #val.shape == x.shape

    @cartesian
    def source(self, p):
        """The right hand side of Possion equation
        INPUT:
            p: array object,
        """
        x = p[...,0]
        y = p[...,1]
        pi= np.pi
        val = -4+3*(x**2+y**2)
        return val

    @cartesian
    def gradient(self, p):
        """The gradient of exact equation
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[...,0] = 2*x
        val[...,1] = 2*y
        return val

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def neumann(self, p, n):
        """
        Neumann boundary condition
        parameters
        -------

        p: (NQ, NE, 2)
        n:(NE, 2)
        grad*n:(NQ,NE,2)
        """
        grad = self.gradient(p)#(NQ,NE,2)
        val = np.sum(grad*n,axis = -1)#(NQ,NE)
        return val

    @cartesian
    def robin(self, p, n):
        grad = self.gradient(p)#(NQ,NE,2)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1,0], dtype=np.float64).reshape(shape)
        val += self.solution(p)
        return val, kappa



pde = MyPde()

# print(pde.solution.coordtype)
mf = MeshFactory()
box = [0,1,0,1]
mesh = mf.boxmesh2d(box, nx=40, ny=40, meshtype='tri')
space = LagrangeFiniteElementSpace(mesh,p=1)
#space提供一个插值函数  uI
uI = space.interpolation(pde.solution)#空间中的有限元函数，也是一个数组
# print('uI[0:10]:',uI[0:10])#打印前面10个自由度的值
bc = np.array([1/3,1/3,1/3],dtype=mesh.ftype)
val = uI(bc)#(1, NC）有限元函数在每个单元的重心处的函数值
# print('val[0:10]:',val[1:10])

val0 = uI(bc)#(NC,）
val1 = uI.grad_value(bc)#(NC,2)
# print('val0[0:10]:',val0[1:10])
# print('val1[0:10]:',val1[1:10])

#插值误差
error0 = space.integralalg.L2_error(pde.solution, uI)
error1 = space.integralalg.L2_error(pde.gradient, uI.grad_value)
# print('L2:',error0,'H1:',error1)
#
#
# fig = plt.figure()
#
# axes = fig.gca(projection = '3d')
# uI.add_plot(axes, cmap = 'rainbow')
#
# plt.show()