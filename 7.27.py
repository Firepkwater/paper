import numpy as np
from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from fealpy.pde.poisson_2d import CosCosData as PDE
import sys
import sympy
from fealpy.decorator import cartesian
from scipy.sparse.linalg import spsolve
from fealpy.boundarycondition import DirichletBC
from pde_2d import MyPde
from fealpy.tools.show import showmultirate, show_error_table
n = 10
p = 1
mf = MeshFactory()
mesh = mf.boxmesh2d([0,1,0,1], nx=n, ny=n,meshtype='tri')
pde = PDE()

space = LagrangeFiniteElementSpace(mesh, p=p)
gdof = space.number_of_global_dofs()
cell2dof = space.cell_to_dof()


cellmeasure = mesh.entity_measure('cell')#计算每个单元的面积
qf = mesh.integrator(1, 'cell')#积分点
bcs, ws = qf.get_quadrature_points_and_weights()
gphi = space.grad_basis(bcs)#(1,2,3,2)一个积分点，2个单元，3个基函数，每个基函数梯度两个分量

a = np.array([(10.0,-1.0),(-1.0,2.0)],dtype=np.float64)
dgphi = np.einsum('mn,ijkn->ijkm',a,gphi)#(2,3,2)
A = np.einsum('i, ijkl, ijml, j->jkm', ws,dgphi, gphi, cellmeasure)#(2,3,3)
I = np.broadcast_to(cell2dof[:,:,None], shape=A.shape)
J = np.broadcast_to(cell2dof[:,None,:],shape=A.shape)
A = csr_matrix((A.flat,(I.flat,J.flat)),shape = (gdof,gdof))#(gdof,gdof)刚度矩阵
# A = space.stiff_matrix(c=a)
# print(A.toarray())
#  #质量矩阵组装
# phi = space.basis(bcs)#(1,1,3)一个积分点，每个单元上的重心坐标
@cartesian
def r(p):
    x = p[...,0]
    y = p[...,1]
    return 1+x**2+y**2
M = space.mass_matrix(c=r)
# # M = np.einsum('i,ijk,ijm,j->jkm',ws,phi,phi,cellmeasure)
# # M = csr_matrix((M.flat,(I.flat, J.flat)),shape=(gdof,gdof))
#  # print(M.toarray())
b = np.array([1.0,1.0], dtype=np.float64)
# B = np.einsum('i,q,ijkq,ijm,j->jkm',ws,b,gphi,phi,cellmeasure)
# B = csr_matrix((B.flat,(I.flat, J.flat)),shape=(gdof,gdof))
# #print(B.toarray())
B = space.convection_matrix(c=b)
#  #组装载荷向量
@cartesian
def f(p):
    x = p[...,0]
    y = p[...,1]
    pi = np.pi
    return 2*pi**2*np.sin(pi*x)*np.sin(pi*y)+12*pi**2*np.cos(pi*x)*np.cos(pi*y)+(-pi)*np.sin(pi*x)*np.cos(pi*y)-pi*np.cos(pi*x)*np.sin(pi*y)+(1+x**2+y**2)
# qf = mesh.integrator(3,'cell')#拿到积分公式
# bcs, ws = qf.get_quadrature_points_and_weights()#拿到积分点
# #print(bcs,ws)有6个积分权重
# phi = space.basis(bcs)#(6,1,3)虽然有2个单元，但是2个单元上的基函数是一样的，所以就存储一个
# ps = mesh.bc_to_point(bcs)#(6,2,2)每个积分点上有2个分量的值
# # print(ps)
# # print(ps[...,0])#x的分量
# # print(ps[...,1])#y的分量
# val = f(ps)#(6,2)
# bb = np.einsum('i,ij,ijk,j->jk',ws,val,phi,cellmeasure)
# F = np.zeros(gdof,dtype=np.float64)
# np.add.at(F,cell2dof,bb)
# uh = space.function()
F = space.source_vector(f)

# bc = DirichletBC(space, pde.dirichlet)
A += B + M
# A, F = bc.apply(A, F, uh)
# uh[:] = spsolve(A, F).reshape(-1)

print(A.toarray())