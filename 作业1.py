import numpy as np
from fealpy.mesh import MeshFactory
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
mf = MeshFactory()
mesh = mf.one_tetrahedron_mesh(meshtype='iso')

node = np.array([[0, 0, 0],[0, 1, 0],[1, 0, 0],[0, 0, 1]],dtype=np.float)
cell = np.array([0, 1, 2, 3],dtype=np.float)
#计算6倍四面体体积
v0 = node[1,:]-node[0,:]
v1 = node[2,:]-node[0,:]
v2 = node[3,:]-node[0,:]
nv = np.cross(v0,-v1)
V = np.dot(nv, v2)#四面体的6倍体积

Dlambda = np.zeros((4,3), dtype=np.float)
W = np.array([[-1,0,0], [0,1,0], [0,0,-1]], dtype=np.float)
node1 = np.array([[1,1,1],node[0:3,1],node[0:3,2]])
node2 = np.array([[1,1,1],node[0:3,0],node[0:3,2]])
node3 = np.array([[1,1,1],node[0:3,0],node[0:3,1]])
a1 = np.linalg.det(node1)
a2 = np.linalg.det(node2)
a3 = np.linalg.det(node3)
Dlambda[0,:] = [a1,a2,a3]@W/V
node4 = np.array([[1,1,1],node[1:4,1],node[1:4,2]])
node5 = np.array([[1,1,1],node[1:4,0],node[1:4,2]])
node6 = np.array([[1,1,1],node[1:4,0],node[1:4,1]])
a4 = np.linalg.det(node4)
a5 = np.linalg.det(node5)
a6 = np.linalg.det(node6)
Dlambda[1,:] = [a4,a5,a6]@W/V
node7 = np.array([[1,1,1],node[[0,1,3],1],node[[0,1,3],2]])
node8 = np.array([[1,1,1],node[[0,1,3],0],node[[0,1,3],2]])
node9 = np.array([[1,1,1],node[[0,1,3],0],node[[0,1,3],1]])
a7 = np.linalg.det(node7)
a8 = np.linalg.det(node8)
a9 = np.linalg.det(node9)
Dlambda[2,:] = [a7,a8,a9]@W/V
node10 = np.array([[1,1,1],node[[0,2,3],1],node[[0,2,3],2]])
node11 = np.array([[1,1,1],node[[0,2,3],0],node[[0,2,3],2]])
node12 = np.array([[1,1,1],node[[0,2,3],0],node[[0,2,3],1]])
a10 = np.linalg.det(node10)
a11 = np.linalg.det(node11)
a12 = np.linalg.det(node12)
Dlambda[3,:] = [a10,a11,a12]@W/V
print(Dlambda)
