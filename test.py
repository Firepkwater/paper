import numpy as np
from fealpy.mesh import IntervalMesh
import matplotlib.pyplot as plt
node = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)
cell = np.array([[0, 1], [1, 2]], dtype=np.int_)
mesh = IntervalMesh(node, cell)
NC = mesh.number_of_cells()
node = mesh.entity('node')

v0= node[cell[: , 1],:] - node[cell[:, 0],:]
Dlambda = np.zeros((NC,2,1), dtype=np.float64)
Dlambda[:,0,:] = (-1) / v0
Dlambda[:,1,:] = 1 / v0
print(Dlambda)
