import numpy as np
import math
import matplotlib.pyplot as plt
x = np.arange(2,8.1,0.1)
n = len(x) - 1
alpha = 0.8
rho = np.zeros((n+1,1))
for i in range(0,n):
    h = x[i + 1] - x[i]
    def f(x):
        return 0.03 * (x - 6) ** 4
    a = f(x[i])
    b = f(x[i+1])
    t = (a**2 + b**2)*h/2
    rho[i] = h**2*t
for i in range (0,n+1):
    c = max(rho)
    if rho[i] > alpha*c:
       x = np.insert(x,0,(x[i+1]+x[i]) /2)
x_1 = np.sort(x)

n1 = len(x_1) - 1
A = np.zeros((n1+1,n1+1))
def Stiffmat1D(x,kappa):
    def a(x):
        return 0.1 * (5 - 0.6 * x)
    for i in range (0,n1):
        if x[i] != x[i+1]:
            h = x[i+1] - x[i]
        xmid = (x[i+1]+x[i])/2
        amid = a(xmid)
        A[i][i] = A[i][i] + amid/h
        A[i][i+1] = A[i][i+1] - amid/h
        A[i+1][i] = A[i+1][i] - amid/h
        A[i+1][i+1] = A[i+1][i+1] + amid/h
    A[0][0] = A[0][0] + kappa[0]
    A[n1][n1] = A[n1][n1] + kappa[1]
    return A
b = np.zeros((n1+1,1))
def LoadVec1D(x,kappa,g):
    #n = len(x) - 1
    def f(x):
        return 0.03 * (x - 6) ** 4
    for i in range(0,n1):
        h = x[i+1] - x[i]
        b[i] = b[i] + f(x[i])*h/2
        b[i+1] = b[i+1] + f(x[i+1])*h/2
    b[0] = b[0] + kappa[0]*g[0]
    b[n1] = b[n1] + kappa[1]*g[1]
    return b


#加密的图像
kappa = np.array([10**6,0])
g = np.array([-1,0])

A1 = Stiffmat1D(x_1,kappa)
b1 = LoadVec1D(x_1,kappa,g)
U = np.linalg.solve(A1,b1)

fig = plt.figure
plt.plot(x_1,U,color = 'blue', linewidth='1.0', linestyle='-')
plt.show()
plt.close()







