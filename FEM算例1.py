import numpy as np
import math
import matplotlib.pyplot as plt
'''
- u^{''} + u = f
边界条件：u'(0) = u'(1) = 0
\f = (1 + {\pi ^2})\cos \pi x\
解析解为：u = \cos \pi x
'''
N = 20
x = np.arange(0,1.05,1/N)
n = len(x) - 1
alpha = 0.8
rho = np.zeros((n+1,1))
for i in range(0,n):
    h = x[i + 1] - x[i]
    def f(x):
        pi = np.pi
        return (1 + pi**2)*np.cos(pi*x)
    a = f(x[i])
    b = f(x[i+1])
    t = (a**2 + b**2)*h/2
    rho[i] = h**2*t
for i in range (0,n+1):
    c = max(rho)
    if rho[i] > alpha*c:
       x = np.insert(x,0,(x[i+1]+x[i]) / 2)

x_1 = list(set(x))
x_1 = np.sort(x_1)
n1 = len(x_1) - 1
A = np.zeros((n1+1,n1+1))
def Stiffmat1D(x):
    for i in range (0,n1):
        if x[i] != x[i+1]:
            h = x[i+1] - x[i]
        A[i][i] = A[i][i] + 1/h
        A[i][i+1] = A[i][i+1] - 1/h
        A[i+1][i] = A[i+1][i] - 1/h
        A[i+1][i+1] = A[i+1][i+1] + 1/h
    return A
M = np.zeros((n1+1,n1+1))
def MassMat1D(x):
    for i in range(0,n1):
        h = x[i+1]-x[i]
        M[i][i] = M[i][i] + h/3
        M[i][i+1] = M[i][i+1] + h / 6
        M[i+1][i] = M[i+1][i] + h / 6
        M[i+1][i+1] = M[i+1][i+1] + h / 3
    return M
b = np.zeros((n1+1,1))
def LoadVec1D(x,f):
    #n = len(x) - 1
    def f(x):
        pi = np.pi
        return (1 + pi ** 2) * np.cos(pi * x)
    for i in range(0,n1):
        h = x[i+1] - x[i]
        b[i] = b[i] + f(x[i])*h/2
        b[i+1] = b[i+1] + f(x[i+1])*h/2
    return b


#加密的图像

A1 = Stiffmat1D(x_1)
b1 = LoadVec1D(x_1,f)
M1 = MassMat1D(x_1)
B = A1 + M1
U = np.linalg.solve(B,b1)

fig = plt.figure
l1, = plt.plot(x_1,U,color = 'blue', linewidth='1.0', linestyle='-')
y1 = np.cos(np.pi*x)
l2, = plt.plot(x,y1,color = 'red', linewidth='2.0', linestyle='--')
plt.legend(handles = [l1,l2],labels = ['FEM solution','analytic solution'],loc='best')
plt.show()
plt.close()
print('加密后的格点:',x_1)
#compute error
e = (y1 - U)**2/2
print('误差值:',e)
fig = plt.figure
x1 = np.linspace(0,1,100)
l=0.01
y1 = x1*(np.exp(-(x1-1/3)**2/l)-np.exp(-4/9*l))#解析解
plt.plot(x1,y1,color = 'blue', linewidth='1.0', linestyle='--')
plt.show()
plt.close()