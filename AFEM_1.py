import numpy as np
import math
import matplotlib.pyplot as plt
'''
- u^{''}  = f
边界条件：u(0) = u(1) = 0
\f = -2\
解析解为：u =x^2-x
'''
N = 24
x = np.arange(0.00,1.01,1/N)
n = len(x)-1
alpha = 0.9
rho = np.zeros((n+1,1))
for i in range(0,n):
    h = x[i + 1] - x[i]
    t = 4*h
    rho[i] = h*math.sqrt(t)
for i in range (0,n+1):
    c = max(rho)
    if rho[i] > alpha*c:
       x = np.insert(x,0,(x[i+1]+x[i]) / 2)

x_1 = list(set(x))
x_1 = np.sort(x_1)
n1 = len(x_1) - 2
# print(n1)
A = np.zeros((n1,n1))
for i in range (0,n1-1):
    # print(i)
    h = x_1[i+2] - x_1[i+1]
    A[i][i] = A[i][i] + 1/h
    A[i][i+1] = A[i][i+1] - 1/h
    A[i+1][i] = A[i+1][i] - 1/h
    A[i+1][i+1] = A[i+1][i+1] + 1/h

A[0][0] = A[0][0] + 1/(x_1[1]-x_1[0])
A[n1-1][n1-1] = A[n1-1][n1-1] + 1/(x_1[n1+1]-x_1[n1])
# print(A)
b = np.zeros((n1,1))

for i in range(0,n1-1):
    h = x_1[i+2] - x_1[i+1]
    b[i] = b[i] + (-2)*h/2
    b[i+1] = b[i+1] + (-2)*h/2
b[0] = b[0] + (x_1[1]-x_1[0])/2
b[n1-1] = b[n1-1] + (x_1[n1+1]-x_1[n1])/2


U = np.linalg.solve(A,b)

# plt.xlim(0.1,0.9)
# plt.ylim(-1,1)
fig = plt.figure
x2 = x_1[1:-1]

l1, = plt.plot(x2,U,color = 'blue', linewidth='1.0', linestyle='-')
y1 = x**2-x
l2, = plt.plot(x,y1,color = 'red', linewidth='2.0', linestyle='--')
# plt.ylim(-1,1)
plt.legend(handles = [l1,l2],labels = ['FEM solution','analytic solution'],loc='best')
# plt.plot(x2, U, color='blue', linewidth='2.0', linestyle='--')
plt.show()
plt.close()
e = np.abs(y1-U)
# error = x_norm=np.linalg.norm(e, ord=1, axis=None, keepdims=False)

print(e)


