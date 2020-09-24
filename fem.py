import numpy as np
import math
import matplotlib.pyplot as plt
N = 4
x_1 = np.arange(0.00,1.01,1/N)
n1 = len(x_1)-2
alpha = 0.9
k=0.01
def f(x):
    return (4 * (x - 1 / 3) / k + (2*x)/k - 4*x*(x-1/3)**2/(k**2)) * np.exp(-(x - 1 / 3) ** 2 / k)

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
    b[i] = b[i] + f(x_1[i+1])*h/2
    b[i+1] = b[i+1] + f(x_1[i+2])*h/2
b[0] = b[0] + f(x_1[1])*(x_1[1]-x_1[0])/2
b[n1-1] = b[n1-1] + f(x_1[n1])*(x_1[n1+1]-x_1[n1])/2


U = np.linalg.solve(A,b)
fig = plt.figure
x2 = x_1[1:-1]

l1, = plt.plot(x2,U,color = 'blue', linewidth='1.0', linestyle='-')
x1 = np.linspace(0,1,100)
y1 = x1*(np.exp(-((x1-1/3)**2)/k)-np.exp(-4/(9*k)))#解析解
l2,=plt.plot(x1,y1,color = 'red', linewidth='2.0', linestyle='--')

plt.legend(handles = [l1,l2],labels = ['AFEM solution','analytic solution'],loc='best')
plt.show()
plt.close()
e = np.abs(y1-U)
# error = x_norm=np.linalg.norm(e, ord=1, axis=None, keepdims=False)

print(e)








