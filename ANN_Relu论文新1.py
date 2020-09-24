import numpy as np
import math
import matplotlib.pyplot as plt
M=5 #max iteration
eta=0.001 #step length
l = 0.01
def f(x):
    return ((4 * (x - 1 / 3))/ l + (2*x)/l - (4*x*(x-1/3)**2)/(l**2)) * np.exp((-(x - 1 / 3) ** 2) / l)

for N in range(23,93,23):
    t0 = np.arange(0.0,1.001,1/N)
    d = 1.0 / (t0[2:]-t0[1:-1]) + 1.0 / (t0[1:-1]-t0[0:-2])
    r = 1.0 / (t0[2:-1]-t0[1:-2])
    A = np.diag(d) -np.diag(r,k=1)-np.diag(r,k=-1)
    b = f(t0[1:-1])*(t0[2:] - t0[:-2])*0.5
    c = b.reshape(b.shape[0], 1)
    ut = np.linalg.solve(A, c)
    uf = np.insert(np.append(ut,[0]), 0, 0)
    uf_T = uf.reshape(uf.shape[0], 1)

    theta = np.zeros(N+1, dtype=np.float)
    for i in range(1, N+1):
        theta[i] = (uf[i]-uf[i-1])/(t0[i]-t0[i-1])
    k=0
    while k < M:
        #update interval t0
        # g = 0.5 * ((theta[1:N-1]) ** 2 - (theta[2:N]) ** 2) - ((theta[2:N]-theta[1:N-1])/6)*((-2*(1-t0[1:N-1]))*(f(1)+\
        #     2*f(0.5*(1+t0[1:N-1])))+(1-t0[1:N-1])**2*gradientf(0.5*(1+t0[1:N-1])))
        g = 0.5 * ((theta[1:N - 1]) ** 2 - (theta[2:N]) ** 2) + f(1) * (theta[2:N] - theta[1:N - 1]) * (1 - t0[1:N - 1])
        t0[1:N-1] = t0[1:N-1] - eta*g
        # print(t0)
        #update theta by finite element method
        d = 1.0 / (t0[2:] - t0[1:-1]) + 1.0 / (t0[1:-1] - t0[0:-2])
        r = 1.0 / (t0[2:-1] - t0[1:-2])
        A = np.diag(d) - np.diag(r, k=1) - np.diag(r, k=-1)
        b = 0.5*f(t0[1:-1])*(t0[2:] - t0[:-2])
        c = b.reshape(b.shape[0], 1)
        ut = np.linalg.solve(A, c)
        uf = np.insert(np.append(ut, [0]), 0, 0)
        uf_T = uf.reshape(uf.shape[0], 1)
        #derive initial parameter theta from uf_T
        theta = np.zeros(N + 1, dtype=np.float)
        for i in range(1, N+1):
            theta[i] = (uf[i] - uf[i - 1]) / (t0[i] - t0[i - 1])
        k += 1
 #compute error
    er = 0
    for k in range(1,N+1):
        er = er+(4/3)*((t0[k]-(1+theta[k])*0.5)**3-(t0[k-1]-(1+theta[k])*0.5)**3)
    H1seminorm = math.sqrt(abs(er))
    print('H1seminorm{}:'.format(N/23),H1seminorm)

    fig = plt.figure
    x1 = np.linspace(0,1,100)
    plt.xlim(0,1)
    plt.ylim(0,0.35)
    y1 = x1*(np.exp(-(x1-1/3)**2/l)-np.exp(-4/(9*l)))#解析解
    l1,=plt.plot(x1,y1,color = 'blue', linewidth='1.0', linestyle='--')

    for i in range(1,N+1):
        x2 = np.linspace(t0[i-1],t0[i],100)
        lx2 = len(x2)
        u2 = np.zeros(lx2, dtype=np.float)
        for j in range(2,i+2):
            u2 = u2 + (theta[j-1]-theta[j-2])*(x2-t0[j-2])
        l2,=plt.plot(x2,u2, color='red', linewidth='1.0', linestyle='-')
    plt.legend(handles=[l1, l2], labels=['analytic solution', 'DNN solution'], loc='best')
    plt.show()
    plt.close()

