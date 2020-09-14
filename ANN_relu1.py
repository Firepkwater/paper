import numpy as np
import math
import matplotlib.pyplot as plt
M=5 #max iteration
eta=0.1 #step length

for N in range(8,25,8):
    t0 = np.arange(0.0,1.001,1/N)
    d = 1.0 / (t0[2:]-t0[1:-1]) + 1.0 / (t0[1:-1]-t0[0:-2])
    r = 1.0 / (t0[2:-1]-t0[1:-2])
    A = np.diag(d) -np.diag(r,k=1)-np.diag(r,k=-1)
    b = -(t0[2:] - t0[:-2])
    c = b.reshape(b.shape[0], 1)
    ut = np.linalg.solve(A, c)
    uf = np.insert(np.append(ut,[0]), 0, 0)
    uf_T = uf.reshape(uf.shape[0], 1)

    theta = np.zeros(N+1, dtype=np.float)
    for i in range(1, N):
        theta[i] = (uf[i]-uf[i-1])/(t0[i]-t0[i-1])

    k=0
    while k < M:
        #update interval t0
        g = 0.5*((theta[1:N-1])**2-(theta[2:N])**2) - 2*(theta[2:N]-theta[1:N-1])*(1-t0[1:N-1])
        t0[1:N-1] = t0[1:N-1] - eta*g

        #update theta by finite element method
        d = 1.0 / (t0[2:] - t0[1:-1]) + 1.0 / (t0[1:-1] - t0[0:-2])
        r = 1.0 / (t0[2:-1] - t0[1:-2])
        A = np.diag(d) - np.diag(r, k=1) - np.diag(r, k=-1)
        b = -(t0[2:] - t0[:-2])
        c = b.reshape(b.shape[0], 1)
        ut = np.linalg.solve(A, c)
        uf = np.insert(np.append(ut, [0]), 0, 0)
        uf_T = uf.reshape(uf.shape[0], 1)
        #derive initial parameter theta from uf_T
        theta = np.zeros(N + 1, dtype=np.float)
        for i in range(1, N):
            theta[i] = (uf[i] - uf[i - 1]) / (t0[i] - t0[i - 1])
        k += 1

    #compute error
    er = 0
    for k in range(1,N):
        er = er+(4/3)*((t0[k]-(1+theta[k])*0.5)**3-(t0[k-1]-(1+theta[k])*0.5)**3)
    H1seminorm = math.sqrt(er)
    print('H1seminorm{}:'.format(N/8),H1seminorm)

    fig = plt.figure
    x1 = np.linspace(0,1,100)
    y1 = x1**2 - x1#解析解
    plt.plot(x1,y1,color = 'blue', linewidth='1.0', linestyle='--')

    for i in range(1,N):

        x2 = np.linspace(t0[i],t0[i-1],100)
        lx2 = len(x2)
        u2 = np.zeros(lx2, dtype=np.float)
        for j in range(2,i+1):
            u2 = u2 + (theta[j-1]-theta[j-2])*(x2-t0[j-2])
        plt.plot(x2,u2, color='red', linewidth='2.0', linestyle='-')
    plt.show()
    plt.close()






