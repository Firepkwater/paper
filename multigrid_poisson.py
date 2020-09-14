import math
import numpy as np
import matplotlib.pyplot as plt


def coarsen2(grid0):
    # dstep = int(2**(level-1))
    n_grid = int((grid0.size - 1) / 2 + 1)  # int((grid0.size-1)/dstep + 1)
    grid1 = np.zeros(n_grid, dtype=float)

    for i in range(1, grid1.size - 1):
        i2 = 2 * i
        grid1[i] = 0.5 * grid0[i2] + 0.25 * grid0[i2 - 1] + 0.25 * grid0[i2 + 1]

    grid1[0] = grid0[0]
    grid1[-1] = grid0[-1]
    return grid1


def h_level(level):
    h0 = h
    dstep = int(2 ** (level - 1))
    h2_grid = int((h0.size) / dstep)
    h2 = np.zeros(h2_grid, dtype=float)
    for i in range(0, h2.size):
        j0 = dstep * i
        h2_step = 0.
        for j in range(dstep):
            h2_step = h2_step + h0[j0 + j]
        h2[i] = h2_step
    return h2


def Relax2(b, phi, h, level_total, leveli):
    om = 1.85
    ite = 4 ** leveli  # the iteration increase with the level to get higher precision
    j = 0
    print("relax method")
    n2 = int(b.size / 2)  # choose the middle point to compare the err in each iteration
    while (j < ite):  # 控制迭代次数
        i = 1  # when the boundary is known,  set i  =  1
        while (i < phi.size - 1):
            phi_1 = np.copy(phi[i])
            phi[i] = om * 0.5 * (phi[i + 1] + phi[i - 1] - (h[i] * h[i]) * b[i]) + (
                        1. - om) * phi_1  # Inhomogeneous grid
            i = i + 1
        j = j + 1
    return phi


def residual(f, v, h):
    r = np.zeros(f.size, dtype=float)
    for i in range(1, v.size - 1):
        r[i] = f[i] - 2 * (v[i + 1] - 2. * v[i] + v[i - 1]) / (h[i] * h[i] + h[i - 1] * h[i - 1])
    return r


def fine(grid_orig, level):  # h is the step of the fine grid
    h = h_level(level)  # 要被处理的网格所在的level
    h2 = h_level(level - 1)  # 需要生成的网格所在的level
    n_fine = (grid_orig.size - 1) * 2 + 1
    grid_fine = np.zeros(n_fine, dtype=float)

    for i in range(0, grid_orig.size - 1):
        i2 = int(i * 2)
        grid_fine[i2] = grid_orig[i]
        grid_fine[i2 + 1] = grid_orig[i] * (h2[i2 + 1] / h[i]) + grid_orig[i + 1] * (h2[i2] / h[i])
    grid_fine[-1] = grid_orig[-1]
    return grid_fine


def MG(phi, x, b, h):
    level_total = int(math.log2(phi.size - 1)) - 5  # at least 2**5 grids to capture small fluctuation
    print(level_total)
    h0 = h_level(1)
    vh = Relax2(b, phi, h, level_total, 1)
    rh = residual(b, vh, h0)
    eh = np.zeros(b.size, dtype=float)  # 初始化

    for i in range(1, level_total):  # coarse and iterate
        print(i)
        h1 = h_level(i + 1)
        r2h = coarsen2(rh)
        e2h0 = coarsen2(eh)
        e2h = Relax2(r2h, e2h0, h1, level_total, i)
        v2h = coarsen2(vh) + e2h
        b2h = coarsen2(b)
        r2h = residual(b2h, v2h, h1)
        rh = r2h
        eh = e2h
        vh = v2h
        b = b2h

    for i in range(1, level_total):  # fine and itearate
        print(i)
        leveli = level_total - i + 1
        h1 = h_level(leveli)
        h2 = h_level(leveli - 1)
        r2h = fine(rh, leveli)
        e2h0 = fine(eh, leveli)
        e2h = Relax2(r2h, e2h0, h2, level_total, leveli)
        b2h = fine(b, leveli)
        v2h = fine(vh, leveli) + e2h
        r2h = residual(b2h, v2h, h2)
        rh = r2h
        eh = e2h
        vh = v2h
        b = b2h

    return vh


n_grid = 1025  # at least 512 grids to reach enough depth
xs = 0.0
xe = math.pi
h = (xe - xs) / (n_grid - 1) * np.ones(n_grid - 1, dtype=float)

x = np.zeros(n_grid, dtype=float)
x[0] = xs
x[-1] = xe
for i in range(1, n_grid - 1):
    x[i] = x[i - 1] + h[i - 1]

phi = np.ones(x.size, dtype=float) * 0.1
phi[0] = 0.
phi[-1] = 0.

b = -16 * np.sin(4 * x)
result2 = np.sin(4 * x)  # precise solution
result = MG(phi, x, b, h)

plt.figure(1)
plt.plot(x, result, label='numerical')
plt.plot(x, result2, label='real')
plt.legend()
plt.show()
plt.close()





