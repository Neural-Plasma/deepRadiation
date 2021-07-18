import numpy as np

#====================== init_func ======================

def gaussian(x, mu, sig, shift):
    return shift + np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def initVar(x_left, x_right, t_end, D0, v0, dx, dt, Nx, Nt, ic):
    xx = np.linspace(x_left, x_right, Nx, dtype=np.float64)
    tt = np.linspace(0, t_end, Nt, dtype=np.float64)
    vv = np.linspace(v0, 10*v0, 10, dtype=np.float64)
    DD = np.linspace(D0, 10*D0, 10, dtype=np.float64)
    u1 = np.zeros((len(DD), len(vv), Nt, Nx), dtype=np.longdouble)
    u2 = np.zeros((len(DD), len(vv), Nt, Nx), dtype=np.longdouble)
    cfl = DD*dt/dx**2
    print(cfl)


    [ic_mean, ic_std1, ic_std2, ic_shift] = ic

    # ICs
    u1[:, :, 0, :] = gaussian(xx, ic_mean, ic_std1, ic_shift)
    u2[:, :, 0, :] = gaussian(xx, ic_mean, ic_std2, ic_shift)

    return DD,vv,tt,xx,u1,u2
