import numpy as np
from tqdm import tqdm
from os.path import join as pjoin

#==================== Advection Solver ==============================
# Euler Stepper
def adv_diff_euler_step(u1l, u1c, u1r, u2l, u2c, u2r, D, v, dx, dt):
    diff_ul = np.zeros(1,dtype=np.longdouble)
    diff_uc = np.zeros(1,dtype=np.longdouble)
    diff_ur = np.zeros(1,dtype=np.longdouble)
    diff_ul = u1l - u2l
    diff_uc = u1c - u2c
    diff_ur = u1r - u2r
    return u1c + D * dt/dx**2 * (diff_ul - 2 * diff_uc + diff_ur) + v * dt/(2*dx) * (u1r - u1l)


def solver(DD,vv,tt,xx,u1,u2,dx, dt, savedir):
    print('Training data does not exist. Generating training data...')
    # Solve
    inputs = []
    outputs = []
    for Di, D in enumerate(tqdm(DD[:-1], desc=" Diffusion", position = 2)):
        for vi, v in enumerate(tqdm(vv[:-1], desc=" Velocity", position = 1, leave=False)):
            for ti, t in enumerate(tqdm(tt[:-1], desc=" Time", position = 0, leave=False)):
                for xi, x in enumerate(xx[1:-1]):
                    u1[Di, vi, ti+1, xi] = adv_diff_euler_step(u1[Di, vi, ti, xi-1], u1[Di, vi, ti, xi], u1[Di, vi, ti, xi+1], u2[Di, vi, ti, xi-1], u2[Di, vi, ti, xi], u2[Di, vi, ti, xi+1], D, v, dx, dt)
                    u2[Di, vi, ti+1, xi] = adv_diff_euler_step(u2[Di, vi, ti, xi-1], u2[Di, vi, ti, xi], u2[Di, vi, ti, xi+1], u1[Di, vi, ti, xi-1], u1[Di, vi, ti, xi], u1[Di, vi, ti, xi+1], D, v, dx, dt)

                    # zero flux BC
                    u1[Di, vi, ti+1, 0]  = u1[Di, vi, ti+1, 1]
                    u1[Di, vi, ti+1, -1] = u1[Di, vi, ti+1, -2]
                    u2[Di, vi, ti+1, 0]  = u2[Di, vi, ti+1, 1]
                    u2[Di, vi, ti+1, -1] = u2[Di, vi, ti+1, -2]

                    # Save data
                    inputs.append([D, v, t, x])
                    outputs.append([u1[Di, vi, ti+1, xi],u2[Di, vi, ti+1, xi]])
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    print('Saving training data...')
    np.savez_compressed(pjoin(savedir,'training_data.npz'),inputs=inputs,outputs=outputs,u1=u1,u2=u2)
    print('Training data saved to %s'%(pjoin(savedir,'training_data.npz')))
