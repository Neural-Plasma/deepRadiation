import numpy as np
import matplotlib.pyplot as plt

from timer import Timer
import time
from tqdm import tqdm
import ini
import argparse

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# LOG code details
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

import os.path
from os.path import join as pjoin

from inputParser import inputParams
from init import initVar
from advecSolver import solver
from dnnPlasma import dnnSim


class radiation_profile:
  def __init__(self, xgrid, ygrid,sFreq,PyRad):
    self.x = xgrid
    self.y = ygrid
    self.sFreq = sFreq
    self.PyRad = PyRad
    self.length = np.max(xgrid[:,0])-np.min(xgrid[:,0])
#Initialize timer
task = Timer()
def main(argv):
    # parser = argparse.ArgumentParser(description='Deep Neural Network for Plasma Radiation (deepRadiation)')
    # parser.add_argument('-i','--input', default='input.ini', type=str, help='Input file name')
    # args        = parser.parse_args()
    # inputFile   = args.input
    #
    # x_left, x_right, t_end, D0, v0, dx, dt, Nx, Nt, nepoch, ic, runSolver, loadModel, savedir, testvals, rawplot, testplot, histplot = inputParams(inputFile)
    #
    # DD,vv,tt,xx,u1,u2 = initVar(x_left, x_right, t_end, D0, v0, dx, dt, Nx, Nt, ic)
    loadModel = False
    histplot = True
    testplot = True

    savedir = 'data'

    datadir = pjoin('..','training_data_processed')
    runName = ['run1','run2','run3','run4','run5','run6','run7','run8','run9','run10'] #np.arange(1,11)
    rad_p = []
    task.start('Loading Training data')
    for i in range(len(runName)):
        if os.path.exists(pjoin(datadir,runName[i]+'_data.npz')):
            print('Training data found. Loading data...')
            data = np.load(pjoin(datadir,runName[i]+'_data.npz'))
            xgrid=data['x']
            ygrid=data['y']
            PyRad=data['PyRad']
            sFreq=data['sFreq']
            for j in range(len(sFreq)):
                rad_p.append(radiation_profile(xgrid,ygrid,sFreq[j],PyRad[j,:,:]))
        else:
            print('Training data not found. Run "dataParser.py"')
            exit()
    # inputs = []
    # outputs = []
    xall = []
    yall = []
    PyRadall = []
    sFreqall = []
    lengthall = []
    # print(len(rad_p))
    # exit()
    for obj in rad_p:
        xall.append(obj.x.reshape(-1))
        yall.append(obj.y.reshape(-1))
        PyRadall.append(obj.PyRad.reshape(-1))
        sFreqall.append(np.vstack([obj.sFreq]*len(obj.x.reshape(-1))).reshape(-1))
        lengthall.append(np.vstack([obj.length]*len(obj.x.reshape(-1))).reshape(-1))
        # for xi, x in enumerate(obj.x[:,0]):
        #     for yi, y in enumerate(obj.y[0,:]):
        #         inputs.append([obj.sFreq,obj.length,x,y])
        #         outputs.append([obj.PyRad[xi,yi]])
    # print(np.array(xall).shape, np.array(sFreqall).shape,np.array(PyRadall).shape)
    # exit()
    xall    = np.array(xall).reshape(-1)
    yall    = np.array(yall).reshape(-1)
    sFreqall = np.array(sFreqall).reshape(-1)
    lengthall = np.array(lengthall).reshape(-1)
    PyRadall = np.array(PyRadall).reshape(-1)
    # xall    = np.array(xall)
    # yall    = np.array(yall)
    # sFreqall = np.array(sFreqall)
    # lengthall = np.array(lengthall)
    # PyRadall = np.array(PyRadall)

    inputs = np.column_stack((xall,yall,sFreqall,lengthall))
    # inputs  = np.array(inputs)
    outputs = PyRadall
    print(inputs.shape,outputs.shape)
    # exit()
    task.stop()


    # if rawplot:
    #     task.start('Plotting data from good old solver')
    #     fig0 = plt.figure(figsize=(8,6))
    #     ax0 = fig0.add_subplot(111)
    #     ax0.plot(xx, u1[-1, -1,  0, :], lw=2, label="u1,$t_0$")
    #     ax0.plot(xx, u1[-1, -1, -1, :], lw=2, label="u1,$t_{end}$")
    #     ax0.plot(xx, u2[-1, -1,  0, :], lw=2, label="u2,$t_0$")
    #     ax0.plot(xx, u2[-1, -1, -1, :], lw=2, label="u2,$t_{end}$")
    #     ax0.set_xlabel('$x$')
    #     ax0.set_ylabel('$u(x, t)$')
    #     ax0.legend()
    #     plt.show()
    #     task.stop()


    task.start('Deep Neural Network for Plasma')

    deep_approx = dnnSim(inputs,outputs,loadModel,histplot,savedir)

    task.stop()

    # exit()

    task.start('Test Model')
    # nplots = 11
    # rmin = 0
    # rmax = 1
    # idxes = np.arange(int(rmin*len(tt)), int(rmax*len(tt)), int((rmax-rmin)*len(tt)/nplots))
    # e1_mean = []
    # e2_mean = []
    # tt_mean = []
    x_chk = np.linspace(0,0.1,64)
    y_chk = np.linspace(0,0.04,32)
    x_grd_chk, y_grd_chk = np.meshgrid(x_chk,y_chk)

    sFreq_chk = 1.5e9
    slength_chk = 0.1


    x_chk = x_grd_chk.reshape(-1)
    y_chk = y_grd_chk.reshape(-1)
    sFreq_chk = np.array(np.ones(len(x_chk))*sFreq_chk)
    slength_chk = np.array(np.ones(len(x_chk))*slength_chk)

    data_in = np.column_stack((x_chk,y_chk,sFreq_chk,slength_chk))

    print(data_in.shape)


    pRad_approx = deep_approx.predict(data_in)
    print(pRad_approx)
    pRad_approx = pRad_approx.reshape(64,32)
    # exit()
    task.stop()

    if testplot:
        fig1 = plt.figure(figsize=(8,6))
        ax1 = fig1.add_subplot(111)

        ax1.contourf(pRad_approx)
        # ax1 = fig.add_subplot(222)
        # ax2 = fig.add_subplot(223)
        # ax3 = fig.add_subplot(224)
        # for idx, i in enumerate(idxes):
        #     data_in = np.array([ [tt[i], x] for x in xx])
        #     u_approx = deep_approx.predict(data_in)
        #     ax0.plot(xx, u_approx[:,0], lw=2, color=c[idx%len(c)])
        #     ax0.plot(xx, u1[i, :], lw=2, linestyle='--')
        #     ax1.plot(xx, u_approx[:,1], lw=2, color=c[idx%len(c)])
        #     ax1.plot(xx, u2[i, :], lw=2, linestyle='--')
        #     tt_mean.append(tt[i])
        #     e1_mean.append( np.mean((u_approx[:, 0] - u1[i, :])**2) )
        #     e2_mean.append( np.mean((u_approx[:, 1] - u2[i, :])**2) )
        #

        # ax2.plot(tt_mean, e1_mean, '.-', lw=2, color=c[0], markersize=10)
        # ax2.plot([(1-test_ratio)*t_end]*2, [min(e1_mean), max(e1_mean)], ':', color=c[1])
        # ax2.legend(['RMSE', 'Train/dev time horizon'])
        #
        # ax3.plot(tt_mean, e2_mean, '.-', lw=2, color=c[0], markersize=10)
        # ax3.plot([(1-test_ratio)*t_end]*2, [min(e2_mean), max(e2_mean)], ':', color=c[1])
        # ax3.legend(['RMSE', 'Train/dev time horizon'])
        #
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$y$')
        # ax1.set_xlabel('$x$')
        # ax1.set_ylabel('$u2(x, t)$')
        # # ax0.legend(['$t^*_{end}$'])
        # ax2.set_ylabel('Error1')
        # ax3.set_ylabel('Error2')
        #
        # fig.tight_layout()

        plt.show()






if __name__== "__main__":
	start = time.time()
	main(sys.argv[1:])
	end = time.time()
	print("Elapsed (after compilation) = %s"%(end - start)+" seconds")
