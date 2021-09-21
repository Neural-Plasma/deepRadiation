#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from os.path import join as pjoin
import os.path
from tqdm import tqdm
import sys
import re
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline

runName = str(sys.argv[1])
n = re.findall(r'\d+', runName)

prefix_path = pjoin('..','..','..','perm_var')

perm_var = ['permit_5','permit_15','permit_20','permit_35','permit_40'
,'permit_45','permit_50','permit_65','permit_70','permit_100']

sFreq = np.array([1e9,1.5e9,1.7e9,2e9,2.2e9,2.45e9,2.7e9,3e9,3.2e9,3.45e9,3.7e9,4e9,4.2e9,4.45e9,4.7e9,5e9,5.2e9,5.45e9,5.7e9,6e9])
perm = np.array([5,15,20,35,40,45,50,65,70,100])

savedir = pjoin('..','training_data_processed')

Nt  = 500
# yLoc = 29
Nx = 64
Ny = 32

PyRad = []

if os.path.exists(pjoin(savedir,runName+'_data.npz')):
    print('Training data found. Loading data...')
    data = np.load(pjoin(savedir,runName+'_data.npz'))
    x=data['x']
    y=data['y']
    PyRad=data['PyRad']
    perm=data['perm']
    sFreq=data['sFreq']
    PyRad1D = np.zeros(perm.shape)
    for i in range(len(perm)):
        PyRad1D[i] = np.sqrt(np.mean(np.square(PyRad[i,:,10])))
    print(perm,sFreq,PyRad1D)

    perm_xi = np.linspace(np.min(perm),np.max(perm),num=1024,endpoint=True)

    rbfPyRad = Rbf(perm, PyRad1D)
    fitPyRad = rbfPyRad(perm_xi)

    # print(fitPyRad.reshape(-1).shape)

    # plt.plot(perm, PyRad1D, '-o')
    # plt.plot(perm_xi, fitPyRad, '-x')
    # plt.show()


    print('Saving training data...')
    np.savez_compressed(pjoin(savedir,runName+'_syn_data.npz'),x=x,y=y,PyRad=fitPyRad,perm=perm_xi,sFreq=sFreq)
    print('Training data saved to %s'%(pjoin(savedir,runName+'_syn_data.npz')))

# print(PyRad.shape)
# exit()



# plt.plot(sFreq,Psum, 'o-')
# plt.xlabel('Length')
# plt.ylabel('Sum Values')
# plt.yscale('log')
#
# plt.show()
