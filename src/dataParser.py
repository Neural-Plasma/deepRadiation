#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from os.path import join as pjoin
import os.path
from tqdm import tqdm
import sys
import re

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
    print(perm,sFreq)
else:
    print('Training data not found. Processing data ... ')
    for f in range(len(perm_var)):
        print('Processing',perm_var[f])
        path = pjoin(prefix_path,runName,perm_var[f],'data')
        # os.system('ls '+path)
        PyAvg = np.zeros((Nx+1, Ny+1))
        for i in tqdm(range(1999500, 1999500+Nt)):
            # print('poynting_%08d'%i+'.dat')
            _, _, x, y, Pmag = np.loadtxt(path+'/poynting_%08d'%i+'.dat', unpack=True)

            x   = np.reshape(x, (Nx+1, Ny+1))
            y   = np.reshape(y, (Nx+1, Ny+1))
            # Pmag  = np.sqrt(Px*Px + Py*Py)
            Pmag  = np.reshape(Pmag, (Nx+1, Ny+1))
            PyAvg += Pmag

        PyAvg /= Nt
        PyRad.append(PyAvg/const.c)
        # PyRms.append(np.sqrt(np.mean(np.square(PyAvg)))/const.c)

    PyRad = np.array(PyRad)

    print('Saving training data...')
    np.savez_compressed(pjoin(savedir,runName+'_data.npz'),x=x,y=y,PyRad=PyRad,perm=perm,sFreq=sFreq[int(n[0])-1])
    print('Training data saved to %s'%(pjoin(savedir,runName+'_data.npz')))

print(PyRad.shape)
# exit()



# plt.plot(sFreq,Psum, 'o-')
# plt.xlabel('Length')
# plt.ylabel('Sum Values')
# plt.yscale('log')
#
# plt.show()
