#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from os.path import join as pjoin
import os.path
from tqdm import tqdm
import sys

runName = str(sys.argv[1])

prefix_path = pjoin('..','..','training_data')

freq_var = ['freq_1e9','freq_1-5e9','freq_1-7e9','freq_2e9','freq_2-2e9'
,'freq_2-45e9','freq_2-7e9','freq_3e9','freq_3-2e9','freq_3-45e9']

sFreq = np.array([1e9,1.5e9,1.7e9,2e9,2.2e9,2.45e9,2.7e9,3e9,3.2e9,3.45e9])

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
    sFreq=data['sFreq']
else:
    print('Training data not found. Processing data ... ')
    for f in range(len(freq_var)):
        print('Processing',freq_var[f])
        path = pjoin(prefix_path,runName,freq_var[f],'poynting')
        PyAvg = np.zeros((Nx+1, Ny+1))
        for i in tqdm(range(1, Nt)):
            x, y, _, Py = np.loadtxt(path+'/poynting_%d'%i+'.txt', unpack=True)

            x   = np.reshape(x, (Nx+1, Ny+1))
            y   = np.reshape(y, (Nx+1, Ny+1))

            Py  = np.reshape(Py, (Nx+1, Ny+1))
            PyAvg += Py

        PyAvg /= Nt
        PyRad.append(PyAvg/const.c)
        # PyRms.append(np.sqrt(np.mean(np.square(PyAvg)))/const.c)

    PyRad = np.array(PyRad)

    print('Saving training data...')
    np.savez_compressed(pjoin(savedir,runName+'_data.npz'),x=x,y=y,PyRad=PyRad,sFreq=sFreq)
    print('Training data saved to %s'%(pjoin(savedir,runName+'_data.npz')))

print(PyRad.shape)
exit()



plt.plot(sFreq,Psum, 'o-')
plt.xlabel('Length')
plt.ylabel('Sum Values')
plt.yscale('log')

plt.show()
