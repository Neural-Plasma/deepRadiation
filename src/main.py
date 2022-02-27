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
from dnnRadiation import dnnSim
from sklearn.metrics import mean_squared_error


class radiation_profile:
  def __init__(self, xgrid, ygrid,sFreq,perm,PyRad):
    self.x = xgrid[:-1,:-1]
    self.y = ygrid[:-1,:-1]
    self.sFreq = sFreq
    self.perm = perm
    self.PyRad = PyRad
    # self.PyRad = np.sqrt(np.mean(np.square(PyRad[:,10])))
#Initialize timer
task = Timer()
def main(argv):
    parser = argparse.ArgumentParser(description='Deep Neural Network for Plasma Radiation (deepRadiation)')
    parser.add_argument('-l','--load', action='store_true', help='Add this if you want to load the model data')
    parser.add_argument('-hp','--histp', action='store_true', help='Add this to plot history')
    parser.add_argument('-tp','--testp', action='store_true', help='Add this to plot test values')
    args        = parser.parse_args()

    loadModel = args.load
    histplot = args.histp
    testplot = args.testp

    savedir = 'data'

    datadir = pjoin('training_data_processed')
    runName = ['run1','run2','run3','run4','run5','run6','run7','run8','run9','run10','run11','run12','run13','run14','run15','run16','run17','run18','run19','run20'] #np.arange(1,11)
    rad_p = []
    task.start('Loading Training data')
    for i in range(len(runName)):
        if os.path.exists(pjoin(datadir,runName[i]+'_syn_data.npz')):
            print('Training data found. Loading data...')
            data = np.load(pjoin(datadir,runName[i]+'_syn_data.npz'))
            xgrid=data['x']
            ygrid=data['y']
            PyRad=data['PyRad']
            perm=data['perm']
            sFreq=data['sFreq']
            for j in range(len(perm)):
                rad_p.append(radiation_profile(xgrid,ygrid,sFreq,perm[j],PyRad[j]))
        else:
            print('Training data not found. Run "dataParser.py"')
            exit()
    inputs = []
    outputs = []
    xall = []
    yall = []
    PyRadall = []
    sFreqall = []
    lengthall = []
    # print(len(rad_p))
    # exit()
    for obj in rad_p:
        inputs.append([obj.sFreq,obj.perm])
        outputs.append([obj.PyRad])

    inputs  = np.array(inputs)
    outputs = np.array(outputs)
    # print(np.max(outputs))
    print(inputs.shape,outputs.shape)
    # print([inputs[0,0],inputs[0,1],outputs[0]])
    sFreqMax = max(inputs[:,0])
    permMax = max(inputs[:,1])
    radMax = np.max(abs(outputs))
    print(sFreqMax,permMax,radMax)
    inputs[:,0] /= sFreqMax
    inputs[:,1] /= permMax
    outputs /= radMax
    # plt.contourf(outputs[:2048].reshape(64,32))
    # plt.show()
    # exit()
    task.stop()


    task.start('Deep Neural Network for Plasma')

    deep_approx = dnnSim(inputs,outputs,loadModel,histplot,savedir)

    task.stop()

    # exit()

    task.start('Test Model')

    sFreq_chk = 2.45e9/sFreqMax
    perm_chk = 0.1/permMax

    len_dataset = 1000
    s1 = 500
    pRad_approx = np.zeros(len_dataset)
    for i in range(len_dataset):
        data_in = []
        data_in.append([inputs[i+s1,0],inputs[i+s1,1]])
        data_in = np.array(data_in)
        pRad_approx[i] = deep_approx.predict(data_in)

    no_data_arr = range(len_dataset)
    pRad_true = outputs.reshape(-1)[s1:(s1+len_dataset)]
    pRad_predict = pRad_approx
    MSE = mean_squared_error(pRad_true,pRad_predict)
    two_sigma = 2 * np.sqrt(MSE)  # 2*sigma ~ 95% confidence region
    if testplot:
        from diagn import comparison_plot as comp_p
        comp_p(no_data_arr,pRad_true,pRad_approx,two_sigma,savedir)
    task.stop()









if __name__== "__main__":
	start = time.time()
	main(sys.argv[1:])
	end = time.time()
	print("Elapsed (after compilation) = %s"%(end - start)+" seconds")
