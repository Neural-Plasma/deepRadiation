import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

##### FIG SIZE CALC ############
figsize = np.array([80,80/1.618]) #Figure size in mm
dpi = 300                         #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

mp.rc('text', usetex=True)
mp.rc('font', family='sans-serif', size=14, serif='Computer Modern Roman')
mp.rc('axes', titlesize=12)
mp.rc('axes', labelsize=12)
mp.rc('xtick', labelsize=12)
mp.rc('ytick', labelsize=12)
mp.rc('legend', fontsize=10)

def comparison_plot(x,y1,y2,two_sigma,dir):

    fig,ax = plt.subplots(1,1,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
    ax.plot(x,y1,'k',lw=1.5, label = "Simulation")
    ax.plot(x,y2,'b',lw=1.5, label = "ML Predict")
    ax.fill_between(x, y2 - two_sigma, y2 + two_sigma,color='red', alpha=0.2, label= '$2\sigma$')
    ax.set_xlim([min(x),max(x)])
    ax.set_xlabel("data set no.")
    ax.set_ylabel('$P_{rad}$')
    ax.legend(loc='best')


    plt.savefig(pjoin(dir, 'model_predict_compare.png'),dpi=dpi)
    plt.show()

def dnn_history_plot(loss,val_loss,dir):
    # idx0 = 1
    fig,ax = plt.subplots(1,1,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
    ax.semilogy(loss, lw=1.5)
    ax.semilogy(val_loss, lw=1.5)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('L2 loss')
    ax.legend(['Training loss', 'Validation loss'])
    plt.savefig(pjoin(dir, 'model_history.png'),dpi=dpi)
    plt.show()
