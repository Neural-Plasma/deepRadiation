import numpy as np
from evtk.hl import gridToVTK
from os.path import join as pjoin
import h5py
import os

def vtkwrite(path):
    file_name = "particle"#"rhoNeutral" #"P"
    if os.path.exists(pjoin(path,'vtkdata')) == False:
        os.mkdir(pjoin(path,'vtkdata'))
    h5 = h5py.File(pjoin(path,file_name+'.hdf5'),'r')

    lx, ly = 10., 10.
    dx, dy = 0.05, 0.05
    nx, ny = int(lx/dx), int(ly/dy)

    x = np.linspace(0, lx, nx, dtype='float64')
    y = np.linspace(0, ly, ny, dtype='float64')

    datavtk = data.reshape(nx,ny)

    dp   = h5.attrs["dp"]
    Nt   = h5.attrs["Nt"]

    data_num = np.arange(start=0, stop=Nt, step=dp, dtype=int)

    for i in range(len(data_num)):
        datax = h5["/%d"%data_num[i]+"/position/x"]
        datay = h5["/%d"%data_num[i]+"/position/y"]
        dataz = h5["/%d"%data_num[i]+"/position/z"]
        datax = np.array(datax)
        datay = np.array(datay)
        dataz = np.array(dataz)
        # pointsToVTK(pjoin(path,'vtkdata','points_%d'%i), datax, datay, dataz)
        gridToVTK(pjoin(path,'vtkdata','points_%d'%i), x, y, pointData = {file_name : datavtk})
