import ini
from os.path import join as pjoin

#====================== input_func ======================
def inputParams(inputFile):

    params = ini.parse(open(inputFile).read())

    # runSolver    = bool(params['control']['runSolver'])
    loadModel    = bool(params['control']['loadModel'])

    # x_left = float(params['grid']['x_left'])
    # x_right = float(params['grid']['x_right'])
    # t_end = float(params['grid']['t_end'])
    # dx = float(params['grid']['dx'])
    # dt = float(params['grid']['dt'])

    # D0 = float(params['fparam']['D0'])
    # v0 = float(params['fparam']['v0'])

    # print('CFL = %.4f'%(cfl))
    Nx = int((x_right - x_left)/dx)
    Nt = int(t_end/dt)

    nepoch = int(params['MLparam']['nepoch'])

    # Gaussian Pulse Params iniCond
    ic_mean = float(params['iniCond']['ic_mean'])
    ic_std1 = float(params['iniCond']['ic_std1'])
    ic_std2 = float(params['iniCond']['ic_std2'])
    ic_shift = float(params['iniCond']['ic_shift'])
    ic = [ic_mean, ic_std1, ic_std2, ic_shift]

    testvals = [float(params['testvals']['D_chk']), float(params['testvals']['v_chk']), float(params['testvals']['time'])]

    rawplot = bool(params['plots']['rawplot'])
    histplot = bool(params['plots']['histplot'])
    testplot = bool(params['plots']['testplot'])


    savedir = pjoin('data')

    return x_left, x_right, t_end, D0, v0, dx, dt, Nx, Nt, nepoch, ic, runSolver, loadModel, savedir, testvals, rawplot, testplot, histplot
