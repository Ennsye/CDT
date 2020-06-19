import ctypes
import os
from pathlib import Path
import sys
import numpy as np
import copy
import time
from numpy import cos, sin
from numpy.ctypeslib import ndpointer
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import traceback

import matplotlib.animation as animation
#import matplotlib.pyplot as plt


parent_path = Path(__file__).parents[1]
if os.name == 'nt':
    libpath = os.path.abspath(os.path.join(parent_path, "build_windoze", "libdyn.dll"))
    # due to horrifying numpy bug which causes problems system-wide, need absolute path here
    lib = ctypes.WinDLL(libpath)
    print("detected system: Windoze")
elif os.name == 'posix':
    libpath = os.path.abspath(os.path.join(parent_path, "build_linux", "libdyn.so"))
    lib = ctypes.CDLL(libpath)
    print("detected system: GNU/LINUX")
else:
    sys.exit("unsupported operating system")

# lib is defined in the __init__ file for the package that contains this module, along with the other Python tools for CDT  

class dyn_params(ctypes.Structure):
    _fields_ = [("La", ctypes.c_double), ("Ls", ctypes.c_double), ("ds", ctypes.c_double), ("mb", ctypes.c_double), 
               ("rc", ctypes.c_double), ("Ia", ctypes.c_double), ("mp", ctypes.c_double), ("g", ctypes.c_double)]
# La: arm length
# Ls: sling length
# ds: sling linear density
# mb: arm mass
# rc: distance from arm center of rotation (CoR) to arm center of mass (positive if between CoR and tip)
# Ia: arm rotational inertia about axis of rotation
# mp: combined mass of projectile and pouch
# g: gravitational field strength
# !Make sure you're using a consistent unit system!
    
class T_params(ctypes.Structure):
    _fields_ = [("kw", ctypes.c_double), ("c0", ctypes.c_double), ("c1", ctypes.c_double), ("c2", ctypes.c_double), 
               ("c3", ctypes.c_double), ("c4", ctypes.c_double), ("c5", ctypes.c_double), ("c6", ctypes.c_double), 
               ("c7", ctypes.c_double), ("c8", ctypes.c_double), ("c9", ctypes.c_double), ("c10", ctypes.c_double)]

#    
def ydot_fast(y, dp, T_drive):   
    lib.dyn_ode.restype = ndpointer(dtype=ctypes.c_double, shape=(4,))
    lib.dyn_ode.argtypes = [ctypes.POINTER(ctypes.c_double), dyn_params, ctypes.c_double,]
    dp_C = dyn_params(dp.La, dp.Ls, dp.ds, dp.mb, dp.rc, dp.Ia, dp.mp, dp.g) # initializes the structure to be 
    #passed in. Type is simply dyn_params, as defined in the classdef
    y_Carr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) # initializes array to be passed in
    dyn_res = lib.dyn_ode(y_Carr, dp_C, T_drive)
    return dyn_res

def ydot_fast_trough(y, dp, T_drive):   
    lib.dyn_ode_trough.restype = ndpointer(dtype=ctypes.c_double, shape=(4,))
    lib.dyn_ode_trough.argtypes = [ctypes.POINTER(ctypes.c_double), dyn_params, ctypes.c_double,]
    dp_C = dyn_params(dp.La, dp.Ls, dp.ds, dp.mb, dp.rc, dp.Ia, dp.mp, dp.g) # initializes the structure to be 
    #passed in. Type is simply dyn_params, as defined in the classdef
    y_Carr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) # initializes array to be passed in
    dyn_res = lib.dyn_ode_trough(y_Carr, dp_C, T_drive)
    print(type(dyn_res))
    return dyn_res

def Tdrive_fast(y, tp):
    lib.T.restype = ctypes.c_double
    lib.T.argtypes = [ctypes.POINTER(ctypes.c_double), T_params,]
    tp_C = T_params(tp.kw, tp.c0, tp.c1, tp.c2, tp.c3, tp.c4, tp.c5, tp.c6, tp.c7, tp.c8, tp.c9, tp.c10)
    y_Carr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    return float(lib.T(y_Carr, tp_C))


def dyn_step_RK4_fast(dt, y, dp, tp):
    lib.dyn_step_RK4.restype = ndpointer(dtype=ctypes.c_double, shape=(4,))
    lib.dyn_step_RK4.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double), dyn_params, T_params,]
    y_Carr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    dp_C = dyn_params(dp.La, dp.Ls, dp.ds, dp.mb, dp.rc, dp.Ia, dp.mp, dp.g)
    tp_C = T_params(tp.kw, tp.c0, tp.c1, tp.c2, tp.c3, tp.c4, tp.c5, tp.c6, tp.c7, tp.c8, tp.c9, tp.c10)
    step_res = lib.dyn_step_RK4(dt, y_Carr, dp_C, tp_C)
    return step_res

def dyn_step_trough_RK4_fast(dt, y, dp, tp):
    lib.dyn_step_trough_RK4.restype = ndpointer(dtype=ctypes.c_double, shape=(4,))
    lib.dyn_step_trough_RK4.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double), dyn_params, T_params,]
    y_Carr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    dp_C = dyn_params(dp.La, dp.Ls, dp.ds, dp.mb, dp.rc, dp.Ia, dp.mp, dp.g)
    tp_C = T_params(tp.kw, tp.c0, tp.c1, tp.c2, tp.c3, tp.c4, tp.c5, tp.c6, tp.c7, tp.c8, tp.c9, tp.c10)
    step_res = lib.dyn_step_trough_RK4(dt, y_Carr, dp_C, tp_C)
    return step_res
    
def vp(y, dp):
    theta = y[0]
    thetad = y[1]
    psi = y[2]
    psid = y[3]
    vpx = -dp.La*np.cos(theta)*thetad - dp.Ls*(psid - thetad)*np.cos(psi - theta)
    vpy = -dp.La*np.sin(theta)*thetad + dp.Ls*(psid - thetad)*np.sin(psi - theta)
    return np.array([vpx, vpy])

def vp_angle(y, dp, verbose=False):
    vpv = vp(y, dp)
    if verbose:
        print("vpx = ", vpv[0])
        print("vpy = ", vpv[1])    
    vpa = np.angle(vpv[0] + vpv[1]*1j)
    return vpa

def ap(y, ydot, dp):
    theta = y[0]
    thetad = y[1]
    thetadd = ydot[1]
    psi = y[2]
    psid = y[3]
    psidd = ydot[3]
    apx =  (dp.La*sin(theta)*thetad**2 - dp.La*cos(theta)*thetadd + dp.Ls*(psid - thetad)**2*sin(psi - theta) 
            - dp.Ls*(psidd - thetadd)*cos(psi - theta))

    apy =  (-dp.La*sin(theta)*thetadd - dp.La*cos(theta)*thetad**2 + dp.Ls*(psid - thetad)**2*cos(psi - theta) 
            + dp.Ls*(psidd - thetadd)*sin(psi - theta))
    return np.array([apx, apy])

def dyn_full_simple(y0, dt, dp, tp, psi_final, verbose=True, history=False, termination_tolerance=1e-12):
    # doesn't find exact release point yet, just a demo
    wmax = 0 # maximum arm rotation speed
    y1 = y0
    ydot0 = copy.copy(ydot_fast(y0, dp, Tdrive_fast(y0, tp)))
    y2 = copy.copy(dyn_step_RK4_fast(dt, y1, dp, tp))
    T = 0 # matches the time at y1 
    if history: 
        H = {'Y': [y0,], 'Yd': [ydot0,], 't': [0,]} # time tracks y1, so does history
    while y2[2] < psi_final:
        y1 = copy.copy(y2)
        T += dt # matches the time at y1
        if history:
            y1d = copy.copy(ydot_fast(y1, dp, Tdrive_fast(y1, tp)))
            H['Y'].append(y1) # y2 doesn't get recorded to history, as it might be after the throw
            H['Yd'].append(y1d)
            H['t'].append(T)
        y2 = copy.copy(dyn_step_RK4_fast(dt, y1, dp, tp))
        if np.absolute(y2[1]) > wmax:
            wmax = np.absolute(y2[1])
        
    # now pinpoint the release position (in state space)
    dti = [0, dt]
    dtc = 0.5*(dti[0] + dti[1])
    yc = copy.copy(dyn_step_RK4_fast(dtc, y1, dp, tp))
    while np.absolute(yc[2] - psi_final) > termination_tolerance:
        # binary search to find release time and position in state space
        if yc[2] > psi_final:
            # went too far
            dti[1] = dtc
        elif yc[2] < psi_final:
            dti[0] = dtc
        else:
            dt_final = dtc
            break
        dtc = 0.5*(dti[0] + dti[1])
        yc = copy.copy(dyn_step_RK4_fast(dtc, y1, dp, tp))
    
    dt_final = dtc
    y_final = yc
    t_final = T + dt_final
    ydotf = copy.copy(ydot_fast(y_final, dp, Tdrive_fast(y_final, tp)))
    if history:
        H['Y'].append(y_final)
        H['Yd'].append(ydotf)
        H['t'].append(t_final)
    
    apf = copy.copy(ap(y_final, ydotf, dp))
    if verbose:
        print("wmax = ", wmax)
        print("y_final = ", y_final)
        print("vp = ", np.linalg.norm(vp(y_final, dp)))
        print("vp angle = ", (vp_angle(y_final, dp)))
        print("final projectile acceleration = ", np.linalg.norm(apf))
    if history:
        return {'yf': y_final, 'tf': t_final, 'H': H}
    else:     
        return {'yf': y_final, 'tf': t_final}

def dyn_full_platform(y0, dt, dp, tp, psi_final, verbose=True, history=False, transition_tolerance=1e-8):
    # dynamics where psi is not allowed to drop below its initial value, by means of a rest that
    # the projectile sits in before the speed force overcomes the acceleration force
    # assumes that launch never occurs while projectile is sitting on platform
    v = verbose
    ydot0 = copy.copy(ydot_fast(y0, dp, Tdrive_fast(y0, tp)))
    T1 = 0
    if history:
        H = {'Y': [y0,], 'Yd': [ydot0,], 't': [0,]}
    if ydot0[3] < 0:
        if verbose: 
            print("psi tending to decrease. Prevented by holder. Switching to captive-projectile dynamics")
        # iterate until psidd > 0, then switch to regular dynamics
        y1 = copy.copy(y0)
        y2 = copy.copy(dyn_step_trough_RK4_fast(dt, y1, dp, tp))
        yd2 = copy.copy(ydot_fast(y2, dp, Tdrive_fast(y2, tp)))
        while yd2[3] < 0:
            y1 = copy.copy(y2)
            T1 += dt # T1 and history track y1
            if history:
                y1d = copy.copy(ydot_fast(y1, dp, Tdrive_fast(y1, tp)))
                H['Y'].append(y1)
                H['Yd'].append(y1d)
                H['t'].append(T1)
            y2 = copy.copy(dyn_step_trough_RK4_fast(dt, y1, dp, tp))
            yd2 = copy.copy(ydot_fast(y2, dp, Tdrive_fast(y2, tp)))

        # we now have an interval in which the sign of psidd changes. Pinpoint the sign change (find dtf)
        dti = [0, dt]
        dtc = 0.5*(dti[0] + dti[1])
        yc = copy.copy(dyn_step_trough_RK4_fast(dtc, y1, dp, tp))
        ydc = copy.copy(ydot_fast(yc, dp, Tdrive_fast(yc, tp)))
        while np.absolute(ydc[3]) > transition_tolerance:
            if ydc[3] > 0:
                dti[1] = dtc
            elif ydc[3] < 0:
                dti[0] = dtc
            else:
                dt_trans = dtc
                break
            dtc = 0.5*(dti[0] + dti[1])
            yc = copy.copy(dyn_step_trough_RK4_fast(dtc, y1, dp, tp))
            ydc = copy.copy(ydot_fast(yc, dp, Tdrive_fast(yc, tp)))
        dt_trans = dtc
        T1 += dt_trans
        y02 = copy.copy(yc)
        # don't append to history because it gets recorded as the starting point of the free dynamics
        if verbose:
            print("transition to full dynamics at: y02 = ", y02)
            print("ydot at transition: ", ydc)
    else:
        y02 = y0
    # now continue with the usual dynamics, checking to ensure psi never drops below the starting value
    psi02 = y02[2]
    sol = dyn_full_simple(y02, dt, dp, tp, psi_final, verbose=verbose, history=history)
    if history:
        H['Y'] = H['Y'] + sol['H']['Y']
        H['Yd'] = H['Yd'] + sol['H']['Yd']
        H['t'] = H['t'] + [e + T1 for e in sol['H']['t']]
        return {'yf': sol['yf'], 'tf': sol['tf'] + T1, 'H': copy.deepcopy(H)} # H is a dictionary of lists
    else:
        return {'yf': sol['yf'], 'tf': sol['tf'] + T1}

def dyn_general(y0, dt, dp, tp, psi_final, verbose=True, platform=True, history=False):
    if platform:
        return dyn_full_platform(y0, dt, dp, tp, psi_final, verbose=verbose, history=history)
    else:
        return dyn_full_simple(y0, dt, dp, tp, psi_final, verbose=verbose, history=history)
        
def dft_vpastop(y0, dt, dp, tp, pfbounds, vpa_target, verbose=False, platform=True, history=False):
    # goal is a given launch angle, parameter is psi_final (NOT finger angle)
    # where pfbounds is a sequence containing the search bounds for the value of psi that causes release
    # this is the basic catapult simulation tool, where the projectile releases at some desired launch angle
    def fun(psi_test):
        res = dyn_general(y0, dt, dp, tp, psi_test, verbose=False, platform=platform, history=False)
        vp = vp_angle(res['yf'], dp)
        return np.absolute(vp - vpa_target)
    sol = minimize_scalar(fun, method='bounded', bounds=pfbounds)
    if verbose:
        print("sol = ", sol)
    D = dyn_general(y0, dt, dp, tp, sol.x, verbose=verbose, platform=platform, history=history)
    H = None
    if history:
        H = D['H']
    return {'psi': sol.x, 'yf': D['yf'], 'tf': D['tf'], 'H': H}

# next, we need to be able to vary sling length to get the throw to happen at a  given value of theta
# we could also, equivalently, do this by varying arm inertia

def dft_opt1a(y0, dt, dp, tp, pfbounds, vpa_target, Lsbounds, theta_target, platform=True, history=False, verbose=False):
    # vary release angle and sling length while all else held fixed to achieve a target launch angle and final arm angle
    def f(x):
        # x = [Ls, psif]
        # f(x) = (g(x) - y_target)^2, g(x) = [thetaf, projectile velocity angle] = y_actual
        dptest = dyn_params(dp.La, x[0], dp.ds, dp.mb, dp.rc, dp.Ia, dp.mp, dp.g)
        psif = x[1]
        D = dyn_general(y0, dt, dptest, tp, psif, verbose=False, platform=platform, history=False)
        yf = D['yf']
        vp = vp_angle(yf, dp)
        thetaf = yf[0]
        return ((thetaf - theta_target))**2 + ((vp - vpa_target))**2
        # without normalizing, there is much larger optimization pressure on vp than on thetaf
        # but the results appear to be adequate all the same
    b = (tuple(Lsbounds), tuple(pfbounds))
    x0 = np.array([(Lsbounds[0] + Lsbounds[1])/2, (pfbounds[0] + pfbounds[1])/2]) # middle of the search space
    sol = minimize(f, x0, method='L-BFGS-B', bounds=b)
    psi_sol = sol.x[1]
    Ls_sol = sol.x[0]
    dp_sol = dyn_params(dp.La, Ls_sol, dp.ds, dp.mb, dp.rc, dp.Ia, dp.mp, dp.g)
    print("@dft_opt1a: sol = ", sol)
    D = dyn_general(y0, dt, dp_sol, tp, psi_sol, platform=platform, history=history, verbose=verbose)
    H = None
    if history:
        H = D['H']
    return {'psi_sol': psi_sol, 'yf': D['yf'], 'Ls_opt': Ls_sol, 'tf': D['tf'], 'H': H}
    
def dft_opt2(y0, dt, dp, tp, pfbounds, vpa_target, Iabounds, theta_target, platform=True, history=False, verbose=False):
    # vary release angle and arm inertia while all else held fixed to achieve a target launch angle and final arm angle
    # untested, but structurally identical to dft_opt1a
    def f(x):
        # x = [Ia, psif]
        # f(x) = (g(x) - y_target)^2, g(x) = [thetaf, projectile velocity angle] = y_actual
        dptest = dyn_params(dp.La, dp.Ls, dp.ds, dp.mb, dp.rc, x[0], dp.mp, dp.g)
        psif = x[1]
        D = dyn_general(y0, dt, dptest, tp, psif, verbose=False, platform=platform, history=False)
        yf = D['yf']
        vp = vp_angle(yf, dp)
        thetaf = yf[0]
        return ((thetaf - theta_target))**2 + ((vp - vpa_target))**2
        # without normalizing, there is much larger optimization pressure on vp than on thetaf
        # but the results appear to be adequate all the same
    b = (tuple(Iabounds), tuple(pfbounds))
    x0 = np.array([(Iabounds[0] + Iabounds[1])/2, (pfbounds[0] + pfbounds[1])/2]) # middle of the search space
    sol = minimize(f, x0, method='L-BFGS-B', bounds=b)
    psi_sol = sol.x[1]
    Ia_sol = sol.x[0]
    dp_sol = dyn_params(dp.La, dp.Ls, dp.ds, dp.mb, dp.rc, Ia_sol, dp.mp, dp.g)
    print("@dft_opt1a: sol = ", sol)
    D = dyn_general(y0, dt, dp_sol, tp, psi_sol, platform=platform, history=history, verbose=verbose)
    H = None
    if history:
        H = D['H']
    return {'psi_sol': psi_sol, 'yf': D['yf'], 'Ia_opt': Ia_sol, 'tf': D['tf'], 'H': H}
    
def solver_wrapper(solver, outpipe, *args, **kwargs):
    try:
        sol = solver(*args, **kwargs)
    except Exception as e:
        # possibly the cleverest thing I've ever done: traceback logs from multiproc! :D
        with open('logfile', 'a') as f:
            f.write('\n')
            f.write(str(e))
            f.write(traceback.format_exc())
    outpipe.send(sol)
    return 0

def y_hist(y0, dt, dp, tp, psif, N=0, platform=True):
    # return the state and its time derivative at N evenly spaced points in the history, endpoints inclusive
    sol = dyn_general(y0, dt, dp, tp, psif, platform=platform, history=True)
    tf = sol['tf']
    Y = np.array(sol['H']['Y'])
    Yd = np.array(sol['H']['Yd'])
    t = np.array(sol['H']['t'])
    if N:
        sample_times = np.linspace(0, tf, num=N, endpoint=True)
        # numpy.interp(x, xp, fp, left=None, right=None, period=None)
        yi = []
        ydi = []
        for i in range(4):
            yi.append(np.interp(sample_times, t, Y[:, i])) # stupid numpy can't interpolate vectors...
            ydi.append(np.interp(sample_times, t, Yd[:, i]))
        ys = np.stack(yi, axis=1)
        yds = np.stack(ydi, axis=1)
    else:
        # return the full history
        sample_times = t
        ys = Y
        yds = Yd
    return {'ts': sample_times, 'ys': ys, 'yds': yds}

def tip_force(y, yd, dp):
    # return force on the tip of the arm in BASE FRAME
    # includes contribution from sling mass
    xdd = ap(y, yd, dp) # projectile acceleration vector
    theta, thetad, thetadd, psi, psid, psidd = (y[0], y[1], yd[1], y[2], y[3], yd[3])
    sx = (dp.Ls*(2*dp.La*(sin(theta)*thetad**2 - cos(theta)*thetadd) +
                 dp.Ls*(sin(psi - theta)*psid**2 - 2*sin(psi - theta)*psid*thetad 
                        + sin(psi - theta)*thetad**2 - cos(psi - theta)*psidd + cos(psi - theta)*thetadd))/2)
    sy = (dp.Ls*(-2*dp.La*(sin(theta)*thetadd + cos(theta)*thetad**2) +
                 dp.Ls*(sin(psi - theta)*psidd - sin(psi - theta)*thetadd + cos(psi - theta)*psid**2 - 
                        2*cos(psi - theta)*psid*thetad + cos(psi - theta)*thetad**2))/2)
    fsx = -dp.mp*xdd[0] - dp.ds*sx
    fsy = -dp.mp*xdd[1] - dp.ds*sy # see notes from 2020-05-11 or dynamics_equation_derivation notebook
    # negative signs to give the load on the tip, rather than the load on the "moving parts"
    return np.array([fsx, fsy])


def sling_tension(Y, Yd, t, dp, plot=False, axis=None):
    # return a numpy array containing sling tension for each specified time point
    # optionally, plot the sling tension
    Fs2 = [np.linalg.norm(tip_force(Y[i,:], Yd[i,:], dp)) for i in range(len(t))]
    if plot:
        if axis is None:
            figFs = plt.figure(0)
            axFs = figFs.add_subplot(111)
            axFs.plot(t, Fs2)
            axFs.set_xlabel("time")
            axFs.set_ylabel("force")
            plt.show()
        else:
            axis.plot(t, Fs2)
            axis.set_xlabel("time")
            axis.set_ylabel("force")
    return np.copy(Fs2)

def T_AB(theta):
    # v_A = T_AB*v_B    converts a vector in base frame to tip frame. See user manual for conventions
    R = np.array([[-np.sin(theta), np.cos(theta)], [-np.cos(theta), -np.sin(theta)]])
    return R

def load_info(Y, Yd, t, dp, plot=False, group=None):
    # group: (figure, axF, axO, axA)
    # return a dictionary containing:
    # omega: angular speed of arm. Convention is omega > 0 causes CW motion (i.e. omega = -y[1])
    # alpha: angular acceleration of arm. alpha > 0 causes increasing omega (i.e. alpha = -yd[1])
    # in a classic-looking design, alpha > 0 causes hogging (tension on TOP of beam)
    # Fsy > 0 causes hogging
    # Fsx > 0 causes arm to elongate
    Ftip_BASEFRAME = np.array([tip_force(Y[i, :], Yd[i, :], dp) for i in range(len(t))])
    R = np.array([T_AB(Y[i, 0]) for i in range(len(t))])
    Ftip_ARMFRAME = np.array([np.matmul(R[i], Ftip_BASEFRAME[i]) for i in range(len(t))])
    Ftx = Ftip_ARMFRAME[:, 0]
    Fty = Ftip_ARMFRAME[:, 1]
    omega = -Y[:,1]
    alpha = -Yd[:,1]
    
    if plot:
        if group is None:
            figLI = plt.figure(1, figsize=(7, 6), dpi=128, tight_layout=True)
            axF = figLI.add_subplot(311)
            axO = figLI.add_subplot(312)
            axA = figLI.add_subplot(313)
            group = (figLI, axF, axO, axA)

        group[1].plot(t, Ftx, label="Fx")
        group[1].plot(t, Fty, label="Fy")
        group[1].legend(loc='upper left')
        group[1].set_xlabel("time")
        group[1].set_ylabel("tip load")
        group[1].yaxis.tick_right()
            
        group[2].plot(t, omega)
        group[2].set_xlabel("time")
        group[2].set_ylabel("arm angular speed")
        group[2].yaxis.tick_right()
            
        group[3].plot(t, alpha)
        group[3].set_xlabel("time")
        group[3].set_ylabel("arm angular\nacceleration")
        group[3].yaxis.tick_right()
    
    return {'Fx_tip': Ftx, 'Fy_tip': Fty, 'omega': omega, 'alpha': alpha}
    

def energy_plot(Y, Yd, t, dp, x='time', axE=None):
    # plots two lines: 
    # line1: projectile energy (kinetic + potential)
    # line2: (sling + arm) energy (kinetic + potential)
    # allows easy visualization of the stall point, and how good it is
    # x-axis can be either time, arm_angle, or projectile_velocity_angle
    def f1(y, yd, t, dp):
        yp = dp.La*np.cos(y[0]) - dp.Ls*np.cos(y[2] - y[0])
        ekp = 0.5*dp.mp*(np.linalg.norm(vp(y, dp)))**2
        return ekp + dp.mp*dp.g*yp # kinetic + pot energy of projectile
    
    def f2(y, yd, t, dp):
        theta = y[0]
        thetad = y[1]
        psi = y[2]
        psid = y[3]
        Ts =  ((1/6)*dp.Ls*dp.ds*(3*dp.La**2*thetad**2 + 3*dp.La*dp.Ls*(psid - thetad)*cos(psi)*thetad + 
                            dp.Ls**2*(psid**2 - 2*psid*thetad + thetad**2)))
        Vs = dp.Ls*dp.ds*dp.g*(dp.La*cos(theta) - 0.5*dp.Ls*cos(psi - theta))
        Ta = 0.5 * dp.Ia * (thetad**2)
        ybcom = dp.rc*cos(theta)
        Va = dp.mb*dp.g*ybcom
        return Ts + Ta + Vs + Va
    line1 = np.array([f1(Y[i, :], Yd[i, :], t[i], dp) for i in range(len(t))])
    line2 = np.array([f2(Y[i, :], Yd[i, :], t[i], dp) for i in range(len(t))])
    if x == 'time':
        xpoints = t
    elif x == 'theta':
        xpoints = np.array([Y[i, 0] for i in range(len(t))])
    elif x == 'psi':
        xpoints = np.array([Y[i, 2] for i in range(len(t))])
    elif x == 'projectile speed':
    # def vp(y, dp)
        xpoints = np.array([np.linalg.norm(vp(Y[i, :], dp)) for i in range(len(t))])
    else:
        sys.exit("@energy_plot: unrecognized x option" + x)
    if axE is None:
        figE = plt.figure(2)
        axE = figE.add_subplot(111)
    axE.plot(xpoints, line1, label="projectile energy")    
    axE.plot(xpoints, line2, label="arm + sling energy")
#     axE.annotate('initial', xy=(x[0], line1[0]), xycoords='initial')
    axE.legend(loc="best")
    axE.set_xlabel(x)
    axE.set_ylabel("energy")

    
def puller_potential_energy(tp, theta0, plot=False, plotdomain=None, axPPE=None):
    # plotdomain is a tuple (xmin, xmax)
    p = np.array([tp.c10, tp.c9, tp.c8, tp.c7, tp.c6, tp.c5, tp.c4, tp.c3, tp.c2, tp.c1, tp.c0])
    R = np.roots(p)
    R = (R[np.argwhere((~np.iscomplex(R)) & (R<theta0))][:,0]).astype(float) # indexing fixes stupid argwhere BS
    # R is list containing the real roots of the torque function that are < theta0
    
    def tau(theta):
        return np.polyval(p, theta)
    
    def s(theta):
        # integral of torque wrt theta
        pI = np.polyint(p)
        return np.polyval(pI, theta)
    
    def taud(theta):
        pd = np.polyder(p)
        return np.polyval(pd, theta)
    
    def plot_PPE(R, axPPE):
        if axPPE is None:
            figPPE = plt.figure(3)
            axPPE = figPPE.add_subplot(111)
        if not plotdomain is None:
            xmin = plotdomain[0]
            xmax = plotdomain[1]
        else:
            if len(R) > 0:
                xmin = min(list(R))
            else:
                xmin = -2*np.pi
            xmax = theta0
        x = np.linspace(xmax, xmin, 100)
        y = np.array([tau(t) for t in x])
        svals = np.array([s(t) for t in x])
        axPPE.plot(x, y, label="torque")
        axPPE.plot(x, svals, label="integral of torque")
        axPPE.legend(loc="best")
        axPPE.plot(x[0], y[0], 'ko') # starting point
        for r in R:
            axPPE.plot(r, tau(r), 'ro')
        axPPE.set_xlabel("theta")
        
    if plot:
        plot_PPE(R, axPPE) # before the complicated logic starts
    
    # R contains the values of theta < theta0 for which s has a local optimum
    s0 = s(theta0) # if this is less than all other candidates the arm will initially move backward
    theta_candidates = [t for t in R if taud(t) > 0] # condition for a minimum of s
    if len(theta_candidates) == 0:
        if (tau(theta0) > 0):
            print("UNLIMITED POWER!") # no candidate roots and initial torque > 0 -> torque ALWAYS > 0
            return np.Inf
        else:
            return 0 # not actually, but the arm goes the wrong way
    s_candidates = [s(t) for t in theta_candidates]
    smin = min(s_candidates) # smallest minimum of the indefinite integral for theta < theta0
    if s0 < smin:
        return 0 # no potential energy - the arm rotates backwards
    else:
        return s0 - smin # the definite integral we're looking for
    
def xproj(y, dp):
    # coordinates of projectile in BASEFRAME
    theta, psi = (y[0], y[2])
    xp = -dp.La*np.sin(theta) - dp.Ls*np.sin(psi - theta)
    yp = dp.La*np.cos(theta) - dp.Ls*np.cos(psi - theta)
    return (xp, yp)

def projectile_path(Y, dp, plot=False, fname=None, scale_factor=1, axPP=None):
    Xp = np.array([xproj(Y[i, :], dp) for i in range(Y.shape[0])])
    if plot:
        if axPP is None:
            figPP = plt.figure(4)
            axPP = figPP.add_subplot(111)
        axPP.plot(Xp[:, 0], Xp[:, 1])
        axPP.plot(0, 0, 'o-', lw=3, color='k')

    if not fname is None:
        ns = Y.shape[0]
        if ns > 100:
            # if > 100, interpolate at 100 evenly spaced time points to keep file size down
            all_inds = np.arange(ns) # if ns=3, all_inds=[0, 1, 2]
            ind_sample = np.linspace(0, all_inds[-1], 100, endpoint=True)
            x_sample = np.interp(ind_sample, all_inds, Xp[:,0]) * scale_factor
            y_sample = np.interp(ind_sample, all_inds, Xp[:,1]) * scale_factor
        else:
            x_sample = Xp[:,0]
            y_sample = Xp[:,1]
        xyarr = np.stack((x_sample, y_sample, np.zeros(x_sample.shape)), axis=-1)
        np.savetxt(fname, xyarr) # list of xyz points in SW format
    return (Xp[:, 0], Xp[:, 1]) # (X, Y)
    
def launch_animation(Y, dp, t, axLA=None, figLA=None):
    # animation of the throwing motion
    # make sure to call on evenly spaced 
#     theta = y[0]
    try:
        Xp, Yp = projectile_path(Y, dp, plot=False)
        Xt = np.array([-dp.La*np.sin(Y[i, 0]) for i in range(len(t))])
        Yt = np.array([dp.La*cos(Y[i, 0]) for i in range(len(t))])
        xmin, xmax, ymin, ymax = (min(min(Xp), min(Xt)), max(max(Xp), max(Xt)), min(min(Yp), min(Yt)), max(max(Yp), max(Yt)))
        xrange = xmax - xmin
        yrange = ymax - ymin
    
        if figLA is None:
            figLA = plt.figure(5)
        
        if axLA is None:
            axLA = figLA.add_subplot(111, autoscale_on=False, xlim=(xmin-(xrange/10), xmax+(xrange/10)), 
                                 ylim=(ymin-(yrange/10), ymax+(yrange/10)))
            axLA.set_aspect('equal')
            axLA.grid()
        else:
            axLA.set_autoscale_on(False)
            axLA.set_xlim(xmin-(xrange/8), xmax+(xrange/10))
            axLA.set_ylim(ymin-(yrange/10), ymax+(yrange/6))
            axLA.set_aspect('equal')
            axLA.grid()
    
        lobj1 = axLA.plot([], [], 'o-', lw=3)[0]
        lobj2 = axLA.plot([], [], lw=1)[0]
        
        time_template = 'time = %.3fs'
        time_text = axLA.text(0.05, 0.9, '', transform=axLA.transAxes)
        artists = [lobj1, lobj2, time_text]
    
        
        def init():
            # no typos allowed - all failures are totally silent
            artists[0].set_data([], [])
            artists[1].set_data([], [])
            artists[2].set_text('')
            return artists
    
        def animate_throw(i):
            # better not make any mistakes in here - fails in COMPLETE silence if you do
            mechx = [0, Xt[i], Xp[i]]
            mechy = [0, Yt[i], Yp[i]]
            pathx = Xp[:i+1]
            pathy = Yp[:i+1]
            artists[0].set_data(mechx, mechy)
            artists[1].set_data(pathx, pathy)
            artists[2].set_text(time_template % t[i])
            return artists

        # testing
        #for i in range(1, len(t)):
            # animate_throw(i)

        ani = animation.FuncAnimation(figLA, animate_throw, range(1, len(t)),
                              interval=40, blit=True, init_func=init, repeat=False)
        # interval is delay between frames in ms
        # any mistakes in the line defining ani are apparently handled with a try: ... except: pass sort of error handling logic
        # don't make mistakes!
        #plt.show()
        return ani
    except Exception as e:
        # possibly the cleverest thing I've ever done: traceback logs from multiproc! :D
        with open('logfile', 'a') as f:
            f.write('\n')
            f.write(str(e))
            f.write(traceback.format_exc())
        return 0
    
def axle_reaction_force(Y, Yd, dp, t, plot=False, fname=None, ax_arf=None):
    # reaction load on axle, in base frame. This is the load APPLIED TO the axle, and is
    # what makes the machine jump around. See user manual if confused about sign convention
    theta = Y[:,0]
    omega = Y[:,1]
    f_va = np.array([dp.mb*dp.rc*(omega[i]**2)*np.array([-np.sin(theta[i]), np.cos(theta[i])]) for i in range(len(t))])
    # reasonably confident the above is correct
    # f_va is in base frame
    # def tip_force(y, yd, dp):
    f_tip = np.array([tip_force(Y[i,:], Yd[i,:], dp) for i in range(len(t))])
    f_ar = f_va + f_tip
    if plot:
        if ax_arf is None:
            fig_arf = plt.figure(6)
            ax_arf =fig_arf.add_subplot(111)
        ax_arf.plot(t, f_ar[:,0], label="force in x direction")
        ax_arf.plot(t, f_ar[:,1], label="force in y direction")
        ax_arf.legend(loc="best")
        ax_arf.set_xlabel("time")
        ax_arf.set_ylabel("axle reaction force")
    if not fname is None:
        np.savetxt(fname, f_ar)    
    return f_ar
