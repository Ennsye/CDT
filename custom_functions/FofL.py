# ALL PARAMETERS EXCEPT L MUST BE STRINGS, L IS A FLOAT
import numpy as np
import os
from pathlib import Path
module_dir = Path(__file__).parents[0]

def constant_force(L, F):
    return float(F)
    
def linear_force(L, F0, L0, k):
    return (L-float(L0))*float(k) + float(F0)
    
def gas_ram(L, F0, L0, x0, adiabatic_index):
    return float(F0)*(float(x0)/(float(x0) - L + float(L0)))**float(adiabatic_index)
    
def interpolate(L, fname, delimiter):
    # loads from a text file with two columns (L, F) separated by delimiter
    # newlines MUST BE \n
    fpath = os.path.abspath(os.path.join(module_dir, fname))
    data = np.loadtxt(fpath, delimiter=delimiter) # float type by default
    Ld = data[:,0]
    Fd = data[:,1]
    return np.interp(L, Ld, Fd)
