# ALL PARAMETERS EXCEPT theta MUST BE STRINGS, theta IS A FLOAT
import numpy as np
import os
from pathlib import Path
module_dir = Path(__file__).parents[0]

def direct(theta, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10):
    p = np.array([c10, c9, c8, c7, c6, c7, c4, c3, c2, c1, c0])
    p[p=='']=0 # any empty inputs get converted to zeros
    p = p.astype('float')
    return np.polyval(p, theta)
    
def interpolate(theta, fname, delimiter):
    # loads from a text file with two columns (theta, tau) separated by delimiter
    # newlines MUST BE \n
    fpath = os.path.abspath(os.path.join(module_dir, fname))
    data = np.loadtxt(fpath, delimiter=delimiter)
    thetad = data[:,0]
    taud = data[:,1]
    return np.interp(theta, thetad, taud)

