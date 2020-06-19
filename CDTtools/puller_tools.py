# case A functions
import numpy as np
import sys

class geometry_custom:
    def __init__(self, Rfunc):
        self.R = Rfunc

class geometry_A:
    def __init__(self, iD):
        self.rs = iD['rs']
        self.d = iD['d']
        if (self.rs < 0) or (self.d < 0):
            sys.exit("One or more of rs, d were negative. Aborting")
        self.beta0 = iD['beta0']
        self.theta0 = iD['theta0']
        
    def beta(self, theta):
        return self.beta0 + theta - self.theta0
    
    def R(self, theta):
        beta = self.beta(theta)
        px = (self.rs*(self.d - self.rs*np.cos(beta))*(-self.d*np.cos(beta) + self.rs)/
              (self.d**2 - 2*self.d*self.rs*np.cos(beta) + self.rs**2) + self.rs*np.cos(beta))
        py = (-self.rs**2*(-self.d*np.cos(beta) + self.rs)*np.sin(beta)/
              (self.d**2 - 2*self.d*self.rs*np.cos(beta) + self.rs**2) + self.rs*np.sin(beta))
        sign = 1
        if py < 0:
            sign = -1
        return sign*((px**2 + py**2)**0.5)
    
    def L(self, theta):
        beta = self.beta(theta)
        return (self.d**2 + self.rs**2 - 2*self.d*self.rs*np.cos(beta))**0.5
        
        
class geometry_B:
    def __init__(self, iD):
        self.rw = iD['rw']
        self.d = iD['d']
        self.rc = iD['rc']
        self.beta0 = iD['beta0']
        self.theta0 = iD['theta0']
        self.gA = geometry_A({'rs': self.rc, 'd': self.d, 'beta0': self.beta0, 'theta0': self.theta0})
        # geometry simplifies to case A for -k <= beta <= k
        self.b = (self.d**2 - self.rw**2)**0.5
        self.a = (self.rc**2 - self.rw**2)**0.5
        self.k = np.arccos(((self.a + self.b)**2 - self.rc**2 - self.d**2)/(-2*self.rc*self.d))
        self.mu = np.arctan(self.a / self.rw)
        self.eta = np.arcsin(self.b / self.d)
        
    def beta(self, theta):
        return self.beta0 + theta - self.theta0
    
    def gamma(self, beta):
        if beta > self.k:
            return beta - self.k
        elif beta < -(self.k):
            return -beta - self.k
        else:
            return 0
        
    def R(self, theta):
        beta = self.beta(theta)
        if beta > self.k:
            return self.rw
        elif beta < -self.k:
            return -self.rw
        else:
            return (self.gA).R(theta)
        
    def L(self, theta):
        beta = self.beta(theta)
        if (beta < self.k) and (beta > -self.k):
            return (self.gA).L(theta)
        else:
            wrapped_len = self.rw*self.gamma(beta)
            return self.rw*self.gamma(beta) + self.a + self.b
            
def taugen(FofL, LofTheta, RofTheta):
    def tau(theta):
        return RofTheta(theta) * FofL(LofTheta(theta))
    return tau
            
            
def torque_fit(f, theta_range, order=10, xdata=None, ydata=None, Npoints=100):
    # where f is a callable giving the exact form of T(theta)
    if (ydata is None) or (xdata is None):
        xdata = np.linspace(theta_range[0], theta_range[1], Npoints, endpoint=True)
        ydata = np.array([f(x) for x in xdata])
    P = np.polynomial.polynomial.Polynomial.fit(xdata, ydata, order, domain=theta_range, window=theta_range)
    # if window!=domain, the object still works properly, but its coefficients are somehow garbage
    return P # a polynomial object that can be evaluated with P.__call__(x)
