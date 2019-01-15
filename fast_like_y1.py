import os
import numpy as np
dirname = os.path.split(__file__)[0]

class FastY1Approx:
    def __init__(self, variant):
        if variant not in ['des', 'des_planck']:
            raise ValueError("Unknown variance - should be des or des_planck")
        limits_file = os.path.join(dirname, variant + '_limits.txt')
        mu_file = os.path.join(dirname, variant + '_mu.txt')
        C_file = os.path.join(dirname, variant + '_C.txt')
        self.names = []
        mins = []
        maxs = []
        self.n = 0
        for line in open(limits_file):
            name, minv, maxv = line.split()
            self.names.append(name)
            maxs.append(float(maxv))
            mins.append(float(minv))
            self.n+=1
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.mu = np.loadtxt(mu_file)
        self.C = np.loadtxt(C_file)
        self.invC = np.linalg.inv(self.C)



    def __call__(self, x):
        x = np.array(x)
        if len(x) != self.n:
            raise ValueError("Require vector of length {} (called with length {})".format(self.n, len(x)))
        if ((x<self.mins)|(x>self.maxs)).any():
            return -np.inf
        d = x-self.mu
        logP = -0.5 * (d @ self.invC @ d)
        return logP

def test():
    import emcee
    import pylab
    logpost = FastY1Approx('des')
    walkers = 64
    ndim = logpost.n
    nsamp = 10000
    p0=emcee.utils.sample_ball(logpost.mu, 0.01*logpost.C.diagonal()**0.5,size=walkers)
    sampler=emcee.EnsembleSampler(walkers, ndim, logpost)
    sampler.run_mcmc(p0, nsamp)
    pylab.hist(sampler.flatchain[:,0], bins=50)
    pylab.show()

if __name__ == '__main__':
    test()
