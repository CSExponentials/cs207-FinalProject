import AD
import AD.ElemFunc as EF

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../MCMC'))
from HMC import HMCSampler
from MALA import MALASampler

import numpy as np

def target(x,y):
    return EF.exp(-(1-x)**2-10*(y-x**2)**2)

def diagfun(samples):
    return np.mean(samples,1)

def printVals(sampler,samples):
    print(sampler.getAcceptRatio())
    print(sampler.getAvgMovesize())
    print(sampler.getVarMovesize())
    sampler.plotSamples(samples)

# Demo for MALA Sampler
sampler=MALASampler(target, tau=0.02)
samples=sampler.sample(steps_=100000, X0=np.zeros(2), liveoutput=2000, diagfun=diagfun)
printVals(sampler,samples)

# Demo for HMC Sampler
sampler=HMCSampler(target, ep=0.05, L=100)
samples=sampler.sample(steps_=2000,  X0=np.zeros(2), liveoutput=200)
printVals(sampler,samples)
