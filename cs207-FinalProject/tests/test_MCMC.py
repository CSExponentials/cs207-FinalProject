import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import AD.ElemFunc as EF
from AD.ADiff import ADiff
from MCMC.HMC import HMCSampler
from MCMC.MALA import MALASampler

import pytest
import math as math
import numpy as np


# We will test HMC sampler with a 2D gaussian distribution centered at 0--we will
# check if the sample mean is close enough to 0
def test_HMC():
    """Boolean condition asserts the MCMC output's mean is close to true mean after running for 500 steps

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    def target(x,y):
        return EF.exp(-x**2-y**2)

    sampler=HMCSampler(target, ep=0.05, L=100)
    samples=sampler.sample(steps_=500,  X0=np.zeros(2), liveoutput=100)

    assert np.mean(samples, 0)==pytest.approx([0,0], abs=0.5)
    assert sampler.getAcceptRatio()>=0
    assert sampler.getAvgMovesize()>=0
    
# We will test MALA sampler with a 2D gaussian distribution centered at 0--we will
# check if the sample mean is close enough to 0
def test_MALA():
    """Boolean condition asserts the MCMC output's mean is close to true mean after running for 500 steps

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    def target(x,y):
        return EF.exp(-x**2-y**2)
    
    def diagfun(samples):
        return np.mean(samples,1)

    sampler=MALASampler(target, tau=0.02)
    samples=sampler.sample(steps_=10000, X0=np.zeros(2), liveoutput=1000, diagfun=diagfun)


    assert np.mean(samples, 0)==pytest.approx([0,0], abs=0.5)
    assert sampler.getAcceptRatio()>=0
    assert sampler.getAvgMovesize()>=0
    

