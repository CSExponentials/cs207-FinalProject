import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../AD'))

import ADiff as AD
import numpy as np

from Sampler import Sampler

class MALASampler(Sampler):
    def __init__(self, target, tau=0.5):

        """
        The constructor initialize all tuning parameters and target
        """
        super(MALASampler, self).__init__(target)
        # To be tuned
        self.tau=tau

    def sample(self, steps_, X0, burnin=0, liveoutput=-1, diagfun=None):

        """ Sample using MALA sampler and current paramters

        INPUTS
        =======
        steps_: Number of steps or samples
        X0: The initial point of the sample
        liveoutput: number of steps to print the current status, -1 means no output
        burnin: Number of initial samples to rid of
        diagfun: a scalar function that takes each sample as input, we output mean of this function
        for diagnostics of convergence

        RETURNS
        ========
        XsmpHa: each row is a sample point and there are steps_ number of samples; all burin period
        samples are deleted
        """

        acceptsteps=0
        summovesize=0

        # Number of arguments of the target function
        dim=self.dim

        # tau as defined by user
        tau=self.tau

        # log target which we will require AD
        AD_logtarget=self.AD_logtarget

        # Preallocate sampling results
        XsmpHa=np.zeros((steps_,dim))

        # Assign initial position
        XsmpHa[0,:]=X0


        for i in range(1,steps_):

            # Extract sample from last step
            Xtm1=XsmpHa[i-1,:]

            # log pi gradient at Xtm1
            dellogpiXtm1=np.array(AD_logtarget.Jac(Xtm1)['diff'])

            # Generate the proposal by Langevin dynamics Xst

            mu=Xtm1+tau*dellogpiXtm1
            cov=2*tau*np.identity(dim)
            Xst=np.random.multivariate_normal(mu,cov,1)
            Xst=Xst[0]

            # If target at the proposal is already 0, we will simply reject the proposal
            if self.target(*Xst).val==0:
                continue

            # log pi gradient at Xst
            dellogpiXst=np.array(AD_logtarget.Jac(Xst)['diff'])

            # Evaluate density q(x'|x) for acceptance ratio
            qXtm1Xst=np.exp((-1/(4*tau)*(np.linalg.norm(Xtm1-Xst-tau*dellogpiXst,2))**2))
            qXstXtm1=np.exp((-1/(4*tau)*(np.linalg.norm(Xst-Xtm1-tau*dellogpiXtm1,2))**2))

            # The acceptance ratio
            alpha=min(1, (self.target(*Xst).val*qXtm1Xst)/(self.target(*Xtm1).val*qXstXtm1))

            # Toss a coin
            coin_=np.random.uniform(0,1)


            if coin_<alpha: # Accept the proposal
                XsmpHa[i,:]=Xst
                acceptsteps=acceptsteps+1
                summovesize=summovesize+np.linalg.norm(Xst-Xtm1,2)
            else: # Reject the proposal
                XsmpHa[i,:]=XsmpHa[i-1,:]

            self.printDiagnostics(i, liveoutput, diagfun, XsmpHa, summovesize, acceptsteps)

        # Populate the diagonostic statistics
        if (acceptsteps==0):
            raise Exception('No proposal was accepted! Consider Re-tuning parameters')

        self.setParams(XsmpHa, acceptsteps, steps_, summovesize)

        return XsmpHa
