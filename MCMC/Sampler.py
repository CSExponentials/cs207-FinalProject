import AD
import AD.ADiff as ADiff
import AD.ElemFunc as EF
import numpy as np

import matplotlib.pyplot as plt

class Sampler():
    def __init__(self, target):

        """
        The constructor initialize all tuning parameters and target
        """
        np.random.seed(1)

        self.target=target
        # Count number of argument for the target
        self.dim=target.__code__.co_argcount

        # log target which we will require AD
        def logtarget(*arg):
            return EF.log(target(*arg))

        # Instantiate the AD object
        self.AD_logtarget=ADiff.ADiff(logtarget)

        # This is used to percentage of accepted proposals for each run of the sampling algorithm
        # For MALA, the ideal rate is around 0.9 or higher, which the user should try to achieve by adjust tau
        self.accptRatio=-1

        # This is the average move size which should assit the user in tuning their parameters
        # in combination with acceptance ratio
        self.accptmoveSize=-1

        # This is the variance of the move size which should assit the user in tuning their parameters
        # in combination with acceptance ratio and the average move size
        self.varaccptmoveSize=-1

    def sample(self, steps_, X0, burnin=0, liveoutput=-1, diagfun=None):

        """ Sample using HMC sampler and current paramters
        INPUTS
        =======
        steps_: Number of steps or samples
        X0: The initial point of the sample
        liveoutput: number of steps to print the current status, -1 means no output
        burnin: Number of initial samples to rid of
        """
        raise NotImplementedError


    def getAcceptRatio(self):

        """ Output acceptance rate of the latest run

        RETURNS
        ========
        self.accptRatio: the acceptance rate of the latest run
        """

        return self.accptRatio

    def getAvgMovesize(self):

        """ Output average move size for accepted proposals

        RETURNS
        ========
        self.accptmoveSize: average move size for accepted proposals
        """

        return self.accptmoveSize

    def getVarMovesize(self):

        """ Output variance of move size for accepted proposals

        RETURNS
        ========
        self.varaccptmoveSize: variance of move size for accepted proposals
        """

        return self.varaccptmoveSize

    def setParams(self, XsmpHa, acceptsteps, steps_, summovesize):
        """
        Sets the class parameters afetr sampling
        """
        self.accptRatio=acceptsteps/steps_
        self.accptmoveSize=summovesize/acceptsteps
        self.varaccptmoveSize=np.var(XsmpHa)

    def printDiagnostics(self, i, liveoutput, diagfun, XsmpHa, summovesize, acceptsteps):
        """
        Prints diagnositics of samples and acceptances
        """
        if liveoutput>0:
            if np.mod(i,liveoutput)==0:
                if diagfun==None:
                    def diagfun(samples):
                        return np.mean(samples,1)

                print('======= Step ', i,'=========')
                print('Avg of diagonostic function: ', np.mean(diagfun(XsmpHa[0:i])))
                print('Number of samples obtained out of',i ,': ',acceptsteps)
                print('Acceptance rate: ', acceptsteps/(i+1))

                if acceptsteps==0:
                    avgsize='No acceptance yet'
                else:
                    avgsize=summovesize/acceptsteps

                print('Avg. Accpt Move size: ', avgsize)
                print('')
        return

    def plotSamples(self,XsmpHa):
        if self.dim == 1:
            plt.plot(XsmpHa)
            plt.show()
        elif self.dim == 2:
            plt.scatter(XsmpHa[:,0],XsmpHa[:,1])
            plt.show()
        else:
            raise Exception('Cannot plot more than 2D')
