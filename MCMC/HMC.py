import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../AD'))

import ElemFunc as EF
import ADiff as AD
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


class HMCSampler():
    def __init__(self, target, ep=0.05, L=10, M=None):

        """The constructor initialize all tuning parameters and target

        """

        self.target=target

        # Coutn number of argument for the target
        self.dim=target.__code__.co_argcount


        # Tuning parameters
        self.ep=ep
        self.L=L

        if M==None:
            self.M=np.identity(self.dim)
        else:
            self.M=M


        # log target which we will require AD
        def logtarget(*arg):
            return EF.log(target(*arg))


        # Instantiate the AD object
        self.AD_logtarget=AD.ADiff(logtarget)

        # This is used to percentage of accepted proposals for each run of the sampling algorithm
        # For MALA, the ideal rate is around 0.9 or higher, which the user should try to achieve by adjust tau
        self.accptRatio=-1


        # This is the average move size which should assit the user in tuning their parameters
        # in combination with acceptance ratio
        self.accptmoveSize=-1

        # This is the variance of the move size which should assit the user in tuning their parameters
        # in combination with acceptance ratio and the average move size
        self.varaccptmoveSize=-1

    def _leapfrog(self, P0, Q0):
        """ A simple leap frog algorithm for HMC

        INPUTS
        =======
        P0: this is initial value of the ODE
        Q0: this is initial value of the ODE

        RETURNS
        ========
        Pend: last point on the trajectory
        Qend: last point on the trajectory
        """

        ep=self.ep
        M=self.M
        L=self.L
        AD_logtarget=self.AD_logtarget

        # Initialize vector of the trajectory
        Q=np.array([Q0]);
        P=np.array([P0]);


        # The following loop is the Leap Frog algorithm
        # The trajectory is simulated in L steps
        for l in range(L):

            # Compute the intermediate steps where AD is used
            Pl12=P[l,:]+(ep/2)*np.array((AD_logtarget.Jac(Q[l,:])['diff']))
            Qlp1=Q[l,:]+ep*(np.dot(np.linalg.inv(M), Pl12))
            Plp1=Pl12+(ep/2)*np.array(AD_logtarget.Jac(Qlp1)['diff'])

            # Append current step
            Q=np.append(Q, [Qlp1], axis=0)
            P=np.append(P, [Plp1], axis=0)

        return P[-1], Q[-1]




    def sample(self, steps_, X0, burnin=0, liveoutput=-1, diagfun=None):

        """ Sample using HMC sampler and current paramters

        INPUTS
        =======
        steps_: Number of steps or samples
        X0: The initial point of the sample
        liveoutput: number of steps to print the current status, -1 means no output
        burnin: Number of initial samples to rid of

        RETURNS
        ========
        XsmpHa: each row is a sample point and there are steps_ number of samples; all burin period
        samples are deleted
        """

        acceptsteps=0
        summovesize=0

        target=self.target
        M=self.M




        # target(*x) evaluates to a AutoDiff object, must extract its val
        # attribute to tgval first
        def H(p,*x):
            tgval=target(*x).val
            return -np.log(tgval)+0.5*(np.dot(np.dot(p, np.linalg.inv(M)),p))


        # Preallocate sampling results
        XsmpHa=np.zeros((steps_,self.dim))


        # Assign initial position
        XsmpHa[0,:]=X0



        for i in range(1,steps_):

            # First step is proposed from a multivariate Gaussian
            P0=np.random.multivariate_normal(np.zeros(self.dim),M,1)[0]

            # Extract last step
            Q0=XsmpHa[i-1,:]

            # Use leap frog to solve trajectory
            Pprop, Xprop=self._leapfrog(P0,Q0)

            # Delst here is the acceptance ratio
            Delst=H(P0, *Q0)-H(Pprop, *Xprop)

            # Toss a coin to determine if accept
            coin_=np.random.uniform(0,1)


            if np.log(coin_)<Delst:

                # Accept the proposal
                XsmpHa[i,:]=Xprop
                acceptsteps=acceptsteps+1
                summovesize=summovesize+np.linalg.norm(Q0-Xprop,2)

            else:

                # Reject the proposal
                XsmpHa[i,:]=XsmpHa[i-1,:]



            # =========== Print Diagnositcs =================
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
            # ================ END ============================



        self.accptRatio=acceptsteps/steps_
        self.accptmoveSize=summovesize/acceptsteps
        self.varaccptmoveSize=np.var(XsmpHa)


        return XsmpHa[burnin:None,:]


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

# Demo

def target(x,y):
    return EF.exp(-(1-x)**2-10*(y-x**2)**2)

sampler=HMCSampler(target, ep=0.05, L=100)
samples=sampler.sample(steps_=2000,  X0=np.zeros(2), liveoutput=200)
plt.scatter(samples[:,0], samples[:,1])
print(sampler.getAcceptRatio())
print(sampler.getAvgMovesize())
print(sampler.getVarMovesize())
