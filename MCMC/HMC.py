import numpy as np

from Sampler import Sampler

class HMCSampler(Sampler):
    def __init__(self, target, ep=0.05, L=10, M=None):

        """
        The constructor initialize all tuning parameters and target
        """
        super(HMCSampler, self).__init__(target)
        # Tuning parameters
        self.ep=ep
        self.L=L

        if M==None:
            self.M=np.identity(self.dim)
        else:
            self.M=M

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
            tgval=self.target(*x).val
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

            self.printDiagnostics(i, liveoutput, diagfun, XsmpHa, summovesize, acceptsteps)

        self.setParams(XsmpHa, acceptsteps, steps_, summovesize)

        return XsmpHa[burnin:None,:]
