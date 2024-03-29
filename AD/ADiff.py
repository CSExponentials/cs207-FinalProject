from .AutoDiff import AutoDiff

class ADiff():
    def __init__(self, func):
        self.func=func

    # Output Jp where the value is a scalar if f is scalar and list if f is
    # vector function; Jp is either scalar for scalar f and list if f is vector
    # function
    def pJac(self,c,p):
        """ Returns J*p where J is Jacobian at c

        INPUTS
        =======
        p: direction to which J is applied to: J*p
        c: point at which J is evaluated at

        RETURNS
        ========
        J*p
        """

        try:
            clen=len(c)
            plen=len(p)
        except TypeError:
            # c,p are scalar
            clen=1
            plen=1


        varNum=clen
        if clen!=plen:
            raise Exception("c's size does not match p's size")

        # Preallocate array for variable objects
        varList=[None]*varNum

        # Instantiate the AutoDiff objects for each variable
        for i in range(varNum):
            try:
                varList[i] = AutoDiff(c[i], p[i])
            except TypeError:
                # c,p are scalar
                varList[i] = AutoDiff(c, p)

        f=self.func(*varList)




        try:
            # Preallocate array for directional derivative output
            derList=[None]*len(f)

            # Preallocate array for directional value output
            valList=[None]*len(f)

            lenf=len(f)

        except TypeError:

            # This is when f returns a scalar
            derList=0
            valList=0
            lenf=1

        for i in range(lenf):
            try:
                derList[i]=f[i].der
                valList[i]=f[i].val

            except TypeError:
                # This is when f returns a scalar
                derList=f.der
                valList=f.val
            except AttributeError:
                # This is when some entry of f is just constant
                derList[i]=0
                valList[i]=f[i]



        return {"value": valList, "diff": derList}

    # Output entire Jacobian
    def Jac(self,c):
        """ Returns J where J is Jacobian at c

        INPUTS
        =======
        c: point at which J is evaluated at

        RETURNS
        ========
        J
        """

        try:
            clen=len(c)
        except TypeError:
            # c is scalar
            clen=1

        varNum=clen


        # Preallocate to store entire Jacobian matrix
        JacList=[None]*varNum

        # Fill row by row
        for i in range(varNum):

            # Compute partial derivative/gradient wrt i-th var
            if varNum==1:
                # when there is one variable only
                p=1

            else:
                p=[0]*varNum
                p[i]=1
            pjaci=ADiff.pJac(self,c,p)
            JacList[i]=pjaci["diff"]

            if i==0:
                valList=pjaci["value"]


        try:
            return {"value": valList, "diff": [list(i) for i in zip(*JacList)]}

        except TypeError:
            # This is when f returns a scalar

            if varNum==1:
                return {"value": valList, "diff": JacList[0]}
            else:
                return {"value": valList, "diff": JacList}
