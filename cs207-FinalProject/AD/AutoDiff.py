import math as math

class AutoDiff():
    # a is the value to evaluate at
    def __init__(self,a,der):
        self.val=a # Store value
        self.der=der # Store derivative

    def __repr__(self):
        return "Value: {}, Derivative: {}".format(self.val,self.der)

    def __str__(self):
        return "AutoDiff Instance with value {} and derivative {}".format(self.val,self.der)

    def __mul__(self, other):

        # We will use ducktype to handle the possibility of other being a
        # real number or is an object of AutoDiffToy class

        ret=AutoDiff(0,0) # This is what we will return eventually
        try:
            ret.val=self.val*other.val

            # This is the product rule of derivative
            ret.der=self.val*other.der+self.der*other.val
        except AttributeError:
            ret.val=self.val*other
            ret.der=other*self.der
        return ret

    def __add__(self, other):
        # We will use ducktype to handle the possibility of other being a
        # real number or is an object of AutoDiffToy class

        ret=AutoDiff(0,0) # This is what we will return eventually

        try:
            ret.val=self.val+other.val
            ret.der=self.der+other.der
        except AttributeError:
            ret.val=self.val+other
            ret.der=self.der
        return ret

    def __sub__(self, other):
        # We will use ducktype to handle the possibility of other being a
        # real number or is an object of AutoDiffToy class

        ret=AutoDiff(0,0) # This is what we will return eventually

        try:
            ret.val=self.val-other.val
            ret.der=self.der-other.der
        except AttributeError:
            ret.val=self.val-other
            ret.der=self.der
        return ret

    def __rsub__(self, other):
        # we know that __sub__ has failed and other must be a number of form
        # other-self

        ret=AutoDiff(other-self.val,-self.der) # This is what we will return
        return ret

    def __neg__(self):

        ret=AutoDiff(-self.val,-self.der) # This is what we will return
        return ret


    def __truediv__(self, other):
        # We will use ducktype to handle the possibility of other being a
        # real number or is an object of AutoDiffToy class

        ret=AutoDiff(0,0) # This is what we will return eventually

        try:
            ret.val=self.val/other.val
            ret.der=(self.der*other.val-other.der*self.val)/(other.val**2)
        except AttributeError:
            ret.val=self.val/other
            ret.der=self.der/other
        return ret

    def __rtruediv__(self, other):
        # Now since __truediv__ failed to excute, we know we are to evaluate
        # other/self where other is a real number

        # This is what we will return eventually
        ret=AutoDiff(other/self.val,-other*self.der/(self.val**2))

        return ret

    def __pow__(self, other):

        ret=AutoDiff(0,0) # This is what we will return eventually

        try:

            ret.val=self.val**other.val
            ret.der=other.val*self.val**(other.val-1)*self.der+\
            math.log(self.val)*self.val**(other.val)*other.der
        except AttributeError:
            ret.val=self.val**other
            ret.der=other*self.val**(other-1)*self.der
        return ret

    def __rpow__(self,other):

        # Now since __truediv__ failed to excute, we know we are to evaluate
        # other^self where other is a real number

        # This is what we will return eventually
        ret=AutoDiff(other**self.val, math.log(other)*other**(self.val)*self.der)

        return ret



    __rmul__=__mul__
    __radd__ = __add__
