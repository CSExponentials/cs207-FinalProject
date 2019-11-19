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
        '''
        INPUTS
        =======
        other: Numeric value or AutoDiff instance

        RETURNS
        ========
        AutoDiff instance where the value is the result of multiplying and the derivative is the derivative of multiplying.
        '''
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
        '''
        INPUTS
        =======
        other: Numeric value or AutoDiff instance

        RETURNS
        ========
        AutoDiff instance where the value is the result of adding and the derivative is the derivative of adding.
        '''

        ret=AutoDiff(0,0) # This is what we will return eventually

        try:
            ret.val=self.val+other.val
            ret.der=self.der+other.der
        except AttributeError:
            ret.val=self.val+other
            ret.der=self.der
        return ret

    def __sub__(self, other):
        '''
        INPUTS
        =======
        other: Numeric value or AutoDiff instance

        RETURNS
        ========
        AutoDiff instance where the value is the result of substracting and the derivative is the derivative of substracting.
        '''

        ret=AutoDiff(0,0) # This is what we will return eventually

        try:
            ret.val=self.val-other.val
            ret.der=self.der-other.der
        except AttributeError:
            ret.val=self.val-other
            ret.der=self.der
        return ret

    def __rsub__(self, other):
        '''
        INPUTS
        =======
        other: Numeric value

        RETURNS
        ========
        AutoDiff instance where the value is the result of substracting and the derivative is the derivative of substracting.
        '''

        ret=AutoDiff(other-self.val,-self.der) # This is what we will return
        return ret

    def __neg__(self):
        ret=AutoDiff(-self.val,-self.der) # This is what we will return
        return ret


    def __truediv__(self, other):
        '''
        INPUTS
        =======
        other: Numeric value or AutoDiff instance

        RETURNS
        ========
        AutoDiff instance where the value is the result of dividing and the derivative is the derivative of dividing.
        '''

        ret=AutoDiff(0,0) # This is what we will return eventually

        try:
            ret.val=self.val/other.val
            ret.der=(self.der*other.val-other.der*self.val)/(other.val**2)
        except AttributeError:
            ret.val=self.val/other
            ret.der=self.der/other
        return ret

    def __rtruediv__(self, other):
        '''
        INPUTS
        =======
        other: Numeric value

        RETURNS
        ========
        AutoDiff instance where the value is the result of dividing and the derivative is the derivative of dividing.
        '''

        ret=AutoDiff(other/self.val,-other*self.der/(self.val**2))

        return ret

    def __pow__(self, other):
        '''
        INPUTS
        =======
        other: Numeric value or AutoDiff instance

        RETURNS
        ========
        AutoDiff instance where the value is the result of powering and the derivative is the derivative of powering.
        '''

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
        '''
        INPUTS
        =======
        other: Numeric value

        RETURNS
        ========
        AutoDiff instance where the value is the result of powering and the derivative is the derivative of powering.
        '''
        ret=AutoDiff(other**self.val, math.log(other)*other**(self.val)*self.der)

        return ret



    __rmul__=__mul__
    __radd__ = __add__
