from AutoDiff import AutoDiff
import math as math

def sin(x):
    """Returns the value and derivative of sine of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is sine of x and the derivative is the derivative of sine of x.

    EXAMPLES
    =========
    >>> sin(π/2)
    Value: 1, Derivative: 0
    >>> a = AutoDiff(π/2,1)
    >>> sin(a)
    Value: 1.0, Derivative: 6.123233995736766e-17
    """
    try:
        return AutoDiff(math.sin(x.val), math.cos(x.val)*x.der)
    except AttributeError:
        return AutoDiff(math.sin(x),0)


def cos(x):
    """Returns the value and derivative of cosine of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is cosine of x and the derivative is the derivative of cosine of x.

    EXAMPLES
    =========
    >>> cos(π/2)
    Value: 6.123233995736766e-17, Derivative: 0
    >>> a = AutoDiff(π/2,1)
    >>> cos(a)
    Value: 6.123233995736766e-17, Derivative: -1.0
    """
    try:
        return AutoDiff(math.cos(x.val), -math.sin(x.val)*x.der)
    except AttributeError:
        return AutoDiff(math.cos(x),0)

def exp(x):
    """Returns the value and derivative of e^x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is e^x and the derivative is the derivative of e^x.

    EXAMPLES
    =========
    >>> exp(π/2)
    Value: 4.810477380965351, Derivative: 0
    >>> a = AutoDiff(π/2,1)
    >>> exp(a)
    Value: 4.810477380965351, Derivative: 4.810477380965351
    """
    try:
        return AutoDiff(math.exp(x.val), math.exp(x.val)*x.der)
    except AttributeError:
        return AutoDiff(math.exp(x),0)

def exp_base(y,x):
    """Returns the value and derivative of the exponential with x as exponent and y as base as an AutoDiff instance.

    INPUTS
    =======
    y: base
    x: exponent: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is the exponential with x as exponent and y as base and the derivative is its derivative.

    EXAMPLES
    =========
    >>> exp_base(2,2)
    Value: 4.0, Derivative: 0
    >>> a = AutoDiff(2,1)
    >>> exp_base(2,a)
    Value 4.0, Derivative 4.0
    """
    try:
        return AutoDiff(math.pow(y,x.val), math.pow(y,x.val)*x.der)
    except AttributeError:
        return AutoDiff(math.pow(y,x),0)

def logistic(x):
    """Returns the value and derivative of the logistic sigmoid of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is the logistic sigmoid of x and the derivative is the derivative of the logistic sigmoid of x.

    EXAMPLES
    =========
    >>> logistic(2)
    Value: 0.8807970779778823, Derivative: 0
    >>> a = AutoDiff(2,1)
    >>> logistic(a)
    Value: 0.8807970779778823, Derivative 0.10499358540350662
    """
    try:
        return AutoDiff((1 / (1 + math.exp(-x.val))), (1 / (1 + math.exp(-x.val)))*(1-(1 / (1 + math.exp(-x.val))))*x.der)
    except AttributeError:
        return AutoDiff(1 / (1 + math.exp(-x)),0)

def cosh(x):
    """Returns the value and derivative of the hyperbolic cosine of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is the hyperbolic cosine of x and the derivative is the derivative of the hyperbolic cosine of x.

    EXAMPLES
    =========
    >>> cosh(2)
    Value: 3.7621956910836314, Derivative: 0
    >>> a = AutoDiff(2,1)
    >>> cosh(a)
    Value 3.7621956910836314, Derivative: 3.6268604078470186
    """
    try:
        return AutoDiff(((math.exp(x.val)+math.exp(-x.val))/2), ((math.exp(x.val)-math.exp(-x.val))/2)*x.der)
    except AttributeError:
        return AutoDiff(((math.exp(x)+math.exp(-x))/2),0)

def sinh(x):
    """Returns the value and derivative of the hyperbolic sine of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is the hyperbolic sine of x and the derivative is the derivative of the hyperbolic sine of x.

    EXAMPLES
    =========
    >>> sinh(2)
    Value: 3.6268604078470186, Derivative 0
    >>> a = AutoDiff(2,1)
    >>> sinh(a)
    Value: 3.6268604078470186, Derivative: 3.7621956910836314

    """
    try:
        return AutoDiff(((math.exp(x.val)-math.exp(-x.val))/2), ((math.exp(x.val)+math.exp(-x.val))/2)*x.der)
    except AttributeError:
        return AutoDiff(((math.exp(x)-math.exp(-x))/2),0)

def tanh(x):
    """Returns the value and derivative of the hyperbolic tangent of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is the hyperbolic tangent of x and the derivative is the derivative of the hyperbolic tangent of x.

    EXAMPLES
    =========
    >>> tanh(2)
    Value: 0.9640275800758169, Derivative: 0
    >>> a = AutoDiff(2,1)
    >>> tanh(a)
    Value 0.9640275800758169, Derivative: 1
    """
    try:
        return AutoDiff(((math.exp(x.val)-math.exp(-x.val))/2)/((math.exp(x.val)+math.exp(-x.val))/2), ((1/((math.exp(x.val)+math.exp(-x.val))/2)*((math.exp(x.val)+math.exp(-x.val))/2))*x.der))
    except AttributeError:
        return AutoDiff(((math.exp(x)-math.exp(-x))/2)/((math.exp(x)+math.exp(-x))/2),0)

def log(x, b = math.e):
    """Returns the value and derivative of log base e of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance
    b: Base. Without this argument the natural logarithm (base e) will be returned

    RETURNS
    ========
    AutoDiff instance where the value is log base e of x and the derivative is the derivative of log base e of x.

    EXAMPLES
    =========
    >>> log(5,2)
    Value: 2.321928094887362, Derivative: 0
    >>> a = AutoDiff(5,1)
    >>> log(a,2)
    Value: 2.321928094887362, Diff: 0.2
    """
    try:
        return AutoDiff(math.log(x.val,b), (1/((x.val)*math.log(b)))*x.der)
    except AttributeError:
        return AutoDiff(math.log(x,b),0)

def tan(x):
    """Returns the value and derivative of tangent of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is tangent of x and the derivative is the derivative of tangent of x.

    EXAMPLES
    =========
    >>> tan(2)
    Value: -2.185039863261519, Derivative: 0
    >>> a = AutoDiff(2,1)
    >>> tan(a)
    Value: -2.185039863261519, Derivative: 5.774399204041918
    """
    try:
        return AutoDiff(math.tan(x.val), (2/(math.cos(x.val*2)+1))*x.der)
    except AttributeError:
        return AutoDiff(math.tan(x),0)

def cot(x):
     """Returns the value and derivative of cotangent of x as an AutoDiff instance.

     INPUTS
     =======
     x: Numeric value or AutoDiff instance

     RETURNS
     ========
     AutoDiff instance where the value is cotangent of x and the derivative is the derivative of cotangent of x.

     EXAMPLES
     =========
     >>> cot(π/2)
     Value: 6.123233995736766e-17, Derivative: 0
     >>> a = AutoDiff(π/2,1)
     >>> cot(a)
     Value: 6.123233995736766e-17, Derivative: -1.0)
     """
     try:
         return AutoDiff(math.cos(x.val)/math.sin(x.val), (2/(math.cos(x.val*2)-1))*x.der)
     except AttributeError:
         return AutoDiff(math.cos(x)/math.sin(x),0)


def sec(x):
    """Returns the value and derivative of secant of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is secant of x and the derivative is the derivative of secant of x.

    EXAMPLES
    =========
    >>> sec(π/2)
    Value: 1.633123935319537e+16, Derivative: 0
    >>> a = AutoDiff(π/2,1)
    >>> sec(a)
    Value: 1.633123935319537e+16, Derivative: 2.667093788113571e+32
    """
    try:
        return AutoDiff(1/(math.cos(x.val)), (math.tan(x.val)*1/(math.cos(x.val))*x.der))
    except AttributeError:
        return AutoDiff(1/math.cos(x),0)

def csc(x):
     """Returns the value and derivative of cosecant of x as an AutoDiff instance.

     INPUTS
     =======
     x: Numeric value or AutoDiff instance

     RETURNS
     ========
     AutoDiff instance where the value is cosecant of x and the derivative is the derivative of cosecant of x.

     EXAMPLES
     =========
     >>> csc(π/2)
     Value: 1, Derivative: 0
     >>> a = AutoDiff(π/2,1)
     >>> csc(a)
     Value: 1.0, Derivative: 6.123233995736766e-17
     """
     try:
         return AutoDiff(1/math.sin(x.val), (-1/math.sin(x.val)*math.cos(x.val)/math.sin(x.val))*x.der)
     except AttributeError:
         return AutoDiff(1/math.sin(x),0)

def arcsin(x):
    """Returns the value and derivative of the inverse of sine of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is the inverse of sine of x and the derivative is the derivative of the inverse of sine of x.

    EXAMPLES
    =========
    >>> arcsin(1)
    Value: 1.5707963267948966, Derivative: 0
    >>> a = AutoDiff(0,1)
    >>> arcsin(a)
    Value: 0.0, Derivative: 1.0
    """
    try:
        return AutoDiff(math.asin(x.val), (1/math.sqrt(1-(x.val)*(x.val)))*x.der)
    except AttributeError:
        return AutoDiff(math.asin(x),0)

def arccos(x):
    """Returns the value and derivative of the inverse of cosine of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is the inverse of cosine of x and the derivative is the derivative of the inverse of cosine of x.

    EXAMPLES
    =========
    >>> arccos(1)
    Value: 0.0, Derivative: 0
    >>> a = AutoDiff(0,1)
    >>> arccos(a)
    Value: 1.5707963267948966, Derivative: -1.0
    """
    try:
        return AutoDiff(math.acos(x.val), (-1/math.sqrt(1-(x.val)*(x.val)))*x.der)
    except AttributeError:
        return AutoDiff(math.acos(x),0)

def arctan(x):
    """Returns the value and derivative of the inverse of tangent of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is the inverse of tangent of x and the derivative is the derivative of the inverse of tangent of x.

    EXAMPLES
    =========
    >>> arctan(1)
    Value: 0.7853981633974483, Derivative: 0
    >>> a = AutoDiff(0,1)
    >>> arctan(a)
    Value: 0.0, Derivative: -1.0
    """
    try:
        return AutoDiff(math.atan(x.val), (1/(1+(x.val)*(x.val))*x.der))
    except AttributeError:
        return AutoDiff(math.atan(x),0)
