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


def log(x):
    """Returns the value and derivative of log base e of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is log base e of x and the derivative is the derivative of log base e of x.

    EXAMPLES
    =========
    >>> sin(π/2)
    Value: 0.4515827052894548, Derivative: 0
    >>> a = AutoDiff(π/2,1)
    >>> sin(a)
    Value: 0.4515827052894548, Derivative: 0.6366197723675814
    """
    try:
        return AutoDiff(math.log(x.val), (1/(x.val))*x.der)
    except AttributeError:
        return AutoDiff(math.log(x),0)

def log10(x):
    """Returns the value and derivative of log base 10 of x as an AutoDiff instance.

    INPUTS
    =======
    x: Numeric value or AutoDiff instance

    RETURNS
    ========
    AutoDiff instance where the value is log base 10 of x and the derivative is the derivative of log base 10 of x.

    EXAMPLES
    =========
    >>> sin(π/2)
    Value: 0.19611987703015263, Derivative: 0
    >>> a = AutoDiff(π/2,1)
    >>> sin(a)
    Value: 0.19611987703015263, Derivative: 0.6366197723675814
    """
    try:
        return AutoDiff(math.log10(x.val), (1/(x.val))*x.der)
    except AttributeError:
        return AutoDiff(math.log10(x),0)

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
    >>> sin(π/2)
    Value: 1.633123935319537e+16, Derivative: 0
    >>> a = AutoDiff(π,1)
    >>> sin(a)
    Value: -1.2246467991473532e-16, Derivative: 1.0
    """
    try:
        return AutoDiff(math.tan(x.val), (2/(math.cos(x.val*2)+1))*x.der)
    except AttributeError:
        return AutoDiff(math.tan(x),0)

# def cot(x):
#     """Returns the value and derivative of cotangent of x as an AutoDiff instance.
#
#     INPUTS
#     =======
#     x: Numeric value or AutoDiff instance
#
#     RETURNS
#     ========
#     AutoDiff instance where the value is cotangent of x and the derivative is the derivative of cotangent of x.
#
#     EXAMPLES
#     =========
#     >>> cot(π/2)
#     ??
#     >>> a = AutoDiff(π/2,1)
#     >>> cot(a)
#     ??
#     """
#     try:
#         return AutoDiff(math.cot(x.val), (2/(math.cos(x.val*2)-1))*x.der)
#     except AttributeError:
#         return AutoDiff(math.cot(x),0)

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

# def csc(x):
#     """Returns the value and derivative of cosecant of x as an AutoDiff instance.
#
#     INPUTS
#     =======
#     x: Numeric value or AutoDiff instance
#
#     RETURNS
#     ========
#     AutoDiff instance where the value is cosecant of x and the derivative is the derivative of cosecant of x.
#
#     EXAMPLES
#     =========
#     >>> csc(π/2)
#     Value: 1, Derivative: 0
#     >>> a = AutoDiff(π/2,1)
#     >>> csc(a)
#     Value: 1.0, Derivative: 6.123233995736766e-17
#     """
#     try:
#         return AutoDiff(1/math.sin(x.val), (-1/math.sin(x.val)*math.cot(x.val))*x.der)
#     except AttributeError:
#         return AutoDiff(1/math.sin(x),0)

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
        return AutoDiff(math.atan(x.val), (-1/(1+(x.val)*(x.val))*x.der))
    except AttributeError:
        return AutoDiff(math.atan(x),0)
