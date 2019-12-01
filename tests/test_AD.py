import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../AD'))

import ElemFunc as EF
from ADiff import ADiff

import pytest
import math as math

# Test a bunch of unit scenarios of one input for scalar functions
def test_log():
    """Boolean condition asserts that value and derivative of the natural logarithm of the AutoDiff instance are equal to the expected value and derivative as calculated in the function.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=14
    def myfunc(x):
        f1=EF.log(x)
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': 1/c, 'value': math.log(c)}

    assert res==expectAns

def test_sin():

    """Boolean condition asserts that value and derivative of sine of the AutoDiff instance are equal to the expected value and derivative as calculated in the function.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=14
    def myfunc(x):
        f1=EF.sin(x)
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': math.cos(c), 'value': math.sin(c)}

    assert res==expectAns


def test_cos():
    """Boolean condition asserts that value and derivative of cosine of the AutoDiff instance are equal to the expected value and derivative as calculated in the function.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=14
    def myfunc(x):
        f1=EF.cos(x)
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': -math.sin(c), 'value': math.cos(c)}

    assert res==expectAns

def test_exp():
    """Boolean condition asserts that value and derivative of e^x of the AutoDiff instance are equal to the expected value and derivative as calculated in the function.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """

    c=14
    def myfunc(x):
        f1=EF.exp(x)
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': math.exp(c), 'value': math.exp(c)}

    assert res==expectAns


def test_exp_con():
    """Boolean condition asserts that value and derivative of e^x of the AutoDiff instance are equal to the expected value and derivative as calculated in the function for the case in which x is a real number.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=14
    assert {'diff':EF.exp(c).der, 'value': EF.exp(c).val}=={'diff':0, 'value': math.exp(c)}

def test_sin_con():
    """Boolean condition asserts that value and derivative of sine of the AutoDiff instance are equal to the expected value and derivative as calculated in the function for the case in which x is a real number.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=14
    assert {'diff':EF.sin(c).der, 'value': EF.sin(c).val}=={'diff':0, 'value': math.sin(c)}

def test_cos_con():
    """Boolean condition asserts that value and derivative of cosine of the AutoDiff instance are equal to the expected value and derivative as calculated in the function for the case in which x is a real number.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=14
    assert {'diff':EF.cos(c).der, 'value': EF.cos(c).val}=={'diff':0, 'value': math.cos(c)}

def test_log_con():
    """Boolean condition asserts that value and derivative of the natural logarithm of the AutoDiff instance are equal to the expected value and derivative as calculated in the function for the case in which x is a real number.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=14
    assert {'diff':EF.log(c).der, 'value': EF.log(c).val}=={'diff':0, 'value': math.log(c)}

def test_log10_con():
    """Boolean condition asserts that value and derivative of the logarithm base 10 of the AutoDiff instance are equal to the expected value and derivative as calculated in the function for the case in which x is a real number.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=14
    assert {'diff':EF.log10(c).der, 'value': EF.log10(c).val}=={'diff':0, 'value': math.log10(c)}


def test_tan():
    """Boolean condition asserts that value and derivative of the tangent of the AutoDiff instance are equal to the expected value and derivative as calculated in the function for the case in which x is a real number.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=0.5
    assert {'diff':EF.tan(c).der, 'value': EF.tan(c).val}=={'diff':0, 'value': math.tan(c)}

def test_arcsin():
    """Boolean condition asserts that value and derivative of the inverse of sine of the AutoDiff instance are equal to the expected value and derivative as calculated in the function for the case in which x is a real number.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=0.5
    assert {'diff':EF.arcsin(c).der, 'value': EF.arcsin(c).val}=={'diff':0, 'value': math.asin(c)}

def test_arccos():
    """Boolean condition asserts that value and derivative of the inverse of cosine of the AutoDiff instance are equal to the expected value and derivative as calculated in the function for the case in which x is a real number.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=0.5
    assert {'diff':EF.arccos(c).der, 'value': EF.arccos(c).val}=={'diff':0, 'value': math.acos(c)}

def test_sec():
    """Boolean condition asserts that value and derivative of the secant of the AutoDiff instance are equal to the expected value and derivative as calculated in the function for the case in which x is a real number.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=14
    assert {'diff':EF.sec(c).der, 'value': EF.sec(c).val}=={'diff':0, 'value': 1/math.cos(c)}

def test_csc():
    """Boolean condition asserts that value and derivative of the cosecant of the AutoDiff instance are equal to the expected value and derivative as calculated in the function for the case in which x is a real number.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=14
    assert {'diff':EF.csc(c).der, 'value': EF.csc(c).val}=={'diff':0, 'value': 1/math.sin(c)}

def test_plus():
    """Boolean condition asserts that value and derivative of an addition using the AutoDiff class are equal to the expected value and derivative as calculated in the function.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=[1,2]
    def myfunc(x,y):
        f1=1+x+y+2
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': [1,1], 'value': 6}

    assert res==expectAns

def test_mult():
    """Boolean condition asserts that value and derivative of a multiplication using the AutoDiff class are equal to the expected value and derivative as calculated in the function.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=[1,2]
    def myfunc(x,y):
        f1=1*x*y*2
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': [4,2], 'value': 4}

    assert res==expectAns

def test_div():
    """Boolean condition asserts that value and derivative of a division using the AutoDiff class are equal to the expected value and derivative as calculated in the function.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=[1,2]
    def myfunc(x,y):
        f1=1/x/y/2
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': [-0.25,-1/8], 'value': 0.25}

    assert res==expectAns

def test_power():
    """Boolean condition asserts that value and derivative of a power function using the AutoDiff class are equal to the expected value and derivative as calculated in the function.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=[1,2]
    def myfunc(x,y):
        f1=3**x**y**1
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': [math.log(3)*3**(c[0]**c[1])*c[1]*c[0]**(c[1]-1),math.log(3)*3**(c[0]**c[1])*math.log(c[0])*c[0]**c[1]], 'value': 3**1**2}

    assert res==expectAns

def test_neg_sub():
    """Boolean condition asserts that value and derivative of a subtraction using the AutoDiff class are equal to the expected value and derivative as calculated in the function.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=[1,2]
    def myfunc(x,y):
        f1=1-x-y-2
        return -f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': [1,1], 'value': 4}

    assert res==expectAns


def test_vec_func1():
    """Boolean condition asserts that value and derivative of a function of the Autodiff class comprising several elementary operations are equal to the expected value and derivative as calculated in the function.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    def myfunc(x,y):
        f1=x*y
        f2=EF.sin(x)
        f3=10
        f4=x+y+EF.sin(x*y)+10
        return [f1+f2, -(f3-f4)]

    f_obj=ADiff(myfunc)
    res=f_obj.Jac([1,2])

    expectAns={'diff': [[2.5403023058681398, 1.0], [0.1677063269057152, 0.5838531634528576]],
 'value': [2.8414709848078967, 3.909297426825681]}

    assert res==expectAns
