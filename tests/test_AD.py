import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../AD'))

import ElemFunc as EF
from ADiff import ADiff

#import pytest
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

def test_tanh():
    """Boolean condition asserts that value and derivative of the cosecant of the AutoDiff instance are equal to the expected value and derivative as calculated in the function.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=14
    def myfunc(x):
        f1=EF.tanh(x)
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)
    
    expectAns={'diff': (1/((math.exp(c)+math.exp(-c))/2)*((math.exp(c)+math.exp(-c))/2)), 'value': ((math.exp(c)-math.exp(-c))/2)/((math.exp(c)+math.exp(-c))/2)}
    
    assert res==expectAns
    
def test_sinh():
    """Boolean condition asserts that value and derivative of the cosecant of the AutoDiff instance are equal to the expected value and derivative in the case in which x is a real number.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=2
    assert {'diff':EF.sinh(c).der, 'value': EF.sinh(c).val}=={'diff':math.cosh(c), 'value': math.sinh(c)}

def test_cosh():
    """Boolean condition asserts that value and derivative of the cosecant of the AutoDiff instance are equal to the expected value and derivative as calculated in the function.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=2

    assert {'diff':EF.cosh(c).der, 'value': EF.cosh(c).val}=={'diff':math.sinh(c), 'value': math.cosh(c)}

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

def test_tanh_con():
    """Boolean condition asserts that value and derivative of the cosecant of the AutoDiff instance are equal to the expected value and derivative as calculated in the function for the case in which x is a real number.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    c=14
    assert {'diff':EF.tanh(c).der, 'value': EF.tanh(c).val}=={'diff':0, 'value': ((math.exp(c)-math.exp(-c))/2)/((math.exp(c)+math.exp(-c))/2)}

    
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


    
def test_trig_pJac():
    """Boolean condition asserts that value and derivative of a function of the Autodiff class comprising several elementary operations are equal to the expected value and derivative as calculated in the function. 

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    p=[1]
    c=[0.5]
    def myfunc(x):
        a = (EF.cos(x))
        b = (EF.arcsin(x))
        c = (EF.arctan(x))
        return a - b + c
    
    f_obj=ADiff(myfunc)
    res=f_obj.pJac(c,p)
    expectAns={'diff': -math.sin(c)-(1/math.sqrt(1-c**2))+(1/(1+c**2)), 'value': math.cos(c)-math.asin(c)+math.atan(c)}

    assert res==expectAns
    
def test_trig2_vector():
    """Boolean condition asserts that value and derivative of a function of the Autodiff class comprising several elementary operations are equal to the expected value and derivative as calculated in the function. 

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    p=[1,1,1]
    c=[0.5,0.5,3]
    def myfunc(x,y,z):
        a = (EF.sin(x))
        b = (EF.arccos(y))
        c = (EF.tan(z))
        return a + b + c
    
    f_obj=ADiff(myfunc)
    res=f_obj.pJac(c,p)
    expectAns={'diff': math.cos(c[0])- 1/(math.sqrt(1-c[1]**2))+ 1/(math.cos(c[2])*math.cos(c[2])), 'value': math.sin(c[0])+ math.acos(c[1]) + math.tan(c[2])}

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
    
def test_vec_func2():
    """Boolean condition asserts that value and derivative of a function of the Autodiff class comprising several elementary operations are equal to the expected value and derivative as calculated in the function.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """
    
    c = [1,2]
    p = [1,1]
    def myfunc(x,y):
        a = EF.exp_base(2,x) #base 2 and exponent x
        b = EF.logistic(y)
        c = EF.log(y,2) #log with base 2
        return a + b + c
        
    f_obj=ADiff(myfunc)
    res=f_obj.pJac(c,p)
    
    expectAns={'diff': math.pow(2,c[0])+1/(1 + math.exp(-c[1]))*(1-(1/(1 + math.exp(-c[1]))))+1/((c[1])*math.log(2)), 'value': math.pow(2,c[0])+(1 / (1 + math.exp(-c[1])))+math.log(c[1],2)}

    assert res==expectAns

def test_eq():
    """Boolean condition asserts that value and derivative of an AutoDiff instance are equal to that of a different Autodiff instance.

    RETURNS
    ========
    If the boolean condition returns True nothing is returned. If it is computed to be false, then an AssertionError is raised.
    """

    def myfunc1(x,y):
        f1=1*x*y*2
        return f1

    def myfunc2(x,y):
        f1=1*x*y*4
        return f1

    f_obj1=ADiff(myfunc1)
    res1 = f_obj1 == f_obj1
    f_obj2=ADiff(myfunc2)
    res2 = f_obj1 == f_obj2

    assert res1==True and res2==False
