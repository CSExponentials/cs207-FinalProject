import sys
sys.path.append('../')

from AD import ElemFunc as EF
from AD.ADiff import ADiff

import pytest
import math as math

# Test a bunch of unit scenarios of one input for scalar functions
def test_log():

    c=14
    def myfunc(x):
        f1=EF.log(x)
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': 1/c, 'value': math.log(c)}

    assert res==expectAns

def test_sin():

    c=14
    def myfunc(x):
        f1=EF.sin(x)
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': math.cos(c), 'value': math.sin(c)}

    assert res==expectAns


def test_cos():

    c=14
    def myfunc(x):
        f1=EF.cos(x)
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': -math.sin(c), 'value': math.cos(c)}

    assert res==expectAns

def test_exp():

    c=14
    def myfunc(x):
        f1=EF.exp(x)
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': math.exp(c), 'value': math.exp(c)}

    assert res==expectAns


def test_exp_con():
    c=14
    assert {'diff':EF.exp(c).der, 'value': EF.exp(c).val}=={'diff':0, 'value': math.exp(c)}

def test_sin_con():
    c=14
    assert {'diff':EF.sin(c).der, 'value': EF.sin(c).val}=={'diff':0, 'value': math.sin(c)}

def test_cos_con():
    c=14
    assert {'diff':EF.cos(c).der, 'value': EF.cos(c).val}=={'diff':0, 'value': math.cos(c)}

def test_log_con():
    c=14
    assert {'diff':EF.log(c).der, 'value': EF.log(c).val}=={'diff':0, 'value': math.log(c)}


def test_plus():
    c=[1,2]
    def myfunc(x,y):
        f1=1+x+y+2
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': [1,1], 'value': 6}

    assert res==expectAns

def test_mult():
    c=[1,2]
    def myfunc(x,y):
        f1=1*x*y*2
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': [4,2], 'value': 4}

    assert res==expectAns

def test_div():
    c=[1,2]
    def myfunc(x,y):
        f1=1/x/y/2
        return f1

    f_obj=AD.ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': [-0.25,-1/8], 'value': 0.25}

    assert res==expectAns

def test_power():
    c=[1,2]
    def myfunc(x,y):
        f1=3**x**y**1
        return f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': [math.log(3)*3**(c[0]**c[1])*c[1]*c[0]**(c[1]-1),math.log(3)*3**(c[0]**c[1])*math.log(c[0])*c[0]**c[1]], 'value': 3**1**2}

    assert res==expectAns

def test_neg_sub():

    c=[1,2]
    def myfunc(x,y):
        f1=1-x-y-2
        return -f1

    f_obj=ADiff(myfunc)
    res=f_obj.Jac(c)

    expectAns={'diff': [1,1], 'value': 4}

    assert res==expectAns


def test_vec_func1():

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


test_log()
