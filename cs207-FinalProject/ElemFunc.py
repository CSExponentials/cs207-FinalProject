import AutoDiff as AD
import math as math


def sin(x):
    try:
        return AD.AutoDiff(math.sin(x.val), math.cos(x.val)*x.der)
    except AttributeError:
        return AD.AutoDiff(math.sin(x),0)


def cos(x):
    try: 
        return AD.AutoDiff(math.cos(x.val), -math.sin(x.val)*x.der)
    except AttributeError:
        return AD.AutoDiff(math.cos(x),0)
 
def exp(x):
    try:
        return AD.AutoDiff(math.exp(x.val), math.exp(x.val)*x.der)
    except AttributeError:
        return AD.AutoDiff(math.exp(x),0)


def log(x):
    try:
        return AD.AutoDiff(math.log(x.val), (1/(x.val))*x.der)
    except AttributeError:
        return AD.AutoDiff(math.log(x),0)

def log10(x):
    try:
        return AD.AutoDiff(math.log10(x.val), (1/(x.val))*x.der)
    except AttributeError:
        return AD.AutoDiff(math.log10(x),0)
        
def tan(x):
    try:
        return AD.AutoDiff(math.tan(x.val), (2/(math.cos(x.val*2)+1))*x.der)
    except AttributeError:
        return AD.AutoDiff(math.tan(x),0)
    
def cot(x):
    try:
        return AD.AutoDiff(math.cot(x.val), (2/(math.cos(x.val*2)-1))*x.der)
    except AttributeError:
        return AD.AutoDiff(math.cot(x),0)

def sec(x):
    try:
        return AD.AutoDiff(1/(math.cos(x.val)), (math.tan(x.val)*1/(math.cos(x.val))*x.der))
    except AttributeError:
        return AD.AutoDiff(1/math.cos(x),0)

def csc(x):
    try:
        return AD.AutoDiff(1/math.sin(x.val), (-1/math.sin(x.val)*math.cot(x.val))*x.der)
    except AttributeError:
        return AD.AutoDiff(1/math.sin(x),0)
    
def arcsin(x):
    try:
        return AD.AutoDiff(math.asin(x.val), (1/math.sqrt(1-(x.val)*(x.val)))*x.der)
    except AttributeError:
        return AD.AutoDiff(math.asin(x),0)
    
def arccos(x):
    try:
        return AD.AutoDiff(math.acos(x.val), (-1/math.sqrt(1-(x.val)*(x.val)))*x.der)
    except AttributeError:
        return AD.AutoDiff(math.acos(x),0)

def arctan(x):
    try:
        return AD.AutoDiff(math.atan(x.val), (-1/(1+(x.val)*(x.val))*x.der))
    except AttributeError:
        return AD.AutoDiff(math.atan(x),0)   


        


