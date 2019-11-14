from AD.AutoDiff import AutoDiff
import math as math


def sin(x):
    try:
        return AutoDiff(math.sin(x.val), math.cos(x.val)*x.der)
    except AttributeError:
        return AutoDiff(math.sin(x),0)


def cos(x):
    try:
        return AutoDiff(math.cos(x.val), -math.sin(x.val)*x.der)
    except AttributeError:
        return AutoDiff(math.cos(x),0)

def exp(x):
    try:
        return AutoDiff(math.exp(x.val), math.exp(x.val)*x.der)
    except AttributeError:
        return AutoDiff(math.exp(x),0)


def log(x):
    try:
        return AutoDiff(math.log(x.val), (1/(x.val))*x.der)
    except AttributeError:
        return AutoDiff(math.log(x),0)
