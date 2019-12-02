import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'./AD'))

import ElemFunc as EF
from ADiff import ADiff
import pprint as pp

def myfunc1(x,y):
    f1=x*y
    f2=EF.sin(x)
    f3=10
    f4=x+y+EF.sin(x*y)+10
    return [f1+f2, -(f3-f4), f1,5]

def myfunc2(x,y,z):
        f1=x*y
        return ((f1+10)/3)/z

def myfunc3(x):
        f1=x**2
        return f1

def myfunc4(x):
        f1=x**2
        f2=x**x
        f3=3**x**x
        return [f1, f2,f3]

def myfunc5(x,y):
        f1=3**x**y
        return f1


print("======= myfunc1 =======")
print("Output Jacobian J at c:")
c=[1,2]
f_obj=ADiff(myfunc1)
res=f_obj.Jac(c)
pp.pprint(res)
print("")

print("Output J*p directly without needing full J:")
c=[1,2]
p=[2,3]
f_obj=ADiff(myfunc1)
res=f_obj.pJac(c,p)
pp.pprint(res)
print("======= END ==========")
print("\n")



print("======= myfunc2 =======")
print("Output Jacobian at c:")
p=[2,3]
c=[1,2,3]
f_obj=ADiff(myfunc2)
res=f_obj.Jac(c)
pp.pprint(res)
print("")

print("Output J*p directly without needing full J:")
p=[2,3,4]
c=[1,2,3]
f_obj=ADiff(myfunc2)
res=f_obj.pJac(c,p)
pp.pprint(res)
print("======= END ==========")
print("\n")



print("======= myfunc3 =======")
print("Output Jacobian at c:")
c=3
f_obj=ADiff(myfunc3)
res=f_obj.Jac(c)
pp.pprint(res)
print("")

print("Output J*p directly without needing full J:")
p=3
c=3
f_obj=ADiff(myfunc3)
res=f_obj.pJac(c,p)
pp.pprint(res)
print("======= END ==========")
print("\n")



print("======= myfunc4 =======")
print("Output Jacobian at c:")
c=2
f_obj=ADiff(myfunc4)
res=f_obj.Jac(c)
pp.pprint(res)
print("")


print("Output J*p directly without needing full J:")
p=3
c=2
f_obj=ADiff(myfunc4)
res=f_obj.pJac(c,p)
pp.pprint(res)
print("======= END ==========")
print("\n")



print("======= myfunc5 =======")
print("Output Jacobian at c:")
c=[1,2]
f_obj=ADiff(myfunc5)
res=f_obj.Jac(c)

pp.pprint(res)
print("")

print("Output J*p directly without needing full J:")
p=[2,3]
c=[1,2]
f_obj=ADiff(myfunc5)
res=f_obj.pJac(c,p)

pp.pprint(res)
print("======= END ==========")
print("\n")
