import ElemFunc as EF
import ADDiff as AD
import pprint as pp

def myfunc(x,y):
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

print("Direction derivative of f at direction p, evaluated at x0")  
p=[2,3]
c=[1,2]
f_obj=AD.ADDiff(myfunc)
res=f_obj.Jac(c)
pp.pprint(res)

print("Direction derivative of f at direction p, evaluated at x0")  
p=[2,3]
c=[1,2,3]
f_obj=AD.ADDiff(myfunc2)
res=f_obj.Jac(c)

pp.pprint(res)

print("Direction derivative of f at direction p, evaluated at x0")  
p=[2,3]
c=3
f_obj=AD.ADDiff(myfunc3)
res=f_obj.Jac(c)

pp.pprint(res)

print("Direction derivative of f at direction p, evaluated at x0")  
p=[2,3]
c=2
f_obj=AD.ADDiff(myfunc4)
res=f_obj.Jac(c)

pp.pprint(res)



print("Direction derivative of f at direction p, evaluated at x0")  
p=[2,3]
c=[1,2]
f_obj=AD.ADDiff(myfunc5)
res=f_obj.Jac(c)

pp.pprint(res)