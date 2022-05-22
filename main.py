import library
import matplotlib.pyplot as plt
import numpy as np
import copy
import math

#Q1

a = np.e/(np.e-1)
def p(y):
    return a*(np.exp(-y))

def y(x):
    return -np.log((1-x)/a)

def f(x):
    return np.exp(-x**2)

# No importance Sampling

N = 200
sol1 = library.montecarlo2(f,0,1,N)

sol2 = library.monteCarloIntegralImportanceSampling(f,p,y,0,1,N)

print(f"integral without importance sampling ={sol1}")
print()
print(f"integral with importance sampling ={sol2}")




#Q2


def f(x,y,z):
    return -n**2*math.pi**2*y

global n
n = 1
X1, Y1 = library.ODE_Shooting(f, 0, 1, 0, 0)
n = 2
X2, Y2 = library.ODE_Shooting(f, 0, 1, 0, 0)


Y1 = library.normalize(Y1)
Y2 = library.normalize(Y2)

plt.plot(X1,Y1,label = 'lowest ')
plt.plot(X2,Y2,label='second lowest')
plt.xlabel('x')
plt.ylabel('psi')
plt.legend()
plt.savefig('Q2.png')


#xsol, sol  = library.rk4(1000,0.001,schrodinger_equation,y1,y0)

#plt.plot(xsol,sol)
#plt.show()




#Q3
#
# Input parameters
N = 50
N1 = N+2
maxiter = 1000

A = np.zeros((N1,N1))

A = library.boundary(A)  #Function defined at top

B = copy.deepcopy(A)

#solving for laplace equation
result = library.Laplace2D(A,maxiter,0.00001)

#Initial
im = plt.imshow(B, cmap = 'jet')

plt.title('Initial')
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('Q3initial.png')

# Final
im = plt.imshow(A, cmap = 'jet')

plt.title('Final')
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('Q3final.png')