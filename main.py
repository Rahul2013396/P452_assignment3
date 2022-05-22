import library
import matplotlib.pyplot as plt
import numpy as np
import copy
import math

#Q1

#Defining importance sampling functions
a = np.e/(np.e-1)
def p(y):
    return a*(np.exp(-y))

def y(x):
    return -np.log(1-x/a)

#integral function
def f(x):
    return np.exp(-x**2)

# No importance Sampling

N = 200
sol1 = library.montecarlo2(f,0,1,N)

#with importance sampling
sol2 = library.monteCarloIntegralImportanceSampling(f,p,y,0,1,N)

print(f"integral without importance sampling ={sol1}")
print()
print(f"integral with importance sampling ={sol2}")




#Q2

# defining function for energy

def f(x,y,z):
    return -n**2*math.pi**2*y


#calling shooting method with rk4
global n
n = 1
x1, y1 = library.Shootingmethod(f, 0, 1, 0, 0)
y1 = library.normalize(y1)
n = 2
x2, y2 = library.Shootingmethod(f, 0, 1, 0, 0)
y2 = library.normalize(y2)


#plotting
plt.plot(x1,y1,label = 'lowest ')
plt.plot(x2,y2,label='second lowest')
plt.xlabel('x')
plt.ylabel('psi')
plt.legend()
plt.savefig('Q2.png')





#Q3
#
# Input parameters
N = 50
N1 = N+2
maxiter = 1000

A = np.zeros((N1,N1))

#function to make boundary conditions
A = library.boundary(A)  

B = copy.deepcopy(A)

#solving for laplace equation
result = library.Laplace2D(A,maxiter,0.00001)



# Final 2D values
im = plt.imshow(A, cmap = 'jet')

plt.title('Final')
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('Q3final.png')
