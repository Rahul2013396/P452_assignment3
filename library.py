from matplotlib import pyplot as plt
import numpy as np
import math
import copy

def print_matrix(Matrix):
    for i in range(len(Matrix)):
        for j in range(len(Matrix[i])):
            print(Matrix[i][j], end=", ")
        print("")

def matrix_multiplication(matrix1, matrix2):
    M1_cross_M2 = []
    for i in range(len(matrix1)):
        M1_cross_M2.append([])
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            sum = 0
            for k in range(len(matrix2)):
                sum = sum + (matrix1[i][k] * matrix2[k][j])
            M1_cross_M2[i].append(sum)
    return M1_cross_M2        

def ChebyPolynomialChiSqFit(x, y, u = None, n=4):
    """
    x & y are dataset
    u: Uncertainty, 1 if not given
    n: number of parameters. (order+1). 2 for linear
    """

    def cheby(i,X):
        if(i == 0): return 1
        if(i == 1): return (2*X)-1
        if(i == 2): return (8*X*X)-(8*X)+1
        if(i == 3): return (32*X*X*X)-(48*X*X)+(18*X)-1
    

    if u == None: u = np.ones(len(x))

    A = np.zeros((n,n))
    B = np.zeros(n)
    for i in range(n):
        for j in range(n):
            sum =0 
            for k in range(len(x)):
                sum  += (cheby(i,x[k])*cheby(j,x[k]))/u[k]**2
            A[i][j] = sum
    for i in range(n):
        sum = 0
        for j in range(len(x)):
            sum += cheby(i,x[j])*y[j]/u[j]**2
        B[i] = sum
    C = gauss_sidel(A,B)
    fit = []
    for i in range(len(x)):
        fit.append(C[0]+C[1]*cheby(1,x[i])+C[2]*(cheby(2,x[i]))+C[3]*(cheby(3,x[i]))) # Edit according the order of polynomial needed
    
    cond = np.linalg.cond(A)
    Cov = Inverse(A)
    #print(Cov)

    return fit,cond

def basisPolynomialChiSqFit(x, y, u = None, n=4):
    """
    x & y are dataset
    u: Uncertainty, 1 if not given
    n: number of parameters. (order+1). 2 for linear
    """
    #Functions for different legendre polynomials
    def leg(i,X):
        if(i == 0): return 1
        if(i == 1): return X
        if(i == 2): return (3*(X**2) - 1)/2
        if(i == 3): return (5*(X**3)-3*X)/2
        if(i == 4): return (35*(X**4)-30*(X**2)+3)/8
        if(i == 5): return (63*(X**5)-70*(X**3)+15*X)/8
        if(i == 6): return (231*(X**6)-315*(X**4)+105*(X**2)-5)/16


    if u == None: u = np.ones(len(x))


    A = np.zeros((n,n))
    B = np.zeros(n)
    for i in range(n):
        for j in range(n):
            sum =0 
            for k in range(len(x)):
                sum  += (leg(i,x[k])*leg(j,x[k]))/u[k]**2
            A[i][j] = sum

    for i in range(n):
        sum = 0
        for j in range(len(x)):
            sum += leg(i,x[j])*y[j]/u[j]**2
        B[i] = sum
    #calculating best fit parameters
    C = gauss_sidel(A,B)
    #calulating fitted curve
    fit = np.zeros(len(x))
    for i in range(n):
        fit2=[]
        for j in range(len(x)):
            fit2.append(C[i]*leg(i,x[j]) )
        fit+=fit2    
    
    cond = np.linalg.cond(A)
    
    #Calulationg covariance matrix
    Cov = Inverse(A)
    #print(Cov)

    return fit,Cov,C    

def gaussquad(function, a , b,x,w):
    s=0
    xm = 0.5*(b+a)
    xr = 0.5*(b-a)
    for i in range(len(x)):
        dx = xr*x[i]
        s += w[i]*(function(xm+dx)+function(xm-dx))
    s *=xr
    return s 


def explicit(dt,dx,stept,stepx,k, initial):
    #intial state
    s = initial
    
    
    result=[]
    alpha = k*stept/(stepx**2)
    #looping over time to update the temperature
    for i in range(1,dt):
        # adding result at every time step
        result.append(s)
        d = np.zeros(dx)
        # updating temp at each location
        for j in range(1,dx-1):
            #print(s[j]+alpha*(s[j+1]-2*s[j]+s[j-1]))
            d[j] = s[j]+(alpha*(s[j+1]-2*s[j]+s[j-1]))
        s = d
        
        
    return result

def implicit(dx,dt):
    s = np.zeros(dx)
    d = np.zeros(dx)
    lam=0.71
    D = np.zeros((dx,dx))
    s0 =100
    sn =0
    for i in range(1,dx-1):
        D[i][i+1] = -lam
        D[i][i] = 1+2*lam
        D[i][i-1] = -lam 
        d[i] = s[i]
        D[0][0] = 1+2*lam
        D[0][1] = -lam
        D[dx-1][dx-1] = 1+2*lam
        D[dx-1][dx-2] = -lam     
        inv = Inverse(D)       
    for j in range(dt):
        d[0] = s[0]*lam+s0
        d[dx-1] = s[dx-1]*lam+sn
        res  = np.dot(inv,np.transpose(d))
        s = copy.deepcopy(res)
        plt.plot(np.arange(0,len(s)),s)
        plt.show()


def choleskyDecompose(A,B = None):
    """
    B[i][j] = ith row and jth column
    """
    m = len(A[0])
    
    L = np.zeros((m,m))
    #Create L row by row
    for i in range(m):
        for j in range(i+1):
            sum = 0
            for k in range(j):
                sum += L[i][k] * L[j][k]

            if (i == j):
                L[i][j] = np.sqrt(A[i][i] - sum)
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - sum))
    
    #Forward Backward substitution for Cholesky
    if(B!=None):
        n = len(B[0])
        #Forward Substitution
        for j in range(n):
            for i in range(m):
                sum = 0
                for k in range(i):
                    sum += L[i][k]*B[k][j]
                B[i][j] = (B[i][j] -  sum)/L[i][i]

        #Backward Substitution
        for j in range(n):
            for i in range(m-1,-1,-1):
                sum = 0
                for k in range(i+1,m):
                    sum += L[k][i]*B[k][j]
                B[i][j] = (B[i][j] - sum)/L[i][i]
            
        return L,B
    else: return L        

def jacobi(matrix ,b ,prec =1e-4):
    a=1
    aarr=[]
    karr=[]
    p=1
    X = []
    X1= []
    for i in range(len(matrix)):
        X.append(0)
        X1.append(0)

    while(a>prec):
       
        a = 0
        for l in range(len(X)):
            X[l] = X1[l]
                
        for i in range(len(matrix)):
            sum = 0
            for j in range(len(matrix)):
                if( i!=j):
                    
                    sum += matrix[i][j]*X[j] 
            
            X1[i] = (b[i]-sum)/matrix[i][i]
        for j in range(len(X)):
            a += (X1[j]-X[j])**2
            
        a = a**(1/2)
        aarr.append(a)
        karr.append(p)
        p += 1
        
    return X1
        



def gauss_sidel(matrix ,b ,prec=1e-4):
    a=1
    aarr=[]
    karr=[]
    p=1
    X = []
    X1= []
    for i in range(len(matrix)):
        X.append(0)
        X1.append(0)
    
    while(a>prec):
        a=0
        for l in range(len(X)):
            X1[l] = X[l]
                
        for i in range(len(matrix)):
            sum = 0
            for j in range(len(matrix)):
                if( i!=j):
                    sum += matrix[i][j]*X[j]         
            
            X[i] = (b[i]-sum)/matrix[i][i]
         
        for j in range(len(X)):
            a += (X1[j]-X[j])**2
        a = a**(1/2)
        aarr.append(a)
       
        karr.append(p)
        p += 1
        
    return X
        
        
def ConjGrad(A,b,x = None, tol = 1e-4, max_iter = 1000):
    n = len(A)
    if x is None: x = np.ones(n)
    r = b - np.dot(A,x)
    d = r
    count = 0
    while (np.dot(r,r)>tol and count<max_iter):
        rn = np.dot(r,r)
        a = (rn)/(np.dot(d,np.dot(A,d)))
        x += a*d
        r -= a*np.dot(A,d)

        b = np.dot(r,r)/rn
        d = r + b*d
        count += 1
    return x



def Inverse(matrix):
    I = np.identity(len(matrix))
    Inv = np.zeros((len(matrix),len(matrix)))
    for i in range(len(matrix)):
        Inv[:,i] = gauss_sidel(matrix, I[i])

    return Inv   


def polynomial_fit(x, y ,n):
    A = np.zeros((n,n))
    B = np.zeros(n)
    for i in range(len(A)):
        for j in range(len(A)):
            sum =0 
            for k in range(len(x)):
                sum  += x[k]**(i+j)
            A[i][j] = sum

    
    for i in range(len(B)):
        sum =0
        for j in range(len(x)):
            sum += x[j]**(i)*y[j]
        B[i] = sum
    X = gauss_sidel(A,B)
    fit = []
    for i in range(len(x)):
        fit.append(math.e**(X[0]+X[1]*x[i]))

    Ainv = Inverse(A)
    

    return X  , Ainv 

def gaussian_quadrature(fun, order, a, b):
    res = np.polynomial.legendre.leggauss(order)

    roots = res[0]
    print((roots))
    weights = res[1]

    sum = 0
    for i in range(len(roots)):
        y = ((b-a)*0.5*roots[i])+((b+a)*0.5)
        wfy = weights[i]*fun(y)
        sum = sum + wfy
    ans = (b-a)*0.5*sum

    return ans


def partial_pivot(matrix):
    j = 1
    for i in range(len(matrix)):
        if i != len(matrix) - 1 and j == 1:
            if matrix[i][i] == 0:
                for l in range(i, len(matrix)):
                    if matrix[l][i] != 0:
                        for k in range(len(matrix[0])):
                            matrix[l][k], matrix[i][k] = matrix[i][k], matrix[l][k]
                            j = 0

        elif j == 1:
            if matrix[i][i] == 0:
                for k in range(len(matrix) + 1):
                    matrix[i - 1][k], matrix[i][k] = matrix[i][k], matrix[i - 1][k]
                    j = 0

    return matrix

def gauss_jordan(A):
    for i in range(len(a)):
        a = partial_pivot(a)
        pivot = a[i][i]
        for j in range(len(a[0])):
            a[i][j] = a[i][j] / pivot
        for k in range(len(a)):
            if k != i:
                ratio = a[k][i]
                for j in range(len(a[0])):
                    a[k][j] = a[k][j] - ratio * a[i][j]

    return a    


# Eigenvalues Calculators

# Function which calculates the largest eigenvalue using power method
def PowerMethodCalc(A, x, tol = 1e-4):
    oldEVal = 0 # Dummy initial instance
    eVal = 2

    while abs(oldEVal-eVal)>tol:
        x = np.dot(A,x)
        eVal = max(abs(x))
        x = x/eVal

        oldEVal=eVal

    return eVal,x


# Wrapper function which allows us to get multiple eigenvalues
def EigPowerMethod(A, x=None, n=1, tol = 1e-4):
    if x is None: x = np.ones(len(A))
    eig = []
    eigvector = []
    E,V = PowerMethodCalc(A,x,tol)
    eigvector.append(V)
    eig.append(E)
    if(n>1):
        iter = n-1
        while iter > 0:
            V = V/np.linalg.norm(V)
            V = np.array([V])
            A = A - E*np.outer(V,V)
            E,V = PowerMethodCalc(A,x,tol)
            eig.append(E)
            eigvector.append(V)
            iter -= 1
    return eig ,eigvector

#Jacobi Method for eigenvalues (Given's Rotation)
def JacobiEig(A):
    n = len(A)
    # Find maximum off diagonal value in upper triangle
    def maxfind(A):
        Amax = 0
        for i in range(n-1):
            for j in range(i+1,n):
                if (abs(A[i][j]) >= Amax):
                    Amax = abs(A[i][j])
                    k = i
                    l = j
        return Amax,k,l

    def GivensRotate(A, tol = 1e-4, max_iter = 5000):
        max = 4
        iter = 1
        while (abs(max) >= tol and iter < max_iter):
            max,i,j = maxfind(A)
            if A[i][i] - A[j][j] == 0:
                theta = math.pi / 4
            else:
                theta = math.atan((2 * A[i][j]) / (A[i][i] - A[j][j])) / 2

            Q = np.eye(n)
            Q[i][i] = Q[j][j] = math.cos(theta)
            Q[i][j] = -1*math.sin(theta)
            Q[j][i] = math.sin(theta) 
            AQ = matrix_multiplication(A,Q)

            # Q inv = Q transpose
            Q = np.array(Q)
            QT = Q.T.tolist()

            A = matrix_multiplication(QT,AQ)
            iter += 1
        return A
    sol = GivensRotate(A)
    return np.diagonal(sol)


def DFT(x):
    """
    Function to calculate the 
    discrete Fourier Transform 
    of a 1D real-valued signal x
    """

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    X = np.dot(e, x)
    
    return X




#####################################################################################




def file_to_matrix(matrix_file):
    Matrix = []
    with open(matrix_file) as file:
        M_row = file.readlines()
    for line in M_row:
        Matrix.append(list(map(lambda i: int(i), line.split(" "))))
    return Matrix

def print_matrix(Matrix):
    for i in range(len(Matrix)):
        for j in range(len(Matrix[i])):
            print(Matrix[i][j], end=", ")
        print("")


def augment(matrix1, matrix2, forright=False):
    for i in range(len(matrix1)):
        if (type(matrix2[i]) == int or type(matrix2[i]) == float):
            if forright:
                matrix1[i].insert(0, matrix2[i])
            else:
                matrix1[i].append(matrix2[i])
        else:
            if forright:
                matrix1[i].insert(0, matrix2[i][0])
            else:
                matrix1[i].append(matrix2[i][0])
    return matrix1


def unaugment(matrix):
    vector = [ 0 for i in range(len(matrix))]
    for i in range(len(matrix)):
        vector[i] = matrix[i].pop(-1)
    return matrix, vector

def partial_pivot(matrix):
    j = 1
    for i in range(len(matrix)):
        if (i!= len(matrix)-1 and j ==1):
            if (matrix[i][i] == 0):
                for l in range(i,len(matrix)):
                    if(matrix[l][i] != 0):
                        for k in range(len(matrix[0])):
                            matrix[l][k] ,matrix[i][k] = matrix[i][k],matrix[l][k]
                            j = 0  
                    
        elif( j == 1):
            if (matrix[i][i] == 0):
                for k in range(len(matrix)+1):
                    matrix[i-1][k] ,matrix[i][k] = matrix[i][k],matrix[i-1][k]
                    j = 0 

def gauss_jordan(a):
    for i in range(len(a)):
        partial_pivot(a)
        pivot = a[i][i]
        for j in range(len(a[0])):
            a[i][j] = a[i][j]/pivot
        for k in range(len(a)):
            if (k!=i):
                ratio  = a[k][i]
                for j in range(len(a[0])):
                    a[k][j] = a[k][j] - ratio*a[i][j]
    return a

def boundary(A):
    """
    Sets up the boundary conditions

    Input:
     - A: Matrix to set boundaries on
     - x: Array where x[i] = hx*i, x[last_element] = Lx

    Output:
     - A is initialized in-place (when this method returns)
    """

    #Boundaries implemented (condensator with plates at y={0,Lx}, DeltaV = 200):
    # A(x,0)  =  1
    # A(x,Ly) = 0
    # A(0,y)  = 0
    # A(Lx,y) = 0

    Nx = A.shape[1]
    Ny = A.shape[0]


    A[Ny-1,:] = 1.0
    A[0,:]    = 0.0
    A[:,0]    =   0
    A[:,Nx-1] = 0
    
    return A


def Laplace2D(A, maxsteps, convergence):
    """
    Relaxes the matrix A until the sum of the absolute differences
    between the previous step and the next step (divided by the number of
    elements in A) is below convergence, or maxsteps is reached.

    Input:
     - A: matrix to relax
     - maxsteps, convergence: Convergence criterions

    Output:
     - A is relaxed when this method returns
    """

    iterations = 0
    diff = convergence +1

    Nx = A.shape[1]
    Ny = A.shape[0]
    
    while iterations < maxsteps and diff > convergence:
        #Loop over all *INNER* points and relax
        Atemp = A.copy()
        diff = 0.0
        
        for y in range(1,Ny-1):
            for x in range(1,Ny-1):
                A[y,x] = 0.25*(Atemp[y,x+1]+Atemp[y,x-1]+Atemp[y+1,x]+Atemp[y-1,x])
                diff  += math.fabs(A[y,x] - Atemp[y,x])

        diff /=(Nx*Ny)
        iterations += 1
        #print("Iteration #", iterations, ", diff =", diff)
    
    return A

def inverse_gd(a):
    for i in range(len(a)):
        vector = [0] * i + [1] + [0] * (len(a) - i)
        new_matrix = augment(a , vector )
    inverse = gauss_jordan(new_matrix)
    inverse_matrix = []
    for i in range(len(a)):
        inverse_matrix.append([])    

    for i in range(len(a)):
        matrix1 , matrix2 = unaugment(inverse)
        inverse_matrix = augment(inverse_matrix ,matrix2, True)
    return(inverse_matrix)

def matrix_multiplication( matrix1 , matrix2):
    M1_cross_M2 = []
    for i in range(len(matrix1)):
        M1_cross_M2.append([])
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            sum = 0
            for k in range(len(matrix2)):
                sum  = sum + (matrix1[i][k]*matrix2[k][j])
            M1_cross_M2[i].append(sum)  
    #print_matrix(matrix1)
    #print_matrix(matrix2)                    
    return M1_cross_M2

def LUdecomposition(A):
    for level in range(len(A)):  # diagonally which stage the algorithm is in
        for col in range(level, len(A)):  # the col for the Upper tri. matrix
            summation = 0
            for sum_term in range(0, level):  # the summation
                summation += A[sum_term][col] * A[level][sum_term]
            A[level][col] = A[level][col] - summation
        for row in range(level, len(A)):  # the row for the lower tri. matrix
            summation = 0
            for sum_term in range(0, level):  # the summation
                summation += A[sum_term][level] * A[row][sum_term]
            if row != level:
                A[row][level] = (A[row][level] - summation) / A[level][level]

def luDecompose(A, n):
    for i in range(n):
        # Upper Triangle Matrix (i is row index, j is column index, k is summation index)
        for j in range(i,n):
            # Summation part
            sum = 0
            for k in range(i):
                if(i==k):
                    sum += A[k][j]  # Since diagonal elements of Lower matrix is 1
                else:
                    sum += A[i][k]*A[k][j]
        
            A[i][j] = A[i][j] - sum
        
        # Lower Triangle Matrix (j is row index, i is column index, k is summation index)
        for j in range(i+1,n):
            # Summation part
            sum = 0
            for k in range(i):
                if(j==k):
                    sum += A[k][i]  # Since diagonal elements of Lower matrix is 1
                else:
                    sum += A[j][k]*A[k][i]
            A[j][i] = (A[j][i] - sum)/A[i][i]

    return A        


def decomposed_solver(matrix):
    n = len(matrix)
    # solving Ly =B
    y = [[0]for i in range(n)]
    k=0
    for i in range(n) :
        sum =0
        if(i>0):
            for j in range(k+1):
                sum += matrix[i][j]*y[j][0]    
        y[i][0] = matrix[i][n] - sum
        k= k+1
    #solving Ux = y
    x = [[0]for i in range(n)]
    k=0
    for i in reversed(range(n)) :
        sum =0
        if (i<n-1):
            for j in range(k+1):
                sum += matrix[i][n-j-1]*x[n-j-1][0]
        x[i][0] = (y[i][0]- sum)/matrix[i][i]
        k= k+1

    return x


def inverse_usingLU(matrix):
    n = len(matrix)
    for i in range(n):
        vector = [0] * i + [1] + [0] * (n- i-1)
        whole_matrix = augment(matrix , vector )   
    for i in range(n):       
        partial_pivot(matrix)

    pivot_i =[]
    for i in reversed(range(n)):
        matrix1 , vec = unaugment(whole_matrix )
        if (i == 0):
            pivot_i = vec
        else:
            pivot_i = augment(pivot_i ,vec) 
    new_matrix = matrix1.copy()
    for i in range(n):
        vector = [0] * i + [1] + [0] * (n- i-1)
        whole_matrix = augment(matrix , vector )   
    for i in range(n):       
        partial_pivot(matrix)

    solution =[]
    for i in reversed(range(n)):
        print_matrix(whole_matrix)
        whole_matrix , vec = unaugment(whole_matrix )
        combined_matrix = augment(new_matrix,vec)
        if (i == 0):
            solution = decomposed_solver(combined_matrix)
        else:
            solution = augment(solution  , list (map(lambda i:i[0] , decomposed_solver(combined_matrix))))

    print_matrix(solution)

def fun_prime(f, h=10 ** -4):

    """
    Returns a function for numerical derivative of f
    """

    def df_dx(x, h=h):
        return (f(x + h) - f(x - h)) / (2 * h)

    return df_dx

def derivative(f , x):
    return (f(x+0.000001)-f(x-0.000001))/0.000002

def bracketing(function ,a, b):
    i =0
    while(function(a)*function(b) > 0 and i<15):
        eta = 1.5
        if (function(a) >0):
            a = a-eta*(b-a)
        else:
            b = b+eta*(b-a)
        i+=1
    return bisection_method(function , a,b) , false_method(function ,a,b)

def bisection_method(function ,a,b):
    iteration , error = [] ,[]
    i=0
    c=0
    while(i<50 and abs(a-b) > 0.000001 ):
        l=c
        c = (a+b)/2
        if (function(a)*function(c)==0):
            return c
        elif (function(a)*function(c)<0):
            b=c
        else:
            a=c
        error.append(abs(l-c))
        iteration.append(i)
        i+=1  
    #plt.title('Bisection_method')
    #plt.xlabel('iteration number')
    #plt.ylabel('absolute error')     
    #plt.plot(iteration,error, marker = '.')
    #plt.show()
    return a   

def RK4_2nd_order(f, x_0, y_0, z_0, x_max, h):
    x = x_0
    y = y_0
    z = z_0
    X, Y = [x_0], [y_0]
    while x < x_max:
        k1_y = h*z
        k1_z = h*f(x,y,z)
        k2_y = h*(z+(k1_z/2))
        k2_z = h*f(x+(h/2), y+(k1_y/2), z+(k1_z/2))
        k3_y = h*(z+(k2_z/2))
        k3_z = h*f(x+(h/2), y+(k2_y/2), z+(k2_z/2))
        k4_y = h*(z+k3_z)
        k4_z = h*f(x+h, y+k3_y, z+k3_z)
        y += (k1_y + 2*k2_y + 2*k3_y + k4_y)/6
        z += (k1_z + 2*k2_z + 2*k3_z + k4_z)/6
        x += h
        X.append(x)
        Y.append(y)
    return X, Y


def ODE_Shooting(f, a, b, y_at_a, y_at_b):
    tolerance = 1e-4
    h = 0.01
    z_at_a = 2
    X, Y = RK4_2nd_order(f, a, y_at_a, z_at_a, b, h)
    if abs(Y[-1]-y_at_b) <= tolerance:
        return X, Y
    else:
        z_old = z_at_a
        y_old = Y[-1]
        z_at_a = float(input(f"Enter some guess slope: "))
        X, Y = RK4_2nd_order(f, a, y_at_a, z_at_a, b, h)
        if (y_old > y_at_b and Y[-1] < y_at_b) or (y_old < y_at_b and Y[-1] > y_at_b):
            z_at_a = z_at_a + (z_old-z_at_a)*(y_at_b-Y[-1])/(y_old-Y[-1])
            X, Y = RK4_2nd_order(f, a, y_at_a, z_at_a, b, h)
        return X, Y
    
def normalize(Y):
    sum = 0
    for y in Y:
      sum += y**2
    sum = math.sqrt(sum)
    for i in range(len(Y)):
      Y[i] = Y[i]/sum
    return Y


def false_method (function ,a,b):
    iteration , error = [] ,[]
    i=0
    c=0
    while(i<50 and abs(a-b) > 0.000001):
        l=c
        c = b-((b-a)*function(b)/(function(b)-function(a)))
        if(function(a)*function(c)==0):
            return c
        elif(function(a)*function(c)<0):
            b=c
        else:
            a=c
        error.append(abs(l-c))
        iteration.append(i)
        i+=1   
    #plt.title('Falsi_method')
    #plt.xlabel('iteration number')
    #plt.ylabel('absolute error')    
    #plt.plot(iteration,error , marker = '.')
    #plt.show()        
    return c   

def newton_raphson(function,x):
    i=1
    x0 = x
    x1 = 0
    iteration , error = [] ,[]
    while(i<50 and abs(x1-x0) > 0.000001):
        x1=x0
        x0 = x0-(function(x0)/derivative(function,x0))
        error.append(abs(x1-x0))
        iteration.append(i)
        i+=1
    #plt.title('Newton Raphson')
    #plt.xlabel('iteration number')
    #plt.ylabel('absolute error')    
    #plt.plot(iteration,error,marker = '.')
    #plt.show()      
    return x0    

def polynomial_generator(a):
    return lambda x: sum(a_i * pow(x, i) for i, a_i in enumerate(a))


def synthetic_division(coefficients, divisor):
    coefficients = coefficients[::-1]
    quotient = []
    quotient.append(coefficients[0])
    for i in range(1, len(coefficients)):
        quotient.append(coefficients[i] + (divisor * quotient[-1]))
    return quotient[0:-1][::-1]


def laguerre(f, order, guess, ϵ=10 ** -4):
    df_dx = fun_prime(f)
    d2f_dx2 = fun_prime(df_dx )

    if (f(guess) == 0):
        return guess
    else:    
        guess_old = guess + 1  # just to start
        while abs(guess_old - guess) > ϵ:
            G = df_dx(guess) / f(guess)
            H = G * G - (d2f_dx2(guess) / f(guess))
            last_term = ((order - 1) * (order * H - G)) ** (1 / 2)
            sum = G + last_term
            diff = G - last_term
            a = order / (
                sum * (abs(sum) > abs(diff))+ diff * (abs(sum) < abs(diff))
            )  # this is less readable but its faster than an if else
            guess_old = guess
            guess -= a
        return guess


def roots_from_laguerre(coefficients):
    ϵ=10 ** -4 
    guess=1.0
    order = len(coefficients) - 1
    roots = []
    for i in range(order):
        f = polynomial_generator(coefficients)
        root = laguerre(f, order, guess, ϵ)
        coefficients = synthetic_division(coefficients, root)
        roots.append(root)
    return roots    


from random import gauss, random



def midpoint(f,a,b,N):
    h = (b-a)/N
    sum =0
    for i in range (N):
        x = (2*a + (2*i+1)*h)/2
        sum += f(x)*h
    return sum  

def trapezoidal(f,a,b,N):
    h = (b-a)/N
    sum =0
    for i in range (1,N):
        x = a+i*h
        sum += f(x)*h
    sum += (f(a)+f(b))*h/2   
    return sum

def simpson(f,a,b,N):
    h = (b-a)/N
    sum =0
    for i in range (1,N):
        x = a+i*h
        weight = 4 if i % 2 else 2
        sum += f(x)*weight
    sum += (f(a)+f(b))
    sum *= h/3
    return sum

def random_no_gen(x0 ,a,m,N):
    #seed
    x1=[x0]
    #generating random numbers
    for i in range(N):
        x1.append(a*x1[-1] % m)
    x1 = [xs/m for xs in x1]    
    return x1

def Randomwalk(N,a,m,x0):
    # setting x and y at origin
    x = y = 0
    xvalue = [0]
    yvalue = [0]
    randno = random_no_gen(x0,a,m,N )
    for i in range(N):
        # generating random angles and moving one step length in that direction
        theta = 0 + 2 * math.pi * randno[i]
        x += math.cos(theta)
        y += math.sin(theta)
        # Saving x,y values of each step
        xvalue.append(x)
        yvalue.append(y)
    # Calculating radial distance
    radialdistance = (x ** 2 + y ** 2) ** 0.5
    return xvalue, yvalue,radialdistance

def RandomWalks(N):
    """
    Random walker moves in 1D.
    """
    x = [0] #Array to keep track of pos (starts from centre)
    rando = random_no_gen(19,35,1021,N )  #Array of random numbers
    for i in range(N):
        if(rando[i]>=0.5): x.append(x[-1]-1) #Go left
        else: x.append(x[-1]+1)
    return x


def montecarlo(f,a,b,N,x0,arand,m):
    h = (b-a)/N
    sum =0
    for i in range (N):
        x1 = random_no_gen(x0,arand,m)
        sum += f(a + (b-a)*x1)*h
        x0=x1
    return sum          

def montecarlo2(f,a,b,N):
    h = (b-a)/N
    sum =0
    for i in range (N):
        x1 = random()
        sum += f(a + (b-a)*x1)*h
        x0=x1
    return sum 


def monteCarloIntegralImportanceSampling(f, p, y, lowerBound, upperBound, N, fileName = None):
    
    
    integral = 0
    width = upperBound - lowerBound
    h = width/N
    

    for i in range(N):
        x = random()
        integral += (f(lowerBound + (width*y(x))))/(p(y(x)))
        #integral += f(lowerBound + (width*random()))
    integral *= h
    
    return integral

def expliciteuler(n , h ,f , x1 ,y0 ):
    #adding first elements
    y=[y0]
    x=[x1]
    y1=y0
    x0=x1
    # calculating different points
    for i in range(n):
        y1 += h*f(y1,x0)
        x0 += h
        y.append(y1)  #appending reslts in array to plot
        x.append(x0)

    return x,y


def rk4(n, h ,f , x1 ,y0):
    """
    n: no of steps
    h: step length
    f: dy/dx
    x1, y0: initial conditions
    """
    #adding first elements
    y=[y0]
    #z=[z0]
    x=[x1]
    y1=y0
    #z1=z0
    x0=x1
    #defining k1,k2,k3,k4 for calculating points using simpson fromula 
    for i in range(n):
        k1 = h*f(y1,x0)
        k2 = h*f(y1+(k1/2),x0+(h/2))
        k3 = h*f(y1+(k2/2),x0+(h/2))
        k4 = h*f(y1+k3,x0+h)
        #l1 = h*g(y1,x0,z1)
        #l2 = h*g(y1+(l1/2),x0+(h/2),z1+(l1/2))
        #l3 = h*g(y1+(l2/2),x0+(h/2),z1+(l2/2))
        #l4 = h*g(y1+l3,x0+h,z1+l3)
        y1 += (k1+2*k2+2*k3+k4)/6
        #z1 += (l1+2*l2+2*l3+l4)/6

        x0 += h
        y.append(y1) #appending reslts in array to plot
        #z.append(z1)
        x.append(x0)
    
    # loop for running backwards 
    
    
    return x,y

def shooting_method(n,
    dz_dx, BV1, BV2,dx, method=rk4, limit=10
):

    count = 0
    guess1 = [BV1[0], 1]
    guess2 = [BV1[0], -1]
    if BV2[1] != 0:
        y1 = [0]
    else:
        y1 = [1]
    x1 = [0]
    while (
        abs(y1[-1] - BV2[1]) >= 10 ** -13
        and count < limit
    ):
        if count == 0:
            guess = guess1.copy()
        elif count == 1:
            guess1.append(y1[-1])
            guess = guess2.copy()
        else:
            if count == 2:
                guess2.append(y1[-1])
            else:
                guess1[2] = y1[-1]
            # generating new guess
            guess = guess1[1] + (guess2[1] - guess1[1]) * (BV2[1] - guess1[2]) / (guess2[2] - guess1[2])
            guess1[1] = guess
            guess = guess1
        # using rk4 to calculate
        x1, z1 = method(int(2*n+1),
            dx/2,
            dz_dx,
            guess[0], guess[1],
        )
        x1 = list(map(lambda x: round(x, 6), x1))

        def dy_dx(y, x):
            return z1[x1.index(round(x, 6))]
        x1, y1 = method(int(n),
            dx,
            dy_dx,
            BV1[0],BV1[1])
        count += 1
    return x1, y1
