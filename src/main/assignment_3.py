import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)
def function(t,y):
    return t- y**2
initial_value= 1
start_point=0
end_point=2
i=10
x=(end_point-start_point)/i
def eulers(initial_value,x,i,start_point):
    j=0
    for i in range(i):
        initial_value= initial_value + (x * function(start_point,initial_value))
        start_point+= x 
        j+=1
        if (j==10):
            print(initial_value)
eulers(initial_value,x,i,start_point)
print()
def rk(x,i,initial_value,start_point):
    for i in range(i):
        k1= x * function(start_point,initial_value)
        k2= x * function(start_point+x/2, initial_value +k1/2)
        k3= x * function(start_point+x/2, initial_value +k2/2)
        k4= x * function(start_point+x, initial_value+k3)
        initial_value+= (k1+ 2*k2+2*k3+k4)/6
        start_point+= x 
    print (initial_value)
rk(x,i,initial_value,start_point)
print()
def GE(A, b):
    n = len(b)
    Ab = np.concatenate((A, b.reshape(n,1)), axis=1)
    x= np.zeros(n)
    for i in range(n):
        pivot= np.argmax(np.abs(Ab[i:,i]))+ i
        Ab[[i, pivot]] = Ab[[pivot,i]]
        for j in range(i+1,n):
            Ab[j] = Ab[j]- ((Ab[j,i] /Ab[i,i]) * Ab[i])
    for i in range(n-1,-1,-1):
        x[i] =(Ab[i,-1] - np.dot(Ab[i,:-1], x)) / Ab[i,i]
    return x 
A = np.array([[2, -1, 1], [1, 3, 1], [-1, 5, 4]])
b = np.array([6, 0, -3])
print(GE(A, b))
print()
def LU(B):
    n = B.shape[0]
    L = np.zeros((n,n))
    U = np.zeros((n, n))
    determiant = np.linalg.det(B)
    for k in range(n):
        L[k, k] = 1
        for i in range(k,n):
            U[k, i] = B[k, i]- np.dot(L[k, :k], U[:k, i])
        for j in range(k+1,n):
            L[j, k] =(B[j, k] - np.dot(L[j, :k], U[:k, k]))/ U[k, k]
    print(determiant)
    print()
    print(L)
    print()
    print(U)
B = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
LU(B)
print()
def dominant(C):
    n = len(C)
    total=0
    for i in range(n):
        total = np.sum(C[i])- C[i][i]
        if (C[i][i]) < total:
            return False
    return True
C=np.array([[9, 0, 5, 2, 1],[3, 9, 1, 2, 1],[0, 1, 7, 2, 3],[4, 2, 3, 12, 2],[3, 2, 4, 0, 8]])
print(dominant(C))
print()
def definite(D):
    ev= np.linalg.eigvals(D)
    n = len(ev)
    checker = 0
    for i in range(n):
        if ev[i]>0:
            checker+=1   
    if checker>0:
        print("True")
    else:
        print("False")
D = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
definite(D)