import numpy as np

def UTRIS(U,b,n):
    
    for i in range (n-1,-1,-1):
        for j in range (i+1,n):
            b[i] -= (U[i][j]*b[j])
        b[i] /= U[i][i]
            
    return b


def GAXPYtrI(A,x,n,y=None):
    r = np.zeros(len(x))
        
    for i in range (0,n):
        for j in range (0,i+1):
            if j == i:
                r[i] += x[j]
            else:
                r[i] += A[i][j]*x[j]
        if y is not None:
            r[i] += y[i]
            
    return r

def main():
    n = 3
    # A = np.random.randn(n,n)

    A = [
        [2,4,-2],
        [4,9,-3],
        [-2,-3,7]
    ]   
    b = [2,8,10]

    
    for k in range(0,n-1):
        for i in range(k+1,n):
            A[i][k] = -A[i][k]/A[k][k]   
            b[i] += A[i][k]*b[k]

            for j in range(k+1,n):
                A[i][j] = A[i][j] + A[k][j]*A[i][k]
                
    # -1 2 2 
                
    print(UTRIS(A,b,n))


if __name__ == "__main__":
    main()