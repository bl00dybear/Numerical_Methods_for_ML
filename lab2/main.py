import numpy as np

def UTRIS(U,b,n):
    
    for i in range (n-1,-1,-1):
        for j in range (i+1,n):
            b[i] -= (U[i][j]*b[j])
        b[i] /= U[i][i]
            
    return b


def main():
    n = 6
    A = np.random.randn(n,n)
    
    b = np.dot(A,[2,-6,1,5.5,1,1])
    
    # print(A)

    # A = [
    #     [2,4,-2],
    #     [4,9,-3],
    #     [-2,-3,7]
    # ]   
    # b = [2,8,10]
    # x = [-1 2 2] 

    
    for k in range(0,n-1):
        for i in range(k+1,n):
            A[i][k] = -A[i][k]/A[k][k]   
            b[i] += A[i][k]*b[k]

            for j in range(k+1,n):
                A[i][j] = A[i][j] + A[k][j]*A[i][k]
                
                
    print(UTRIS(A,b,n))


if __name__ == "__main__":
    main()