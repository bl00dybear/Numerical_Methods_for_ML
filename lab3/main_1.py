import numpy as np

def norma_2(x):
    x=np.asarray(x,dtype=float)
    return np.sqrt(np.sum(x * x))

def main():
    n=6
    tol=0.00001
    max_iter=1000
    A=np.ones((n, n), dtype=float)

    y=[1]*n
    iter=0
    err=1
    while err>tol:
        if iter>max_iter:
            break

        y = y/norma_2(y)
        z=np.dot(A,y)
        z=z/norma_2(z)
        err =np.abs(1-np.abs(np.dot(z.T,y)))
        y=z
        iter+=1
        # print(y)

    lambda_1 = np.dot(y.T, np.dot(A, y)) / np.dot(y.T, y)

    print("Valoare proprie:", lambda_1)
    print("Vector propriu:", y)

    vals, vecs = np.linalg.eig(A)
    index = np.argmax(np.abs(vals))
    lambda_2 = vals[index]
    y_2 = vecs[:, index]

    print("\nValoare proprie numpy:", lambda_2)
    print("Vector propriu numpy:", y_2)



if __name__=="__main__":
    main()