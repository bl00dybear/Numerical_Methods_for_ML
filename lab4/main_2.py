import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    A = np.array(Image.open("Baboon.bmp"))

    U,S,V=np.linalg.svd(A,full_matrices=False)

    plt.figure(figsize=(10,8))

    plt.subplot(2,2,1)
    plt.imshow(A,cmap='gray')
    plt.title(f"Rang {len(S)}")
    plt.axis('off')

    k = 256
    for i in range(3):
        k_int=int(k) 
        U_k=U[:,:k_int]
        S_k=np.diag(S[:k_int])
        Vt_k=V[:k_int,:]
        
        A_approx=np.dot(U_k,np.dot(S_k,Vt_k))
        
        plt.subplot(2,2,i+2)
        plt.imshow(A_approx,cmap='gray')
        plt.title(f"Rang k={k_int}")
        plt.axis('off')

        k=k//2 

    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    main()