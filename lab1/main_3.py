import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    dataframe = pd.read_csv('regresie.csv', header=None)
    
    first_col=dataframe.iloc[:, 0].tolist()
    A = [[x,1] for x in first_col]
    b = dataframe.iloc[:,1].tolist()

    coeffs,_,_,_=np.linalg.lstsq(A,b)

    x,y=coeffs

    xmin = min(first_col) - 0.2
    xmax = max(first_col) + 0.2

    X=np.linspace(xmin,xmax,2)
    Y=x*X+y
    
    plt.plot(X,Y)
    plt.plot(first_col, b, 'o')
    plt.show()
    

if __name__ == "__main__":
    main()