import numpy as np
import pandas as pd
import random
from sklearn import svm
# from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def classification_err(conf_matrix, test_num):
    n = len(conf_matrix)

    correct=0
    for i in range(n):
        correct+=conf_matrix[i][i]

    wrong=test_num-correct

    print(f"Eroarea de clasificare: {wrong/test_num*100:.2f}%")


def confusion_matrix(Y_test,Y_pred,class_num):
    mat = np.zeros((class_num,class_num), dtype=int)
    
    for real, prezis in zip(Y_test,Y_pred):
        mat[int(real-1)][int(prezis-1)] += 1
        
    return mat

    

def main():
    df = pd.read_csv('iris.csv', header=None)
    
    labels={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
    X =df.iloc[:,0:4].values 
    y = df.iloc[:,4].values
    y=np.array([labels[yy] for yy in y])

    df=np.column_stack((X,y))
    # random.shuffle(df)
    np.random.shuffle(df)


    X=df[:,0:4]
    y=df[:,4]    

    X_train=X[:100]
    X_test=X[100:]
    Y_train=y[:100]
    Y_test=y[100:]

    # Observatii:

    # pentru setul asta de date pare ca functioneaza mai bine PCA, in sensul in care 
    # eroarea de clasificare sta constant la 2% pentru ambele tipuri de kernel.
    # fara scalare si pca, eroarile de clasificare sar de la 0% la 6%, pentru oricare 
    # din cele 2 kerneluri

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=3)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    print("\nLinear\n")

    clsf=svm.SVC(kernel='linear')
    clsf.fit(X_train,Y_train)

    y_pred = clsf.predict(X_test)

    conf_matrix=confusion_matrix(Y_test,y_pred,3)
    print(conf_matrix)

    classification_err(conf_matrix,len(Y_test))

    print("\nRBF\n")

    clsf=svm.SVC(kernel='rbf')
    clsf.fit(X_train,Y_train)

    y_pred = clsf.predict(X_test)

    conf_matrix=confusion_matrix(Y_test,y_pred,3)
    print(conf_matrix)

    classification_err(conf_matrix,len(Y_test))



if __name__=="__main__":
    main()