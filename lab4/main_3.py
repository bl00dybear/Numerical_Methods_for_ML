from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('iris.csv', header=None)

    X =df.iloc[:,0:4].values 
    y = df.iloc[:,4].values  

    comp = 2
    pca = PCA(n_components=comp)
    components = pca.fit_transform(X)

    print(components)

    df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    df_pca['target'] = y

    plt.figure(figsize=(10, 6))
    targets = df_pca['target'].unique()
    colors = ['r','g','b'] 

    for target, color in zip(targets, colors):
        indices = df_pca['target'] == target
        plt.scatter(df_pca.loc[indices, 'PC1'],df_pca.loc[indices, 'PC2'],c=color,s=50)

    plt.xlabel('Componenta Principală 1')
    plt.ylabel('Componenta Principală 2')
    plt.title('Proiecția PCA a setului de date')
    plt.legend(targets)
    plt.grid()
    plt.show()

if __name__=="__main__":
    main()