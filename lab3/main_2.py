import pickle
import numpy as np
import networkx
import matplotlib.pyplot as plt

def is_complete(v,n):
    index = np.argmax(np.abs(v))
    if abs(v[index]-n+1) > 0.001:
        return False
    
    for i in range(n):
        if i != index:
            if abs(v[i]+1) > 0.001:
                return False
            
    return True


def is_bipartit(v,n):
    index_0=0
    v=sorted(v)
    for i in range(n):
        if v[i] == 0:
            index_0=i

    for i in range(index_0):
        if v[i]!=v[n-1-i]:
            return False
        
    return True

def largest_clique(v):
    return int(np.abs(v[np.argmax(np.abs(v))]))


def plot_and_save(G, filename):
    pos = networkx.circular_layout(G)
    plt.figure(figsize=(5, 5))
    networkx.draw(G, pos,with_labels=True)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    with open("grafuri.pickle", "rb") as f:
        x1,x2,x3 = pickle.load(f)

    n1,n2,n3=len(x1),len(x2),len(x3)

    net1=networkx.convert_matrix.from_numpy_array(x1)
    net2=networkx.convert_matrix.from_numpy_array(x2)
    net3=networkx.convert_matrix.from_numpy_array(x3)

    plot_and_save(net1, "graf1.pdf")
    plot_and_save(net2, "graf2.pdf")
    plot_and_save(net3, "graf3.pdf")

    vals1, _ = np.linalg.eig(x1)
    vals2, _ = np.linalg.eig(x2)
    vals3, _ = np.linalg.eig(x3)

    print(f"Graf 1 complet: {is_complete(vals1,n1)}")
    print(f"Graf 2 complet: {is_complete(vals2,n2)}")
    print(f"Graf 3 complet: {is_complete(vals3,n3)}")

    print(f"\nGraf 1 bipartit: {is_bipartit(vals1,n1)}")
    print(f"Graf 2 bipartit: {is_bipartit(vals2,n2)}")
    print(f"Graf 3 bipartit: {is_bipartit(vals3,n3)}")

    print(f"\nCea mai mare clica graf 1: {len(networkx.algorithms.approximation.clique.max_clique(net1))}")
    print(f"Cea mai mare clica graf 2: {len(networkx.algorithms.approximation.clique.max_clique(net2))}")
    print(f"Cea mai mare clica graf 3: {len(networkx.algorithms.approximation.clique.max_clique(net3))}")


if __name__=="__main__":
    main()