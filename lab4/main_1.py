import numpy as np


def main():
    m, r = 10, 4
    A = np.random.randn(m,r)
    rank=np.linalg.matrix_rank(A)
    
    # print(A)
    print(f"a) Rangul matricii: {rank}")

    col1=A[:,:1]
    col2=A[:,1:2]
    col3=A[:,2:3]

    new_col=2*col1-2/3*col2+10*col3
    A_extins=np.hstack((A,new_col))
    new_col=-2*col1-2/3*col2-10*col3
    A_extins=np.hstack((A_extins,new_col))
    new_col=20*col1+3*col2+2*col3
    A_extins=np.hstack((A_extins,new_col))
    new_col=5*col1-3*col2-9*col3
    A_extins=np.hstack((A_extins,new_col))

    rank_extins=np.linalg.matrix_rank(A_extins)
    print(f"b) Rang extins: {rank_extins}")

    medie = 0
    dispersie = 0.2
    zgomot = np.random.normal(medie,dispersie,size=A_extins.shape)

    A_zgomot=A_extins+zgomot
    rank_zogomot = np.linalg.matrix_rank(A_zgomot)
    print(f"Rang matrice cu zgomot: {rank_zogomot}")

    U,S,V=np.linalg.svd(A_zgomot)
    print(f"Valorile singulare ale matricii extinse cu zgomot:\n {S}")

    # Mie imi afiseaza 4 valori singulare >=1 si celelalte 4 mai mici. Presupun ca functioneaza 
    # sa pun trashhol ul la 1, dar presupun cafunctioneaza pentru ca am facut combinatii liniare 
    # cu coeficienti mari

if __name__=="__main__":
    main() 