from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def MFE_Sequence(dt,N,istart,Re=800,Lx=4*np.pi,Lz=2*np.pi,Estart=0.01,a0=[1,0,0.187387,0.040112,0.047047,0,0,0.013188,0]):
#-----------------------------------
# Moehlis-Faisst-Eckhardt model
#-----------------------------------

    a1,a2,a3 = np.zeros(N), np.zeros(N), np.zeros(N)
    a4,a5,a6 = np.zeros(N), np.zeros(N), np.zeros(N)
    a7,a8,a9 = np.zeros(N), np.zeros(N), np.zeros(N)
    
    a1[0] = a0[0];          # base flow
    a2[0] = a0[1];
    a3[0] = a0[2];   # streamwise vortex
    a4[0] = a0[3];   # spanwise flow
    a5[0] = a0[4];   # spanwise flow
    a6[0] = a0[5];
    a7[0] = a0[6];
    a8[0] = a0[7];   # fully three-dimensional mode
    a9[0] = a0[8];
    '''
    a_value=np.sqrt(Estart-1)/2
    a2[0] = a_value;          # base flow
    a3[0] = a_value;   # streamwise vortex
    a4[0] = a_value;   # spanwise flow
    a5[0] = a_value;   # spanwise flow
    #a8[0] = 0.013188;   # fully three-dimensional mode
    '''
    a,b,c = 2*np.pi/Lx, np.pi/2, 2*np.pi/Lz
    kabc,kac,kbc = np.sqrt(a*a+b*b+c*c),np.sqrt(a*a+c*c),np.sqrt(b*b+c*c)

    # coefficients for first equation
    c1_1,c1_2,c1_3 =  b*b/Re, -np.sqrt(3/2)*b*c/kabc, np.sqrt(3/2)*b*c/kbc

    # coefficients for second equation
    c2_1,c2_2 = -(4*b*b/3+c*c)/Re, 5/3*np.sqrt(2/3)*c*c/kac
    c2_3,c2_4 = -c*c/np.sqrt(6)/kac, -a*b*c/np.sqrt(6)/kac/kabc
    c2_5      = -np.sqrt(3/2)*b*c/kbc

    # coefficients for third equation
    c3_1,c3_2 = -(b*b+c*c)/Re, 2/np.sqrt(6)*a*b*c/kac/kbc
    c3_3      =  (b*b*(3*a*a+c*c) - 3*c*c*(a*a+c*c))/np.sqrt(6)/kac/kbc/kabc
    
    # coefficients for fourth equation
    c4_1,c4_2,c4_3 = -(3*a*a+4*b*b)/3/Re, -a/np.sqrt(6), -10/3/np.sqrt(6)*a*a/kac
    c4_4,c4_5 = -np.sqrt(3/2)*a*b*c/kac/kbc, -np.sqrt(3/2)*a*a*b*b/kac/kbc/kabc
    c4_6 = -a/np.sqrt(6)
    
    # coefficients for fifth equation
    c5_1,c5_2,c5_3 = -(a*a+b*b)/Re, a/np.sqrt(6), a*a/np.sqrt(6)/kac
    c5_4,c5_5      = -a*b*c/np.sqrt(6)/kac/kabc, a/np.sqrt(6)
    c5_6           =  2/np.sqrt(6)*a*b*c/kac/kbc

    # coefficients for sixth equation
    c6_1,c6_2,c6_3 = -(3*a*a+4*b*b+3*c*c)/3/Re, a/np.sqrt(6), np.sqrt(3/2)*b*c/kabc
    c6_4,c6_5 =  10/3*(a*a-c*c)/np.sqrt(6)/kac, -2*np.sqrt(2/3)*a*b*c/kac/kbc
    c6_6,c6_7 =  a/np.sqrt(6), np.sqrt(3/2)*b*c/kabc
    
    # coefficients for seventh equation
    c7_1,c7_2 = -(a*a+b*b+c*c)/Re, -a/np.sqrt(6)
    c7_3,c7_4 =  (c*c-a*a)/np.sqrt(6)/kac, a*b*c/np.sqrt(6)/kac/kbc

    # coefficients for eighth equation
    c8_1,c8_2 = -(a*a+b*b+c*c)/Re, 2/np.sqrt(6)*a*b*c/kac/kabc
    c8_3      =  c*c*(3*a*a-b*b+3*c*c)/np.sqrt(6)/kac/kbc/kabc

    # coefficients for ninth equation
    c9_1,c9_2,c9_3 = -9*b*b/Re, np.sqrt(3/2)*b*c/kbc, -np.sqrt(3/2)*b*c/kabc
    printed=False
    for i in range(N-1):
        #if (i%1000000==0 and i>0): print('iteration = ',i) 
        L1 = 1 + dt*c1_1
        N1 = c1_2*a6[i]*a8[i] + c1_3*a2[i]*a3[i]
        
        L2 = 1 - dt*c2_1
        N2 = c2_2*a4[i]*a6[i] + c2_3*a5[i]*a7[i] + c2_4*a5[i]*a8[i] + \
            c2_5*a1[i]*a3[i] + c2_5*a3[i]*a9[i]

        L3 = 1 - dt*c3_1
        N3 = c3_2*(a4[i]*a7[i] + a5[i]*a6[i]) + c3_3*a4[i]*a8[i]
        
        L4 = 1 - dt*c4_1
        N4 = c4_2*a1[i]*a5[i] + c4_3*a2[i]*a6[i] + c4_4*a3[i]*a7[i] + \
            c4_5*a3[i]*a8[i] + c4_6*a5[i]*a9[i]
        
        L5 = 1 - dt*c5_1
        N5 = c5_2*a1[i]*a4[i] + c5_3*a2[i]*a7[i] + c5_4*a2[i]*a8[i] + \
            c5_5*a4[i]*a9[i] + c5_6*a3[i]*a6[i]
        
        L6 = 1 - dt*c6_1
        N6 = c6_2*a1[i]*a7[i] + c6_3*a1[i]*a8[i] + c6_4*a2[i]*a4[i] + \
            c6_5*a3[i]*a5[i] + c6_6*a7[i]*a9[i] + c6_7*a8[i]*a9[i]
        
        L7 = 1 - dt*c7_1
        N7 = c7_2*(a1[i]*a6[i] + a6[i]*a9[i]) + c7_3*a2[i]*a5[i] + c7_4*a3[i]*a4[i]
        
        L8 = 1 - dt*c8_1
        N8 = c8_2*a2[i]*a5[i] + c8_3*a3[i]*a4[i]
        
        L9 = 1 - dt*c9_1
        N9 = c9_2*a2[i]*a3[i] + c9_3*a6[i]*a8[i]
        
        a1[i+1] = (a1[i] + dt*N1 + dt*c1_1)/L1
        a2[i+1] = (a2[i] + dt*N2)/L2
        a3[i+1] = (a3[i] + dt*N3)/L3
        a4[i+1] = (a4[i] + dt*N4)/L4
        a5[i+1] = (a5[i] + dt*N5)/L5
        a6[i+1] = (a6[i] + dt*N6)/L6
        a7[i+1] = (a7[i] + dt*N7)/L7
        a8[i+1] = (a8[i] + dt*N8)/L8
        a9[i+1] = (a9[i] + dt*N9)/L9
        
    X   = np.vstack((a1,a2,a3,a4,a5,a6,a7,a8,a9)).T 
    pi4 = np.pi*np.pi/4
    D1  = pi4
    D2  = pi4 + c*c
    D3  = np.pi*np.pi + 2*c*c
    D4  = a*a + pi4
    D5  = a*a + pi4
    D6  = 2*(a*a + pi4 + c*c)
    D7  = 2*(a*a + pi4 + c*c)
    D8  = 3*(a*a + pi4 + c*c)
    D9  = 9*pi4
    DD  = np.array([D1,D2,D3,D4,D5,D6,D7,D8,D9])

    X      = X[istart:,:]
    D      = (X*X)@DD
    E      = np.sum(X*X,axis=1) 

    return D,E,X

if __name__=='__main__':
    
    dt,N = 0.05,310000         # time-step and number of trajectory points
    print('generating sequence') 
    D,E,X = MFE_Sequence(dt,N)   # generate data sequence 
    nclust = 10                # number of clusters ('boxes')  
    
    plt.figure(1)             # visualization 
    plt.ion()
    plt.clf()
    plt.plot(E,D)
    plt.gca().set_aspect(0.1)
    plt.show()

    DE = np.vstack((D,E)).T
    print('clustering in phase space') 
    km = KMeans(n_clusters=nclust, init='k-means++').fit(DE)  # 'discretize' into clusters 

    C  = km.labels_                     # cluster membership 
    CC = km.cluster_centers_            # cluster centroids 
    c  = cm.tab10(np.linspace(0, 1, nclust))  # color coding
    colors = [c[label % len(c)] for label in C]
    for i in range(len(CC)):
        ii = np.where(C==i)[0]
        DD,EE = D[ii],E[ii]
    plt.scatter(E,D,s=5,c=colors)
    plt.show()
    
